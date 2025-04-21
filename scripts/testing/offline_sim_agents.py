import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import logging 
import tarfile
import glob
import math
import os
import multiprocessing
from tqdm import tqdm
import pickle
import argparse

import torch
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2, sim_agents_metrics_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import estimators

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils import trajectory_utils
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Helper functions for smoothing and outlier handling
import pickle
import math
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features

def split_lists(all_shards, shard_suffixes, n):
    # 计算每个子列表的大小
    shard_size = math.ceil(len(all_shards) / n)

    # 分割 all_shards 和 shard_suffixes 成 n 个子列表
    shard_chunks = [all_shards[i * shard_size : (i + 1) * shard_size] for i in range(n)]
    suffix_chunks = [shard_suffixes[i * shard_size : (i + 1) * shard_size] for i in range(n)]

    return shard_chunks, suffix_chunks

def save_to_pkl(shard_chunks, suffix_chunks, file_path):
    with open(file_path, 'wb') as f:
        # 保存两个列表到 pkl 文件
        pickle.dump((shard_chunks, suffix_chunks), f)

def smooth_window(arr, window_size):
    """
    Smooth a 1D array using a simple moving average.
    :param arr: 1D array of values (e.g., x, y, or heading).
    :param window_size: Size of the moving average window.
    :return: Smoothed array.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(arr, window, 'valid')
    return np.concatenate([arr[:int(window_size/2)], res, arr[-int(window_size/2):]])

# Metric class
class SimAgentsMetric:
    def __init__(self, name: str = 'sim_agents_metric'):
        self._name = name
    
    @staticmethod
    def joint_scene_from_states(states: tf.Tensor, object_ids: tf.Tensor) -> sim_agents_submission_pb2.JointScene:
        states = states.numpy()
        simulated_trajectories = []
        for i_object in range(len(object_ids)):
            simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
                center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
                object_id=int(object_ids[i_object])
            ))
        return sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)

    @staticmethod
    def scenario_rollouts_from_states(scenario: scenario_pb2.Scenario, states: tf.Tensor, object_ids: tf.Tensor) -> sim_agents_submission_pb2.ScenarioRollouts:
        joint_scenes = []
        for i_rollout in range(states.shape[0]):
            joint_scenes.append(SimAgentsMetric.joint_scene_from_states(states[i_rollout], object_ids))
        return sim_agents_submission_pb2.ScenarioRollouts(joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)

    @staticmethod
    def linear_interp_and_expand_any_freq(hv_10Hz, h_valid_10Hz, pv_nHz, p_valid_nHz, exp_ratio):
        assert h_valid_10Hz[-1]
        l_final = exp_ratio + 1
        if p_valid_nHz:
            interp_out = np.zeros((l_final, 4))
            interp_out[:, :3] = np.linspace(hv_10Hz[-1, :3], pv_nHz[:3], l_final)
            theta1, theta2 = hv_10Hz[-1, 3], pv_nHz[3]
            V1 = np.array([np.cos(theta1), np.sin(theta1)])
            V2 = np.array([np.cos(theta2), np.sin(theta2)])
            V_interp = np.linspace(V1, V2, l_final)
            V_norm = V_interp / np.linalg.norm(V_interp, axis=1, keepdims=True)
            theta_interp = np.arctan2(V_norm[:, 1], V_norm[:, 0])
            interp_out[:, 3] = theta_interp
            interp_out = interp_out[1:, :]
            new_interp_hf = np.zeros((exp_ratio)).astype(bool)
        else:
            if h_valid_10Hz[-2]:
                v = hv_10Hz[-1, :] - hv_10Hz[-2, :]
            else:
                v = 0
            interp_out = np.zeros((exp_ratio, 4))
            for dt in range(exp_ratio):
                interp_out[dt, :] = hv_10Hz[-1, :] + v * (dt + 1)
            new_interp_hf = np.ones((exp_ratio)).astype(bool)
        return interp_out, new_interp_hf

    @staticmethod
    def extract_map_points(waymo_scenario):
        map_points = []
        for map_point in waymo_scenario.map_features:
            for polyline in map_point.road_line.polyline:
                map_points.append([polyline.x, polyline.y, polyline.z])
        map_points = np.array(map_points)
        return map_points

    @staticmethod
    def compute_single_speed_interp_any_freq_smoothing(path, pred_freq=2, return_metrics=True, query_agent=-1):

        if not return_metrics:
            pred_path = path
        else:
            pred_path, output_dir = path
        
        with open(pred_path, 'rb') as f:
            sim_agents_rollouts = pickle.load(f)
            all_lf_values = sim_agents_rollouts['values']
            all_lf_valid = sim_agents_rollouts['valid']
            waymo_scenario_paths = sim_agents_rollouts['waymo_scenario_paths']
            local_to_global_transforms = sim_agents_rollouts['transform']
        with open(waymo_scenario_paths, 'rb') as f:
            waymo_scenario = pickle.load(f)
        evaluated_sim_agent_ids = tf.convert_to_tensor(
            submission_specs.get_evaluation_sim_agent_ids(waymo_scenario)
        )
        logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(waymo_scenario)
        agent_ids = submission_specs.get_sim_agent_ids(waymo_scenario)
        logged_trajectories = logged_trajectories.gather_objects_by_id(
            tf.convert_to_tensor(agent_ids))
        logged_trajectories = logged_trajectories.slice_time(
            start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
        logged_trajectories_x = logged_trajectories.x.numpy()
        logged_trajectories_y = logged_trajectories.y.numpy()
        logged_trajectories_z = logged_trajectories.z.numpy()
        logged_trajectories_heading = logged_trajectories.heading.numpy()
        logged_trajectories_states = np.stack([
            logged_trajectories_x, logged_trajectories_y, logged_trajectories_z, logged_trajectories_heading],
            axis=-1)
        logged_trajectories_valid = logged_trajectories.valid.numpy()
        map_points = SimAgentsMetric.extract_map_points(waymo_scenario)
        zvalue_regressor = KNeighborsRegressor(n_neighbors=4)
        try:
            zvalue_regressor.fit(map_points[:, :2], map_points[:, 2])
        except:
            psudo_map_points = np.vstack([
                logged_trajectories_x.ravel(), 
                logged_trajectories_y.ravel(),
                logged_trajectories_z.ravel(),
                ]).T
            zvalue_regressor.fit(psudo_map_points[:, :2], psudo_map_points[:, 2])
        N_worlds, N_agents, N_steps, N_dims = 32, len(agent_ids), 80, 4
        N_pred_frames = 8 * pred_freq + 1
        preds = np.zeros((N_worlds, N_agents, N_steps, N_dims))
        is_interp = np.zeros((N_worlds, N_agents, N_steps)).astype(bool)
        missing_agent = []
        for ri in range(N_worlds):
            missing_agent_ri = []
            for agent_i, _ in enumerate(agent_ids):
                values = all_lf_values[ri][agent_i].numpy()
                values[:, 3] = values[:, 3] % (2 * np.pi)
                valid = all_lf_valid[ri][agent_i].numpy().astype(bool)
                missing_agent_ri.append(~valid[0])
                l2g_mat = local_to_global_transforms.numpy()
                xys = values[:, :2].reshape(-1, 2).T
                xys = xys.astype(l2g_mat.dtype)
                xys = l2g_mat[:2, :2] @ xys + l2g_mat[:2, -1:]
                values[:, :2] = xys.T
                theta = np.arctan2(l2g_mat[1, 0], l2g_mat[0, 0])
                values[:, 3] += theta
                z_values = zvalue_regressor.predict(values[:, :2])
                assert logged_trajectories_states[agent_i].shape[0] == 11
                logz = logged_trajectories_states[agent_i][10][2]
                knn_z = z_values[0]
                values[:, 2] = z_values - knn_z + logz
                values_hf = np.zeros((91, 4))
                values_hf[:11, :] = np.array(logged_trajectories_states[agent_i])
                valid_hf = np.zeros((91)).astype(bool)
                valid_hf[:11] = np.array(logged_trajectories_valid[agent_i])
                is_interp_hf = np.zeros((91)).astype(bool)
                assert 10 % pred_freq == 0
                frame_expand_ratio = 10 // pred_freq
                for i in range(1, N_pred_frames):
                    n_hist = 11 + frame_expand_ratio * (i - 1)
                    hist_values = values_hf[:n_hist, :]
                    hist_valid = valid_hf[:n_hist]
                    pred_value = values[i]
                    pred_valid = valid[i]
                    new_values, new_interp_hf = SimAgentsMetric.linear_interp_and_expand_any_freq(
                        hist_values, hist_valid, pred_value, pred_valid, exp_ratio=frame_expand_ratio)
                    n_future = n_hist + frame_expand_ratio
                    values_hf[n_hist:n_future, :] = new_values
                    valid_hf[n_hist:n_future] = True
                    is_interp_hf[n_hist:n_future] = new_interp_hf
                preds[ri, agent_i] = values_hf[11:, :]
                is_interp[ri, agent_i] = is_interp_hf[11:]
            missing_agent.append(missing_agent_ri)
        missing_agent = np.array(missing_agent)
        if not query_agent < 0:
            missing_agent = np.ones_like(missing_agent)
            missing_agent[:, query_agent] = 0
        missing_agent_rate = missing_agent.sum() / len(missing_agent.reshape(-1))
        smoothing_factors = [9, 9, 1, 9, 9]
        smoothed_trajectory = np.zeros(list(preds.shape[:3]) + [5])
        smoothed_trajectory[..., :3] = preds[..., :3]
        smoothed_trajectory[..., 3] = np.sin(preds[..., 3])
        smoothed_trajectory[..., 4] = np.cos(preds[..., 3])
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                for dim in range(preds.shape[3]):
                    if smoothing_factors[dim] > 1:
                        smoothed_trajectory[i, j, :, dim] = smooth_window(smoothed_trajectory[i, j, :, dim], smoothing_factors[dim])
        smoothed_trajectory[..., 3] = np.arctan2(smoothed_trajectory[..., 3], smoothed_trajectory[..., 4])
        simulated_states = tf.convert_to_tensor(smoothed_trajectory[..., :4])
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # pkl_out_path = os.path.join(output_dir, pred_path.split('/')[-1])
        # with open(pkl_out_path, 'wb') as f:
        #     pickle.dump(simulated_states, f)
        scenario_rollouts = SimAgentsMetric.scenario_rollouts_from_states(
            waymo_scenario, simulated_states, logged_trajectories.object_id)
        submission_specs.validate_scenario_rollouts(scenario_rollouts, waymo_scenario)
        if not return_metrics:
            return scenario_rollouts
        config = metrics.load_metrics_config()
        scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
            config, waymo_scenario, scenario_rollouts)
        scenario_result = [scenario_metrics.scenario_id, scenario_metrics.metametric,
                           scenario_metrics.linear_speed_likelihood,
                           scenario_metrics.linear_acceleration_likelihood, scenario_metrics.angular_speed_likelihood,
                           scenario_metrics.angular_acceleration_likelihood, scenario_metrics.distance_to_nearest_object_likelihood,
                           scenario_metrics.collision_indication_likelihood, scenario_metrics.time_to_collision_likelihood,
                           scenario_metrics.distance_to_road_edge_likelihood, scenario_metrics.offroad_indication_likelihood,
                           scenario_metrics.average_displacement_error, scenario_metrics.min_average_displacement_error,
                           scenario_metrics.simulated_collision_rate,scenario_metrics.simulated_offroad_rate]
        return scenario_result

    @staticmethod
    def pack_shared(shard, shard_suffix, submission_info, output_dir):
        scenario_rollouts = []
        for p in shard:
            sr = SimAgentsMetric.compute_single_speed_interp_any_freq_smoothing(path=p, pred_freq=2, return_metrics=False, query_agent=-1)
            scenario_rollouts.append(sr)

        # Now that we have few `ScenarioRollouts` for this shard, we can package them into a `SimAgentsChallengeSubmission`.
        # Remember to populate the metadata for each shard.
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(scenario_rollouts=scenario_rollouts, **submission_info)

        # Now we can export this message to a binproto, saved to local storage.
        output_filename = f'submission.binproto{shard_suffix}'
        with open(os.path.join(output_dir, output_filename), 'wb') as f:
            f.write(shard_submission.SerializeToString())
        return output_filename
    
    @staticmethod
    def pack_shared_wrapper(args):
            return SimAgentsMetric.pack_shared(*args)

    @staticmethod
    def process_with_multiprocessing(all_shards, shard_suffixes, num_processes, submission_info, output_dir, use_multiprocessing=False):

        SimAgentsMetric.pack_shared = partial(SimAgentsMetric.pack_shared, submission_info=submission_info, output_dir=output_dir)
        # SimAgentsMetric.pack_shared_wrapper((all_shards[0], shard_suffixes[0]))
        if use_multiprocessing:
            with multiprocessing.Pool(num_processes) as pool:
                results = list(tqdm(pool.imap(SimAgentsMetric.pack_shared_wrapper, zip(all_shards, shard_suffixes)), total=len(shard_suffixes)))
        else:
            results = []
            for shard, shard_suffix in tqdm(zip(all_shards, shard_suffixes), total=len(shard_suffixes)):
                results.append(SimAgentsMetric.pack_shared(shard, shard_suffix))
        return results

    def pack_submission_single_machine(self, shards_and_suffixes_path, shard_idx, output_dir, submission_info, shared_size=16, num_processes=32):
        with open(shards_and_suffixes_path, 'rb') as f:
            all_shards, shard_suffixes = pickle.load(f)
        shards = all_shards[shard_idx - 1]
        shard_suffixes = shard_suffixes[shard_idx -1]
        output_filenames = SimAgentsMetric.process_with_multiprocessing(shards, shard_suffixes, num_processes, submission_info, output_dir)
        return output_filenames 

    def shard_lists(self, list1, list2, n):

        if len(list1) != len(list2):
            raise ValueError("两个列表必须具有相同的长度")

        # 计算每一份的大小
        total_length = len(list1)
        shard_size = total_length // n

        # 初始化结果
        all_shards = []
        shard_suffixes = []

        # 切分列表
        for i in range(n):
            start_index = i * shard_size
            if i == n - 1:  # 最后一块可能包含剩余的元素
                end_index = total_length
            else:
                end_index = start_index + shard_size

            # 切分并添加到结果
            shard_list1 = list1[start_index:end_index]
            shard_list2 = list2[start_index:end_index]
            all_shards.append(shard_list1)
            shard_suffixes.append(shard_list2)

        return all_shards, shard_suffixes

    def check_save_missing_shards(self, shards_and_suffixes_path, output_dir, submission_info, shared_size=16, num_processes=32):
        with open(shards_and_suffixes_path, 'rb') as f:
            all_shards, shard_suffixes = pickle.load(f)
        output_filenames = []
        
        new_all_shards, new_shard_suffixes = [], []
        for i, (shards, shard_suffixes) in tqdm(enumerate(zip(all_shards, shard_suffixes))):
            
            for shard, shard_suffix in zip(shards, shard_suffixes):
                output_filename = f'submission.binproto{shard_suffix}'
                if not os.path.exists(os.path.join(output_dir, output_filename)):
                    # print(f"Processing shard with suffix {shard_suffix}")
                    # output_filename = SimAgentsMetric.pack_shared(shard, shard_suffix, submission_info, output_dir)
                    # output_filenames.append(output_filename)

                    new_all_shards.append(shard)
                    new_shard_suffixes.append(shard_suffix)
        
        # new_all_shards, new_shard_suffixes = self.shard_lists(new_all_shards, new_shard_suffixes, 10)           
        # with open(os.path.join(output_dir, 'new_shards_and_suffixes.pkl'), 'wb') as f:
        #     pickle.dump((new_all_shards, new_shard_suffixes), f)
        # return output_filenames

                    # for file in shard:
                    #     try:
                    #         with open(file, 'rb') as f:
                    #             sim_agents_rollouts = pickle.load(f)
                    #     except:
                    #         output_filenames.append(file)
            
        return output_filenames

    def compute_final_submission(self, save_dir, output_dir=None):
        output_dir = save_dir if not output_dir else output_dir
        output_files = glob.glob(os.path.join(save_dir, '*.binproto*'))
        
        submission_tar = os.path.join(output_dir, 'submission.tar.gz')
        with tarfile.open(submission_tar, 'w:gz') as tar:
            for output_filename in tqdm(output_files, total=len(output_files)):
                tar.add(output_filename,arcname=os.path.basename(output_filename))

        return submission_tar

    def pack_submission(self, pkl_dir, output_dir, submission_info, shared_size=16, num_processes=32):
        all_preds = glob.glob(os.path.join(pkl_dir, '*.pkl'))
        all_shards = []
        shard_suffixes = []
        N_shard = math.ceil(len(all_preds) / shared_size)
        assert N_shard < 99999, 'too many shards!'
        total_number_str = str(N_shard).zfill(5)

        for shard_i in range(N_shard):
            shard = all_preds[shard_i * shared_size : (shard_i+1) * shared_size]
            index_str = str(shard_i+1).zfill(5)
            shard_suffix = f"-{index_str}-of-{total_number_str}"
            all_shards.append(shard)
            shard_suffixes.append(shard_suffix)
        
        # shard_chunks, suffix_chunks = split_lists(all_shards, shard_suffixes, 10)
        # save_to_pkl(shard_chunks, suffix_chunks, 'shards_and_suffixes.pkl')
        # return

        output_filenames = SimAgentsMetric.process_with_multiprocessing(all_shards, shard_suffixes, num_processes, submission_info, output_dir)

        # Once we have created all the shards, we can package them directly into a
        # tar.gz archive, ready for submission.
        submission_tar = os.path.join(output_dir, 'submission.tar.gz')
        with tarfile.open(submission_tar, 'w:gz') as tar:
            for output_filename in output_filenames:
                tar.add(os.path.join(output_dir, output_filename),arcname=output_filename)

        return submission_tar

    def compute(self, pkl_dir, output_dir, num_processes, use_multiprocessing=False):
        if '.csv' in pkl_dir:
            import pandas as pd
            all_preds = pd.read_csv(pkl_dir)['file_name'].tolist()
        else:
            all_preds = sorted(glob.glob(os.path.join(pkl_dir, '*_rollouts.pkl')))
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
        paths = [(pred, output_dir) for pred in all_preds]

        # you can use a subset of the data for debugging 
        # import random
        # paths = random.sample(paths, 1000)

        if use_multiprocessing:
            with multiprocessing.Pool(num_processes) as pool:
                results = list(tqdm(pool.imap(SimAgentsMetric.compute_single_speed_interp_any_freq_smoothing, paths), total=len(paths)))
        else:
            results = []
            metric_names = [
                    'metametric', 'linear_speed_likelihood',
                    'linear_acceleration_likelihood', 'angular_speed_likelihood',
                    'angular_acceleration_likelihood', 'distance_to_nearest_object_likelihood',
                    'collision_indication_likelihood', 'time_to_collision_likelihood',
                    'distance_to_road_edge_likelihood', 'offroad_indication_likelihood', 'average_displacement_error',
                    'min_average_displacement_error', 'collision_rate','offroad_rate'
                ]
            batch_size = 50
            results = []
            for i in tqdm(range(0, len(paths), batch_size)):
                batch_paths = paths[i:i + batch_size]
                for path in batch_paths:
                    try:
                        result = SimAgentsMetric.compute_single_speed_interp_any_freq_smoothing(path)
                    except Exception as e:
                        continue
                    results.append(result)
                    tqdm.write(f"Current result for {result[0]}: {dict(zip(metric_names, result[1:]))}")
                if results:
                    batch_numerical_results = [np.array(r[1:]) for r in results]
                    batch_numerical_results = np.vstack(batch_numerical_results)
                    batch_mean_results = batch_numerical_results.mean(axis=0)
                    tqdm.write(f"Acc {len(results)} meand results: {dict(zip(metric_names, batch_mean_results))}")
        
        numerical_results = [np.array(r[1:]) for r in results]
        numerical_results = np.vstack(numerical_results)
        # low_score_mask = numerical_results[:, 0] < 0.25
        # low_score_results = [res for res, is_low in zip(all_preds, low_score_mask) if is_low]
        # with open('low-score-results.pkl', 'wb') as f:
        #     pickle.dump(low_score_results, f)
        final_results = numerical_results.mean(axis=0)
        # with open('full-results.pkl', 'wb') as f:
        #     pickle.dump(final_results, f)
        entries = [
            'metametric',  
            'linear_speed_likelihood',
            'linear_acceleration_likelihood',
            'angular_speed_likelihood',
            'angular_acceleration_likelihood',
            'distance_to_nearest_object_likelihood',
            'collision_indication_likelihood',
            'time_to_collision_likelihood',
            'distance_to_road_edge_likelihood',
            'offroad_indication_likelihood',
            'average_displacement_error',
            'min_average_displacement_error',
            'collision_rate',
            'offroad_rate'
        ]
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        for entry, v in zip(entries, final_results):
            print('{}: \t{}'.format(entry, v))
        
        with open(os.path.join(output_dir, pkl_dir.split('/')[-1][:-4] + '_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        return

def main():
    args = argparse.ArgumentParser()
    
    args.add_argument('--pkl_dir', type=str, default='/cpfs01/user/yenaisheng/validation') # pkl directory where the rollouts are stored
    args.add_argument('--output_dir', type=str, default='/cpfs01/user/yenaisheng/res') # output directory
    args.add_argument('--nprocess', type=int, default=32) # number of processes to use
    args.add_argument('--shared_size', type=int, default=16) # how many pkl files in one shard

    args = args.parse_args()

    offline_simulation = True # True for offline simulation, False for online submission
    
    if offline_simulation:
        # the following lines are used for offline simulation 
        # please note that this only works for evaluation set as we have access to the ground truth in only this case
        pkl_dir = args.pkl_dir
        output_dir = args.output_dir
        nprocess = args.nprocess
        metric = SimAgentsMetric()    
        metric.compute(pkl_dir, output_dir, nprocess)
    
    else:
        # the following lines are used for online submission and simulation 
        # this works for both evaluation and testing set, but packaging and submitting to offical website can be time-consuming.
        pkl_dir = args.pkl_dir
        output_dir = args.output_dir
        nprocess = args.nprocess
        shared_size = args.shared_size
        # modify the submission info to your own
        submission_info = dict(
        submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
        account_name='your_email',
        unique_method_name='your_model_name',
        authors=['author1', 'author2'],
        affiliation='your_affiliation',
        description='description of your method',
        method_link='url_to_your_method',

        uses_lidar_data=False,
        uses_camera_data=False,
        uses_public_model_pretraining=False,
        num_model_parameters='your_model_size',
        acknowledge_complies_with_closed_loop_requirement=True
        )
        
        metric = SimAgentsMetric()
        metric.pack_submission(pkl_dir, output_dir, submission_info, shared_size=shared_size, num_processes=nprocess)

if __name__ == '__main__':
    main()