from typing import List
import os

import torch
import torch.nn.functional as F
import numpy as np
import pickle

from torchmetrics import Metric
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils
from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType

import logging
logger = logging.getLogger(__name__)
# import matplotlib
# matplotlib.use('Agg')  # Use 'Agg' for non-interactive plots
# import matplotlib.pyplot as plt


class SimAgentsMetric(Metric):
    """
    Metric representing the probability density of GT agents trajectories over the distribution predicted by model
    """

    def __init__(self, 
                 name: str = 'sim_agents_metric',
                 basepath: str = '/tmp'
                 ) -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        super(SimAgentsMetric, self).__init__()
        self._name = name  
        self.basepath = basepath 
        logger.info('inference basepath: {}'.format(self.basepath))
        self.all_scenario_id = []
        self.all_metametric = []

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["sim_agents"]
    
    @staticmethod
    def joint_scene_from_states(
            states: tf.Tensor, object_ids: tf.Tensor
            ) -> sim_agents_submission_pb2.JointScene:
        states = states.numpy()
        simulated_trajectories = []
        for i_object in range(len(object_ids)):
            simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
                center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
                object_id=int(object_ids[i_object])
            ))
        return sim_agents_submission_pb2.JointScene(
            simulated_trajectories=simulated_trajectories)

    @staticmethod
    def scenario_rollouts_from_states(
            scenario: scenario_pb2.Scenario, 
            states: tf.Tensor, object_ids: tf.Tensor,
            ) -> sim_agents_submission_pb2.ScenarioRollouts:
        joint_scenes = []
        for i_rollout in range(states.shape[0]):
            joint_scenes.append(SimAgentsMetric.joint_scene_from_states(states[i_rollout], object_ids))
        
        return sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)


    def update(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """     
        # batch of predictions, 
        # each prediction contains 32 parallel rollouts, 
        # each rollout contains a prediction of agents trajectories 8s into future (16 frames @ 2Hz)
        sim_agents_rollouts = predictions['sim_agents_rollouts']  
        agents_map = predictions['agents_map']

        bs = len(sim_agents_rollouts)

        # batch of targets
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        waymo_scenario_paths = targets['sim_agents'].pkl_path
        local_to_global_transforms = targets['sim_agents'].local_to_global_transform

        n_frames = 16+1 # 8s @ 2Hz future + 1 current frame
        # update part
        for bi in range(bs):            
            
            # process GT part
            with open(waymo_scenario_paths[bi], 'rb') as f:
                waymo_scenario = pickle.load(f)
            basepath = self.basepath
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            target_agent_idx = int(targets['sim_agents'].agent_idx[bi])
            scene_path = os.path.join(basepath, '{}_{}_rollouts.pkl'.format(waymo_scenario.scenario_id, target_agent_idx))
            agent_ids = submission_specs.get_sim_agent_ids(waymo_scenario)
            # process pred part
            parallel_rollouts = sim_agents_rollouts[bi]  # 32 rollouts
            all_lf_values = []
            all_lf_valid = []
            # create a raw_id -> agent index list map
            map_agent = {}
            for index, agent_id in enumerate(agents_map[bi]):
                agent_id = int(agent_id)
                if agent_id == -1:
                    break 
                map_agent[agent_id] = index

            for ri, rollout in enumerate(parallel_rollouts):
                agent_values = []
                agent_valid = []
                # import time
                for agent_i, idx in enumerate(agent_ids):
                    if not idx in map_agent:
                        print('Missing pred for agent: {} {}, all agents: {}'.format(idx, waymo_scenario.scenario_id, map_agent.keys()))
                        agent_values.append(torch.zeros(n_frames, 6).cpu())
                        agent_valid.append(torch.zeros(n_frames).cpu())
                        continue

                    agent_pred = rollout[map_agent[idx]]  # timestep,features(x,y,cos(yaw),sin(yaw),vx,vy,l,w)
                    # nHz mask and values
                    values = torch.zeros(n_frames, 6) # x,y,z,yaw,width,length
                    valid = torch.ones(n_frames)
                    values[:, 0] = agent_pred[4:,0] # x
                    values[:, 1] = agent_pred[4:,1] # y
                    values[:, 3] = torch.arctan2(agent_pred[4:,3], agent_pred[4:,2]) # yaw
                    values[:, 4] = float(agent_pred[4,-1]) # width 
                    values[:, 5] = float(agent_pred[4,-2]) # length

                    agent_values.append(values.cpu())
                    agent_valid.append(valid.cpu())
                
                # import pdb; pdb.set_trace()
                all_lf_values.append(agent_values)
                all_lf_valid.append(agent_valid)
            
            with open(scene_path, 'wb') as f:
                pickle.dump({'values': all_lf_values, 'valid': all_lf_valid, 
                             'waymo_scenario_paths': waymo_scenario_paths[bi], 'transform': local_to_global_transforms[bi].cpu()}, f)
            if not os.path.exists(scene_path):
                print('Failed to save scene path: {}'.format(scene_path))
        return

    
    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric.

        :return: metric scalar tensor
        """
        
        
        result = np.array(self.all_metametric).mean()
        
        return result        
    

    def log(self, logger, data):
        pass