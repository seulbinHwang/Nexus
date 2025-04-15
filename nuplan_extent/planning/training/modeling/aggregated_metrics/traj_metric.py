from typing import List

import torch
from nuplan.planning.training.modeling.types import TargetsType

from torchmetrics import Metric

from nuplan_extent.planning.training.preprocessing.feature_builders.nexus_feature_builder import (
    decode_scene_tensor,unnormalize_roadgraph
)
from nuplan_extent.planning.training.modeling.models.tokenizers.base_tokenizer_utils import (
    check_collision,
)
from numba import jit, prange
import numpy as np
import math
import torch

def mertic_AAE(pred_traj: torch.tensor):
    """
    Average Angular Expansion ⬆️
    AAE=\frac{1}{N} \sum_{i=1}^{N} \frac{\sum_{t=1}^{T-1}\left|\theta_{t+1}-\theta_{t}\right|}{T-1}
    """
    traj_angles = torch.atan2(pred_traj[..., 3], pred_traj[..., 2])
    angle_diff = torch.abs(traj_angles[..., 1:] - traj_angles[..., :-1])
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    AAE = torch.abs(angle_diff).mean()*180/math.pi
    return AAE


def mertic_ASV(pred_traj: torch.tensor, delta_t=0.5):
    """
    Average Speed Variation ⬆
    ASV=\frac{1}{N} \sum_{i=1}^{N} \frac{\sum_{t=1}^{T-1}\left|v_{t+1}-v_{t}\right|}{T-1}
    """
    delta = pred_traj[..., 1:, :2] - pred_traj[..., :-1, :2]
    delta = (delta[...,0]**2 + delta[...,1]**2)**0.5
    velocities = delta / delta_t
    delta_v = velocities[..., 1:] - velocities[..., :-1]
    ASV = torch.abs(delta_v).mean()
    return ASV


def mertic_comfort(pred_traj: torch.tensor, delta_t=0.5):
    """
    \text{LonAcc}=\frac{1}{T} \sum_{t=1}^{T}\left|a_{\mathrm{lon}}^{t}\right|
    \text{LonJerk}=\frac{1}{T} \sum_{t=1}^{T}\left|\frac{d a_{\mathrm{lon}}^{t}}{d t}\right|
    \text{LatAcc}=\frac{1}{T} \sum_{t=1}^{T}\left|a_{\text {lat }}^{t}\right|
    \text{LatJerk}=\frac{1}{T} \sum_{t=1}^{T}\left|\frac{d a_{\mathrm{lat}}^{t}}{d t}\right|
    """
    delta = pred_traj[..., 1:, :2] - pred_traj[..., :-1, :2]
    delta = (delta[...,0]**2 + delta[...,1]**2)**0.5
    velocities = delta / delta_t
    delta_v = velocities[..., 1:] - velocities[..., :-1]
    acceleration = delta_v / delta_t
    delta_a = acceleration[..., 1:] - acceleration[..., :-1]
    jerk = delta_a / delta_t

    traj_angles = torch.atan2(pred_traj[..., 3], pred_traj[..., 2])
    angle_delta = torch.abs(traj_angles[..., 1:] - traj_angles[..., :-1])
    angle_delta = (angle_delta + math.pi) % (2 * math.pi) - math.pi
    angle_velocities = angle_delta / delta_t
    angle_delta_v = angle_velocities[..., 1:] - angle_velocities[..., :-1]
    angle_acceleration = angle_delta_v / delta_t
    angle_delta_a = angle_acceleration[..., 1:] - angle_acceleration[..., :-1]
    angle_jerk = angle_delta_a / delta_t


    # lon acc
    acceleration_lon = acceleration # x
    # max_acc_lon = torch.max(acceleration_lon, dim=1)[0]
    max_acc_lon = acceleration_lon
    mean_acc_lon = torch.abs(max_acc_lon).mean()

    # lon jerk
    jerk_lon = jerk # x
    # max_jerk_lon = torch.max(jerk_lon, dim=1)[0]
    max_jerk_lon = jerk_lon
    mean_jerk_lon = torch.abs(max_jerk_lon).mean()

    # lat acc
    acceleration_lat = angle_acceleration # y
    # max_acc_lat = torch.max(acceleration_lat, dim=1)[0]
    max_acc_lat = acceleration_lat
    mean_acc_lat = torch.abs(max_acc_lat).mean()*180/math.pi

    # lat jerk
    jerk_lat = angle_jerk # x
    # max_jerk_lat = torch.max(jerk_lat, dim=1)[0]
    max_jerk_lat = jerk_lat
    mean_jerk_lat = torch.abs(max_jerk_lat).mean()*180/math.pi

    return mean_acc_lon, mean_jerk_lon, mean_acc_lat, mean_jerk_lat


def compute_min_distance_torch(scene_tensor, map_tensor, map_valid):
    """
    Compute the minimum Euclidean distance from each vehicle position to any valid lane point.

    Parameters:
        scene_tensor (torch.Tensor): [B, NA, NT, D] vehicle tensor (first two dims of D are x, y)
        map_tensor (torch.Tensor): [B, L, N, D] lane tensor (first two dims of D are x, y)
        map_valid (torch.Tensor): [B, L, N, D] boolean tensor indicating valid lane points (use first channel)

    Returns:
        min_dist (torch.Tensor): [B, NA, NT] tensor with the minimum distance from each vehicle at each timestep
    """
    B, NA, NT, D = scene_tensor.shape
    _, L, N, _ = map_tensor.shape

    # Extract x, y coordinates
    scene_xy = scene_tensor[..., :2]  # [B, NA, NT, 2]
    map_xy = map_tensor[..., :2]        # [B, L, N, 2]

    # Expand dims to broadcast: 
    # scene: [B, NA, NT, 1, 1, 2], map: [B, 1, 1, L, N, 2]
    scene_xy_exp = scene_xy.unsqueeze(3).unsqueeze(3)  # [B, NA, NT, 1, 1, 2]
    map_xy_exp = map_xy.unsqueeze(1).unsqueeze(1)        # [B, 1, 1, L, N, 2]

    # Compute Euclidean distances
    diff = scene_xy_exp - map_xy_exp                    # [B, NA, NT, L, N, 2]
    dist = torch.norm(diff, dim=-1)                     # [B, NA, NT, L, N]

    # Create a mask from map_valid (using the first channel) and set invalid points to a large value.
    map_valid_mask = map_valid[..., 0].bool()           # [B, L, N]
    map_valid_exp = map_valid_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L, N]
    dist = dist.masked_fill(~map_valid_exp, 1e6)

    # Get the minimum distance over lanes and points.
    min_dist = torch.min(torch.min(dist, dim=-1)[0], dim=-1)[0]  # [B, NA, NT]
    return min_dist

def metric_offroad_rate(scene_tensor, valid_mask, map_tensor, map_valid, lane_threshold=2.75, obs_frames=5):
    """
    Computes the off-road rate for vehicles based on their distances to lane lines.

    A vehicle is considered "in-lane" if at least one valid frame in the first obs_frames is within lane_threshold.
    If any valid frame after obs_frames has a distance greater than lane_threshold, the vehicle is marked as off-road.

    The off-road rate is computed as:
        off-road rate = (# vehicles that went off-road) / (# vehicles that were initially in-lane)

    Parameters:
        scene_tensor (torch.Tensor): [B, NA, NT, D] vehicle tensor (first two dims of D are x, y)
        valid_mask (torch.Tensor): [B, NA, NT] boolean tensor indicating valid vehicle frames
        map_tensor (torch.Tensor): [B, L, N, D] lane tensor (first two dims of D are x, y)
        map_valid (torch.Tensor): [B, L, N, D] boolean tensor indicating valid lane points
        lane_threshold (float): distance threshold for being "in-lane"
        obs_frames (int): number of initial frames for observing in-lane behavior

    Returns:
        offroad_rate (float): Ratio of vehicles that went off-road among those initially in-lane.
    """
    B, NA, NT, _ = scene_tensor.shape

    # Compute the minimum distance from each vehicle position to any lane point
    min_dist = compute_min_distance_torch(scene_tensor, map_tensor, map_valid)  # [B, NA, NT]

    in_lane = torch.zeros(B, NA, dtype=torch.bool, device=scene_tensor.device)
    offroad = torch.zeros(B, NA, dtype=torch.bool, device=scene_tensor.device)

    for b in range(B):
        for i in range(NA):
            # Get valid mask for vehicle i in batch b (shape [NT])
            valid_frames = valid_mask[b, i, :]  # Bool tensor
            if not torch.any(valid_frames[:obs_frames]):
                continue  # Skip if no valid observation in first obs_frames
            # Check if any valid frame in the observation period is within lane threshold
            if torch.any((min_dist[b, i, :obs_frames] <= lane_threshold) & valid_frames[:obs_frames].bool()):
                in_lane[b, i] = True
                # Check subsequent valid frames for off-road condition
                if torch.any((min_dist[b, i, obs_frames:] > lane_threshold) & valid_frames[obs_frames:].bool()):
                    offroad[b, i] = True

    num_in_lane = in_lane.sum().item()
    num_offroad = offroad.sum().item()
    num_valid = torch.any(valid_mask == 1, dim=-1).bool().sum().item()
    offroad_rate = num_offroad / num_valid if num_valid > 0 else 0.0
    count = 1. if num_valid > 0 else 0.0
    return offroad_rate, count


class TrajMetric(Metric):
    def __init__(
        self,
        name: str = "ADE",
        prediction_output_key: str = "behavior_pred",
        n_future_timesteps: int = 16,
        interval: int = 2,  # every one second, coupled with delta_t
        start_idx: int = 1,  # start at first second not 0.5
        delta_t: float = 0.5,  # time between steps
        ego_idx: int = 0,  # where the ego is in the scene tensor
        exclude_ego: bool = True,
        use_only_ego: bool = False,
    ) -> None:
        """
        Initializes the class.
        :param name: the name of the metric (used in logger)
        """
        super(TrajMetric, self).__init__()
        self._name = name
        self._nt = n_future_timesteps
        self._interval = interval
        self._delta_t = delta_t
        self._start_idx = start_idx
        self._key = prediction_output_key
        self._ego_idx = ego_idx
        self._exclude_ego = exclude_ego
        self._use_only_ego = use_only_ego
        assert not (exclude_ego and use_only_ego)
        assert n_future_timesteps % interval == 0

        self.times_idx = torch.arange(start_idx, n_future_timesteps, interval)
        self.times = (self.times_idx + 1) * delta_t
        self.entries_at_t: torch.Tensor
        self.add_state("entries_at_t", default=torch.zeros(1),)
        self.M_AAE: torch.Tensor
        self.add_state("M_AAE", default=torch.zeros(1),)
        self.M_ASV: torch.Tensor
        self.add_state("M_ASV", default=torch.zeros(1),)
        self.M_LonACC: torch.Tensor
        self.add_state("M_LonACC", default=torch.zeros(1),)
        self.M_LonJerk: torch.Tensor
        self.add_state("M_LonJerk", default=torch.zeros(1),)
        self.LatAcc: torch.Tensor
        self.add_state("LatAcc", default=torch.zeros(1),)
        self.LatJerk: torch.Tensor
        self.add_state("LatJerk", default=torch.zeros(1),)
        self.add_state("offroad_rate", default=torch.zeros(1),)
        self.add_state("count", default=torch.zeros(1))
        
        

    def name(self) -> str:
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        return []

    def update(self, predictions: TargetsType, targets: TargetsType) -> None:
        # import pdb; pdb.set_trace()
        if "behavior_prediction_out" not in predictions:
            return

        if self._key not in predictions["behavior_prediction_out"]:
            return

        valid = predictions["behavior_prediction_out"]["valid_mask"][
            :, :, -self._nt :
        ]  # B, NA, NT
        gt = predictions["behavior_prediction_out"]["scene_tensor_gt"][
            :, :, -self._nt :
        ]
        pred = predictions["behavior_prediction_out"][self._key][:, :, -self._nt :]

            
        road_graph = predictions["behavior_prediction_out"]["road_graph"]
        road_graph_validity = predictions["behavior_prediction_out"]["road_graph_validity"]
        # if road_graph is all zeros, then the scene is offroad
            
        offroad_rate,count = metric_offroad_rate(decode_scene_tensor(predictions["behavior_prediction_out"][self._key]),
                                                    predictions["behavior_prediction_out"]["valid_mask"],
                                                    unnormalize_roadgraph(road_graph), 
                                                    road_graph_validity)
        self.offroad_rate += offroad_rate
        self.count += count
        # print(f"offroad_rate: {self.offroad_rate}, count: {self.count}, offroad_rate/count: {self.offroad_rate/self.count if self.count != 0 else 999}")
        gt = decode_scene_tensor(gt)
        pred = decode_scene_tensor(pred)

        index_matrix = torch.all(valid == 1, dim=-1).bool()
        pred_traj = pred[index_matrix]

        traj_sum = index_matrix.sum()
        self.M_AAE += mertic_AAE(pred_traj)*traj_sum
        self.M_ASV += mertic_ASV(pred_traj)*traj_sum
        M_Comfort = mertic_comfort(pred_traj)
        self.M_LonACC, self.M_LonJerk, self.LatAcc, self.LatJerk = self.M_LonACC+M_Comfort[0]*traj_sum, self.M_LonJerk+M_Comfort[1]*traj_sum, self.LatAcc+M_Comfort[2]*traj_sum, self.LatJerk+M_Comfort[3]*traj_sum

        self.entries_at_t += traj_sum

    def compute(self) -> dict:
        """
        Computes the metric.
        :return: metric scalar tensor
        """
        out = {}
        if self.entries_at_t.sum() == 0:
            return out
        # note that the attributes are tensors, but have been addded an extra dimention corresponding to the number of
        # gpus
        out["M_AAE"] = self.M_AAE.sum() / self.entries_at_t.sum()
        out["M_ASV"] = self.M_ASV.sum() / self.entries_at_t.sum()
        out["M_LonACC"] = self.M_LonACC.sum() / self.entries_at_t.sum()
        out["M_LonJerk"] = self.M_LonJerk.sum() / self.entries_at_t.sum()
        out["LatAcc"] = self.LatAcc.sum() / self.entries_at_t.sum()
        out["LatJerk"] = self.LatJerk.sum() / self.entries_at_t.sum()
        out["offroad_rate"] = self.offroad_rate.sum() / self.count.sum()
        print(out["offroad_rate"])
        return out

    def log(self, logger, data: dict):
        if not data:
            return
        prefix = f"aggregated_metrics/{self._name}/"
        for k, v in data.items():
            logger(prefix + k, v.detach().cpu().item())

