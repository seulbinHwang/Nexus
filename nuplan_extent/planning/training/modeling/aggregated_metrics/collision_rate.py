from typing import List

import torch
from nuplan.planning.training.modeling.types import TargetsType

from torchmetrics import Metric

from nuplan_extent.planning.training.preprocessing.feature_builders.nexus_feature_builder import (
    decode_scene_tensor,
)
from nuplan_extent.planning.training.modeling.models.tokenizers.base_tokenizer_utils import (
    check_collision,
)
from numba import jit, prange
import numpy as np

class CollisionMetric(Metric):
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
        super(CollisionMetric, self).__init__()
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
        self.collision_at_t: torch.Tensor
        self.add_state("collision_at_t", default=torch.zeros(n_future_timesteps))
        self.add_state(
            "entries_at_t",
            default=torch.zeros(n_future_timesteps),
        )

    def name(self) -> str:
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        return []

    def update(self, predictions: TargetsType, targets: TargetsType) -> None:
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

        gt = decode_scene_tensor(gt)
        pred = decode_scene_tensor(pred)

        self.entries_at_t += valid.sum(axis=(0, 1))  # NT

        collied_agents = self.calculate_collision_rate_batch(pred.cpu().numpy(), valid.cpu().numpy(), self._use_only_ego)
        collied_agents = torch.tensor(collied_agents).to(self.collision_at_t.device)
        self.collision_at_t += collied_agents
       

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
        for time_idx, time in zip(self.times_idx, self.times):
            out[f"collision@{time:1f}s"] = (
                self.collision_at_t[..., time_idx].sum()
                / self.entries_at_t[..., time_idx].sum()
            )
        sum_ = 0
        
        for k,v in out.items():
            sum_ += v
        out["collision@avg"] = sum_/len(out)

        return out

    def log(self, logger, data: dict):
        if not data:
            return
        prefix = f"aggregated_metrics/{self._name}/"
        for k, v in data.items():
            logger(prefix + k, v.detach().cpu().item())

    def calculate_collision_rate_batch(self, predicted_tokenized_array, valid, ego_only=False):
        collision_list = []
        for t in range(valid.shape[-1]):
            collision_rate = 0
            for i in range(len(predicted_tokenized_array)):
                collision_rate += self.calculate_collision_rate(predicted_tokenized_array[i,:,t], valid[i,:,t], ego_only)
            collision_list.append(collision_rate)
        return collision_list

    def calculate_collision_rate(self, predicted_tokenized_array, valid, ego_only=False, filter_dist=10):
        predicted_tokenized_array = predicted_tokenized_array[valid==1]
        collied_agents = 0
        if ego_only:
            agent_num = 1
        else:
            agent_num = len(predicted_tokenized_array)
        for token_index in range(agent_num):
            current_token = predicted_tokenized_array[token_index]
            surrounding_agents_array = predicted_tokenized_array[~np.isin(np.arange(len(predicted_tokenized_array)), token_index)]
            for other_token in surrounding_agents_array:

                xa, ya, cos_a, sin_a, _, _, la, wa = current_token
                xb, yb, cos_b, sin_b, _, _, lb, wb = other_token
                ha, hb = np.arctan2(sin_a, cos_a), np.arctan2(sin_b, cos_b)

                # prefailter
                if (xa - xb) ** 2 + (ya - yb) ** 2 > filter_dist ** 2:
                    continue
                is_collided = check_collision(np.array([xa, ya, ha, wa, la]), np.array([xb, yb, hb, wb, lb]))
                if is_collided:
                    collied_agents += 1
        if not ego_only:
            # the collision is counted twice for each pair of agents
            collied_agents /= 2
        return collied_agents