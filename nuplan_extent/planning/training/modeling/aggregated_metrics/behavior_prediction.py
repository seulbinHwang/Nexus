from typing import List

import torch
from nuplan.planning.training.modeling.types import TargetsType

from torchmetrics import Metric

from nuplan_extent.planning.training.preprocessing.feature_builders.nexus_feature_builder import (
    decode_scene_tensor,
)


class BehaviorPredictionMetric(Metric):
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
        super(BehaviorPredictionMetric, self).__init__()
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
        self.displacement_at_t: torch.Tensor
        self.add_state("displacement_at_t", default=torch.zeros(n_future_timesteps))
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

        if self._use_only_ego:
            valid = valid[:, self._ego_idx : self._ego_idx + 1]
            gt = gt[:, self._ego_idx : self._ego_idx + 1]
            pred = pred[:, self._ego_idx : self._ego_idx + 1]

        if self._exclude_ego:
            valid[:, self._ego_idx] = 0

        self.entries_at_t += valid.sum(axis=(0, 1))  # NT

        errors = torch.norm(gt[..., :2] - pred[..., :2], dim=-1)  # B, NA, NT

        errors *= valid
        self.displacement_at_t += errors.sum(axis=(0, 1))

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
            out[f"displacement@{time:1f}s"] = (
                self.displacement_at_t[..., time_idx].sum()
                / self.entries_at_t[..., time_idx].sum()
            )
        total=len(out)
        sum_8 = 0
        sum_7 = 0
        for i,(k,v) in enumerate(out.items()):
            if i < 7:
                sum_7 += v
            sum_8 += v
        out["displacement@avg_8s"] = sum_8/total
        out["displacement@avg_7s"] = sum_7/(total-1)

        return out

    def log(self, logger, data: dict):
        if not data:
            return
        prefix = f"aggregated_metrics/{self._name}/"
        for k, v in data.items():
            logger(prefix + k, v.detach().cpu().item())