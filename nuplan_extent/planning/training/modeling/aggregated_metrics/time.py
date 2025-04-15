from typing import List
import torch
from nuplan.planning.training.modeling.types import TargetsType
from torchmetrics import Metric


class TimeMetric(Metric):
    def __init__(
        self,
        name: str = "time",
        prediction_output_key: str = "time_consume",
    ) -> None:
        """
        Initializes the class.
        :param name: the name of the metric (used in logger)
        """
        super().__init__()
        self._name = name
        self._key = prediction_output_key
        self.add_state("time", default=torch.zeros(1,dtype=torch.float32))
        self.add_state("count", default=torch.zeros(1,dtype=torch.float32))
    def name(self) -> str:
        return self._name

    def update(self, predictions: TargetsType, targets: TargetsType) -> None:
        if "behavior_prediction_out" not in predictions:
            return

        if self._key not in predictions["behavior_prediction_out"]:
            return

        self.time += predictions["behavior_prediction_out"][self._key]
        self.count += 1

    def compute(self) -> dict:
        """
        Computes the metric.
        :return: metric scalar tensor
        """
        out = {}
        # note that the attributes are tensors, but have been addded an extra dimention corresponding to the number of
        # gpus
        out["time"] = self.time.sum() / self.count.sum()

        return out

    def log(self, logger, data: dict):
        if not data:
            return
        prefix = f"aggregated_metrics/"
        for k, v in data.items():
            logger(prefix + k, v.detach().cpu().item())
