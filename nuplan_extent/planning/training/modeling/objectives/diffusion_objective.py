from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from torch.nn import functional as F


class DiffusionObjective(AbstractObjective):
    """
    A class that represents the speed heatmap objective for trajectory prediction models in autonomous driving.
    Enforces the predicted heatmap to be close to the optimal speed heatmap.
    Can improve speed limit compliance, and ego progress along expert routes.
    """

    def __init__(self,
                 scenario_type_loss_weighting: Dict[str, float],
                 weight: float = 1.0):
        """
        """
        self._name = f'diffusion_objective'
        self._weight = weight

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: FeaturesType, targets: TargetsType,
                scenarios: ScenarioListType) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'diffusion_out' not in predictions or predictions['diffusion_out'] == None:
            loss = torch.tensor(0.0).to(device)
            return loss

        if "metric" not in predictions['diffusion_out']:
            loss = predictions['diffusion_out']['loss'].to(device)
            return loss * self._weight
        return predictions['diffusion_out']['metric'].to(device)