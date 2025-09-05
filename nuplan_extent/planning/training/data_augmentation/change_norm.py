import numpy as np
import numpy.typing as npt
import torch
import logging
from typing import List, Optional, Tuple, cast
from copy import deepcopy

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ConstrainedNonlinearSmoother,
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_augmented_ego_raster,
    rotate_tilt_angle,
)
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import agent_bbox_to_corners

logger = logging.getLogger(__name__)

from nuplan_extent.planning.training.preprocessing.features.scene_tensor import FEATURE_MEANS, FEATURE_STD, OLD_FEATURE_MEANS, OLD_FEATURE_STD


class ChangeNormAugmentor(AbstractAugmentor):
    """
    This class copies the target trajectory to the feature dictionary.
    Sometimes the target trajectory is used as a input feature during training for the model.
    such as multibin model, we use target trajectory to generate high level command.

    """

    def __init__(
        self,
    ) -> None:
        """
        :param trajectory_steps: int, total steps of expert trajectory.
        """
        self._augment_prob = 1.0

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType,
        scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        dtype = features["scene_tensor"].tensor.dtype
        # print('scene tensor shape: ', features["scene_tensor"].tensor.shape)
        # print('original STD: ', np.array(OLD_FEATURE_STD, dtype=dtype))
        # print('current STD: ', np.array(FEATURE_STD, dtype=dtype))
        # print("(0) encoded ego at future 1: ", features["scene_tensor"].tensor[0,6])
        features["scene_tensor"].tensor = features["scene_tensor"].tensor * np.array(OLD_FEATURE_STD, dtype=dtype) * 2.0 + np.array(OLD_FEATURE_MEANS, dtype=dtype)
        # print("(1) decoded ego at future 1: ", features["scene_tensor"].tensor[0,6])
        features["scene_tensor"].tensor = (features["scene_tensor"].tensor - np.array(FEATURE_MEANS, dtype=dtype)) / np.array(FEATURE_STD, dtype=dtype) / 2.0
        # print("(2) encoded ego at future 1: ", features["scene_tensor"].tensor[0,6])
        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['']

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f'{self._augment_prob}='.partition('=')[0].split('.')
            [1],
            scaling_direction=ScalingDirection.MAX,
        )
