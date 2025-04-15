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
logger = logging.getLogger(__name__)
Pose = Tuple[float, float, float]  # (x, y, yaw)

from nuplan_extent.planning.training.preprocessing.features.scene_tensor import decode_scene_tensor, encode_scene_tensor


class CenterlineVAugmentor(AbstractAugmentor):
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
        road_graph = features["scene_tensor"].road_graph
        if road_graph.shape[-1] == 5:
            features["scene_tensor"].road_graph = np.concatenate([road_graph, np.zeros_like(road_graph[..., :2])], axis=-1)
            features["scene_tensor"].road_graph_validity = np.concatenate([features["scene_tensor"].road_graph_validity, features["scene_tensor"].road_graph_validity[..., :2]], axis=-1)
        elif road_graph.shape[-1] == 7:
            road_graph = features["scene_tensor"].road_graph
            mask = ~(road_graph[...,2] + road_graph[...,3] == 1)[...,None].repeat(7,2)
            features["scene_tensor"].road_graph_validity = np.where(
                mask, 
                np.zeros_like(features["scene_tensor"].road_graph_validity), 
                features["scene_tensor"].road_graph_validity
            )
        scene_tensor = decode_scene_tensor(features["scene_tensor"].tensor)
        validity = features["scene_tensor"].validity
        scene_tensor = compute_velocity(scene_tensor, validity)
        features["scene_tensor"].tensor = encode_scene_tensor(scene_tensor).astype(features["scene_tensor"].tensor.dtype)

        # only use fisrt 64 agents
        features["scene_tensor"].tensor = features["scene_tensor"].tensor[:64]
        features["scene_tensor"].validity = features["scene_tensor"].validity[:64]
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

def compute_velocity(scene_tensor, valid_mask):

    NT = scene_tensor.shape[1]
    freq = 2
    v_pred = []
    for t in range(NT-1):
        v_avg = (scene_tensor[:,t+1:t+2,:2]-scene_tensor[:,t:t+1,:2]) * freq
        valid = valid_mask[:,t+1:t+2] * valid_mask[:,t:t+1]
        v_avg = v_avg * valid[...,None]
        v_avg = np.where(abs(v_avg).sum() == 0, scene_tensor[:,t:t+1,4:6], v_avg)
        v_pred.append(v_avg)
    v_pred.append(v_avg)
    v_pred = np.concatenate(v_pred, 1)
    
    scene_tensor[:, :, 4:6] = v_pred.astype(scene_tensor.dtype)
    
    return scene_tensor