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

from nuplan_extent.planning.training.preprocessing.features.scene_tensor import decode_scene_tensor, encode_scene_tensor
import pickle


class ResetVAugmentor(AbstractAugmentor):
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
        # invalid the past 4 frames of ego (unsee ego history)
        # features["scene_tensor"].validity[0,:4] = 0
        # data = {
        #     "features": features,
        #     "targets": targets,
        #     "scenario": scenario
        # }
        # # 使用 pickle 保存到文件
        # with open('/cpfs01/user/yenaisheng/test.pkl', "wb") as f:
        #     pickle.dump(data, f)
        #     print("dumped")
        #     exit()
        # import pdb; pdb.set_trace()
        # NA,NT,Ndim
        # decode scene_tensor
        scene_tensor = decode_scene_tensor(features["scene_tensor"].tensor[...,:8])

        # copy w,l of ego to all agents:  scene_tensor[0,0,-2:]
        # scene_tensor[:,5:,-2:] = scene_tensor[0,0,-2:].reshape(1,1,2)
        valid_mask = features["scene_tensor"].validity
        # change v
        detas=[]
        freq = 2
        NT = 21
        for t in range(NT-1):
            deta = (scene_tensor[:, t + 1:t + 2] - scene_tensor[:, t:t + 1]) * freq
            valid = valid_mask[:, t + 1:t + 2] * valid_mask[:, t:t + 1]
            valid = valid.astype(bool)  # Convert to boolean
            # Use np.where to handle the condition
            deta = np.where(valid[..., np.newaxis], deta, scene_tensor[:, t:t + 1])
            detas.append(deta)
        # Insert the first scene_tensor at the beginning
        detas.insert(0, scene_tensor[:, 0:1])
        # Concatenate along the second dimension
        new_st = np.concatenate(detas, axis=1)
        scene_tensor[...,[4,5]] = new_st[...,[4,5]] 
        # encode scene_tensor
        features["scene_tensor"].tensor[...,:8] = encode_scene_tensor(scene_tensor).astype(dtype)
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
