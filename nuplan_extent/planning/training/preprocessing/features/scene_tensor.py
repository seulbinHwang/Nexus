from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union
import numpy as np
import torch
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    EgoInternalIndex,
)
from nuplan_extent.planning.scenario_builder.wod_db.wod_scenario import WodScenario
import nuplan_extent.planning.training.preprocessing.features.vector_builders as vb
from nuplan_extent.planning.scenario_builder.prepared_scenario import (
    NpAgentState as AS,
)
from nuplan_extent.planning.scenario_builder.prepared_scenario import (
    NpEgoState,
    PreparedScenario,
)

from nuplan_extent.planning.training.preprocessing.features.raster_builders import (
    PreparedMapObject,
)
import numpy as np
import tensorflow as tf


SCENE_TENSOR_FEATURES = [
    "x",
    "y",
    "cos(yaw)",
    "sin(yaw)",
    "vx",
    "vy",
    "l",
    "w",
]
N_SCENE_TENSOR_FEATURES = len(SCENE_TENSOR_FEATURES)
VEH_PARAMS = get_pacifica_parameters()
EGO_WIDTH = VEH_PARAMS.width
EGO_LENGTH = VEH_PARAMS.front_length + VEH_PARAMS.rear_length
CLASSES = ["EGO", "VEHICLE", "PEDESTRIAN", "BICYCLE"]

FEATURE_MEANS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 2.0]
FEATURE_STD = [52.0, 52.0, 0.5, 0.5, 2.0, 2.0, 2.5, 0.8] # [70.0, 70.0, 0.5, 0.5, 10.0, 10.0, 2.5, 0.8]

OLD_FEATURE_MEANS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 2.0]
OLD_FEATURE_STD = [52.0, 52.0, 0.5, 0.5, 2.0, 2.0, 2.5, 0.8]


@dataclass
class SceneTensor(AbstractModelFeature):
    tensor: FeatureDataType
    validity: FeatureDataType
    road_graph: FeatureDataType
    road_graph_validity: FeatureDataType

    def serialize(self) -> Dict[str, Any]:
        return {
            "tensor": self.tensor,
            "validity": self.validity,
            "road_graph": self.road_graph,
            "road_graph_validity": self.road_graph_validity,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        return SceneTensor(
            tensor=data["tensor"],
            validity=data["validity"],
            road_graph=data["road_graph"],
            road_graph_validity=data["road_graph_validity"],
        )

    def to_feature_tensor(self) -> AbstractModelFeature:
        return SceneTensor(
            tensor=to_tensor(self.tensor),
            validity=to_tensor(self.validity),
            road_graph=to_tensor(self.road_graph),
            road_graph_validity=to_tensor(self.road_graph_validity),
        )

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        self.tensor = to_tensor(self.tensor).to(device=device)
        self.validity = to_tensor(self.validity).to(device=device)
        self.road_graph = to_tensor(self.road_graph).to(device=device)
        self.road_graph_validity = to_tensor(self.road_graph_validity).to(device=device)

        return self

    def unpack(self) -> List[AbstractModelFeature]:
        if len(self.tensor.shape) == 3:
            return [self]
        else:
            return [
                SceneTensor(
                    tensor=self.tensor[i],
                    validity=self.validity[i],
                    road_graph=self.road_graph[i],
                    road_graph_validity=self.road_graph_validity[i],
                )
                for i in range(self.tensor.shape[0])
            ]

    @classmethod
    def collate(cls, batch: List["SceneTensor"]) -> "SceneTensor":  # type: ignore
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        largest_n_actors = max([sample.tensor.shape[0] for sample in batch])
        # pad the tensors
        for sample in batch:
            n_actors = sample.tensor.shape[0]
            if n_actors < largest_n_actors:
                pad = torch.zeros(
                    (
                        largest_n_actors - n_actors,
                        sample.tensor.shape[1],
                        N_SCENE_TENSOR_FEATURES,
                    )
                )
                pad_validity = torch.zeros(
                    (largest_n_actors - n_actors, sample.tensor.shape[1])
                )
                sample.tensor = torch.concat([sample.tensor, pad], dim=0)
                sample.validity = torch.concat([sample.validity, pad_validity], dim=0)

        return cls(
            tensor=torch.stack([sample.tensor for sample in batch]),
            validity=torch.stack([sample.validity for sample in batch]),
            road_graph=torch.stack([sample.road_graph for sample in batch]),
            road_graph_validity=torch.stack(
                [sample.road_graph_validity for sample in batch]
            ),
        )


def unnormalize_roadgraph(road_graph: Union[np.ndarray, torch.Tensor]):
    if isinstance(road_graph, torch.Tensor):
        road_graph[..., :2] = (
            road_graph[..., :2] * 2 * torch.tensor(FEATURE_STD,device=road_graph.device)[:2]
            + torch.tensor(FEATURE_MEANS,device=road_graph.device)[:2]
        )
    else:
        road_graph[..., :2] = (
            road_graph[..., :2] * 2 * np.array(FEATURE_STD)[:2]
            + np.array(FEATURE_MEANS)[:2]
        )
    return road_graph


def decode_scene_tensor(
    scene_tensor: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(scene_tensor, torch.Tensor):
        return scene_tensor * 2.0 * torch.tensor(
            FEATURE_STD[:scene_tensor.shape[-1]], device=scene_tensor.device
        ) + torch.tensor(FEATURE_MEANS[:scene_tensor.shape[-1]], device=scene_tensor.device)
    else:
        return scene_tensor * 2.0 * np.array(FEATURE_STD[:scene_tensor.shape[-1]]) + np.array(FEATURE_MEANS[:scene_tensor.shape[-1]])

def encode_scene_tensor(
    scene_tensor: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(scene_tensor, torch.Tensor):
        return (scene_tensor - torch.tensor(FEATURE_MEANS[:scene_tensor.shape[-1]], device=scene_tensor.device)) / (2.0 * torch.tensor(FEATURE_STD[:scene_tensor.shape[-1]], device=scene_tensor.device))         
    else:
        return (scene_tensor - np.array(FEATURE_MEANS[:scene_tensor.shape[-1]])) / ( 2.0 * np.array(FEATURE_STD))

def encode_agent(
    cls: str,
    agent_state: AS,
) -> np.ndarray:
    """
    Encode agent data into a feature tensor.
    """

    state = np.stack(
        [
            agent_state[:, AgentInternalIndex.x()],
            agent_state[:, AgentInternalIndex.y()],
            np.cos(agent_state[:, AgentInternalIndex.heading()]),
            np.sin(agent_state[:, AgentInternalIndex.heading()]),
            agent_state[:, AgentInternalIndex.vx()],
            agent_state[:, AgentInternalIndex.vy()],
            agent_state[:, AgentInternalIndex.length()],
            agent_state[:, AgentInternalIndex.width()],
        ],
        axis=1,
    )
    return state


def encode_ego(
    ego_state: NpEgoState,
) -> np.ndarray:

    state = np.stack(
        [
            ego_state[:, EgoInternalIndex.x()],
            ego_state[:, EgoInternalIndex.y()],
            np.cos(ego_state[:, EgoInternalIndex.heading()]),
            np.sin(ego_state[:, EgoInternalIndex.heading()]),
            ego_state[:, EgoInternalIndex.vx()],
            ego_state[:, EgoInternalIndex.vy()],
            np.full(ego_state.shape[0], EGO_LENGTH),
            np.full(ego_state.shape[0], EGO_WIDTH),
        ],
        axis=1,
    )
    return state



