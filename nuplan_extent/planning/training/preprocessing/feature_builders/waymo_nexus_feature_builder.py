from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union
from waymo_open_dataset.utils.sim_agents import submission_specs
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
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_vector_feature_builder_v2 import (
    HorizonVectorFeatureBuilderV2,
    HorizonVectorV2,
)
from nuplan_extent.planning.training.preprocessing.features.raster_builders import (
    PreparedMapObject,
)
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils import trajectory_utils
import pickle as pkl
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput

from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from .waymo_vector_feature_builder_v2 import WaymoVectorFeatureBuilderV2
from .waymo_map_feature_builder import WaymoMapFeatureBuilder
from ..features.scene_tensor import (SceneTensor,
    SCENE_TENSOR_FEATURES,N_SCENE_TENSOR_FEATURES,  
    FEATURE_MEANS,FEATURE_STD,
    VEH_PARAMS,EGO_WIDTH,EGO_LENGTH,CLASSES,
    unnormalize_roadgraph,decode_scene_tensor,encode_agent,encode_ego)  
from nuplan_extent.planning.training.preprocessing.features.dict_tensor_feature import DictTensorFeature
# waymo ego vehicle parameters
N_TIMESTEPS=21
EGO_WIDTH=2.3320000171661377
EGO_LENGTH=5.285999774932861

class NexusFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        max_num_map_objects: int, 
        num_points_each_polyline: int,
        center_offset: List[float],
        agent_features: List[str],
        radius: float = 100.0, 
        longitudinal_offset: float = 0,
        past_time_horizon: float = 2.0,
        past_num_steps: int = 4,
        future_time_horizon: int = 8.0,
        future_num_steps: int = 16,
        num_max_agents: List[int] = [256,128,32],
        num_max_used_agents: int = 256,
        map_dim_points_polyline: int = 7
    ):
        self.mapFeatureBuilder = WaymoMapFeatureBuilder(max_num_map_objects, num_points_each_polyline,
                                                        radius, center_offset)
        self.vectorFeatureBuilder = WaymoVectorFeatureBuilderV2(agent_features, radius,
                                        longitudinal_offset, past_time_horizon,
                                        past_num_steps, future_time_horizon,
                                        future_num_steps, num_max_agents)
        self.past_num_steps = past_num_steps
        self.past_time_horizon = past_time_horizon
        self.future_num_steps = future_num_steps
        self.future_time_horizon = future_time_horizon
        # we set the max limit for vehicles,pedestrains,bicycles in num_max_agents
        self._num_max_agents = num_max_agents
        # since we'll combine all agents into one tensor, we need to set the max limit for all agents named num_max_used_agents
        self._num_max_used_agents = num_max_used_agents
        self._map_num_max_polylines = max_num_map_objects
        self._map_num_points_polylines = num_points_each_polyline
        self._map_dim_points_polyline = map_dim_points_polyline
        self._agent_features = agent_features
        self._radius = radius
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "scene_tensor"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return SceneTensor  # type: ignore

    def get_features_from_scenario(
        self, scenario:AbstractScenario, iteration: int = 10
    ) -> SceneTensor:
        map_features = self.mapFeatureBuilder.get_features_from_scenario(scenario)
        agents_features,sim_eval_agent_ids = self.vectorFeatureBuilder.get_features_from_scenario(scenario,iteration)
        return self._compute_features(map_features, agents_features, scenario, sim_eval_agent_ids)

    def get_features_from_prepared_scenario(
        self, scenario: PreparedScenario, iteration: int, ego_state: NpEgoState
    ) -> AbstractModelFeature:
        pass

    def get_features_from_simulation(
            self, current_input: PlannerInput,
            initialization: PlannerInitialization) -> HorizonVectorV2:
        """Inherited, see superclass."""
        pass

    def _compute_features(self, map_features: DictTensorFeature, agents_features: HorizonVectorV2, scenario:AbstractScenario, sim_eval_agent_ids) -> SceneTensor:
        """
        Compute the scene tensor from the map and agent features.
        Args:
            map_features: 
                1. map_polylines: (max_num_map_objects, num_points_each_polyline, 9)
                    x,y,z, dir_x,dir_y,dir_z, map_object_type, pre_x,pre_y
                    you can refer to third_party.MTR.mtr.datasets.waymo.waymo_types.polyline_type to get details of map_object_type,
                    and the map_object_type includes [BASELINE_PATHS, LANE, EXTENDED_PUDO, STOP_SIGN, CROSSWALK, SPEED_BUMP] semantics.
                2. map_polylines_mask: (max_num_map_objects, num_points_each_polyline)
                    valid mask for map_polylines
                3. map_polylines_center: (max_num_map_objects, 3)
                    center of each polyine in x,y,z channels
            agents_features:
                1. ego: (n_timesteps, 7) 
                    x,y,heading,vx,vy,ax,ay 
                2. agents(VEHICLE, PEDESTRAIN, BICYCLE): (n_timesteps, num_max_agents[i], 9)
                    index(0_start), vx,vy, heading, width,length, x,y, track_id
        Note:
            1. n_timesteps = past_num_steps + 1 + future_num_steps
            2. default: max_num_map_objects = 256,  num_points_each_polyline = 20, num_max_agents = [256,128,32], n_timesteps = 21
            3. unvalid features in agents_features will be presented as torch.nan,
                while unvalid features in map_features will be presented as 0 with valid mask as 0 too.
            """
        # import pdb; pdb.set_trace()
        vehicles = agents_features.data["agents"]["VEHICLE"]  # type: ignore
        peds = agents_features.data["agents"]["PEDESTRIAN"]  # type: ignore
        bic = agents_features.data["agents"]["BICYCLE"]  # type: ignore
        vehicles[np.any(np.isnan(vehicles), axis=-1)] = np.nan 
        peds[np.any(np.isnan(peds), axis=-1)] = np.nan
        bic[np.any(np.isnan(bic), axis=-1)] = np.nan

        ego = agents_features.data["ego"]  # t x 7 tensor

        # waymo only provides the 1 second past, so the frist 2 timesteps are invalid
        vehicles[:2] = np.nan
        peds[:2] = np.nan
        bic[:2] = np.nan

        n_timesteps = len(vehicles)

        # n_agents = len(ids_veh) + len(ids_ped) + len(ids_bic)
        n_max_agents = self._num_max_used_agents

        sim_agent_ids = sim_eval_agent_ids['sim_agent_ids']
        eval_agent_ids = sim_eval_agent_ids['eval_agent_ids']
        ego_id = sim_eval_agent_ids["ego_id"]

        origin_ids_veh = set(np.unique(vehicles[~np.isnan(vehicles[..., 0]), -1].astype(int)))
        origin_ids_ped = set(np.unique(peds[~np.isnan(peds[..., 0]), -1].astype(int)))
        origin_ids_bic = set(np.unique(bic[~np.isnan(bic[..., 0]), -1].astype(int)))
        veh_sim = origin_ids_veh & sim_agent_ids
        ped_sim = origin_ids_ped & sim_agent_ids
        bic_sim = origin_ids_bic & sim_agent_ids

        # get all unique track ids
        ids_veh = np.unique(vehicles[~np.isnan(vehicles[..., 0]), 0])
        ids_ped = np.unique(peds[~np.isnan(peds[..., 0]), 0])
        ids_bic = np.unique(bic[~np.isnan(bic[..., 0]), 0])

        # create nan tensor, +1 to include agent id at end
        scene_tensor = np.zeros((n_max_agents, n_timesteps, N_SCENE_TENSOR_FEATURES))
        agents_id = np.ones((n_max_agents,N_TIMESTEPS,1))*-1.0
        scene_tensor_validity = np.zeros((n_max_agents, n_timesteps))
        keep_mask = np.zeros((n_max_agents, n_timesteps), dtype=bool)
        n_agents = 0
        max_veh = n_max_agents - len(ped_sim) - len(bic_sim)
        for i, id_ in enumerate(ids_veh):
            if i >= max_veh :
                break
            mask = vehicles[..., 0] == id_
            time_validity_ = mask.sum(axis=1).astype(bool)
            # if  ~ time_validity_[self.past_num_steps] and time_validity_.sum() < 10: # if current is invalid and less than 10 valid timesteps,skip
            #     continue
            scene_tensor[n_agents, time_validity_] = encode_agent(
                "VEHICLE",
                vehicles[mask],  # type: ignore
            )
            agent_id = vehicles[mask][0, -1]
            agents_id[n_agents] = agent_id
            scene_tensor_validity[n_agents] = time_validity_
            if agent_id in eval_agent_ids:
                keep_mask[n_agents][2:5] = time_validity_[2:5]
                keep_mask[n_agents][np.where(time_validity_)[0][-1]] = True
                scene_tensor_validity[n_agents][2:] = 1.
            else:
                keep_mask[n_agents] = time_validity_
            n_agents += 1

        max_ped = n_max_agents - n_agents - len(bic_sim)
        for i, id_ in enumerate(ids_ped):
            if i >= max_ped:
                break
            mask = peds[..., 0] == id_
            time_validity_ = mask.sum(axis=1).astype(bool)
            if ~ time_validity_[self.past_num_steps] and time_validity_.sum() < 5:
                continue

            scene_tensor[n_agents, time_validity_] = encode_agent(
                "PEDESTRIAN",
                peds[mask],  # type: ignore
            )
            agent_id = peds[mask][0, -1]
            agents_id[n_agents] = agent_id
            scene_tensor_validity[n_agents] = time_validity_
            if agent_id in eval_agent_ids:
                keep_mask[n_agents][2:5] = time_validity_[2:5]
                keep_mask[n_agents][np.where(time_validity_)[0][-1]] = True
                scene_tensor_validity[n_agents][2:] = 1.
            else:
                keep_mask[n_agents] = time_validity_            
            n_agents += 1

        for i, id_ in enumerate(ids_bic):
            if n_agents >= n_max_agents:
                break
            mask = bic[..., 0] == id_
            time_validity_ = mask.sum(axis=1).astype(bool)
            if ~ time_validity_[self.past_num_steps] and time_validity_.sum() < 5:
                continue
            scene_tensor[n_agents, time_validity_] = encode_agent("BICYCLE", bic[mask])  # type: ignore
            agent_id = bic[mask][0, -1]
            agents_id[n_agents] = agent_id
            scene_tensor_validity[n_agents] = time_validity_
            if agent_id in eval_agent_ids:
                keep_mask[n_agents][2:5] = time_validity_[2:5]
                keep_mask[n_agents][np.where(time_validity_)[0][-1]] = True
                scene_tensor_validity[n_agents][2:] = 1.
            else:
                keep_mask[n_agents] = time_validity_   
            n_agents += 1

        ego_state = encode_ego(ego)
        agents_id = np.concatenate([np.ones((1,N_TIMESTEPS,1))*ego_id, agents_id], axis=0)

        scene_tensor = np.concatenate(
            [np.expand_dims(ego_state, 0), scene_tensor], axis=0
        )
        scene_tensor_validity = np.concatenate(
            [np.ones((1, n_timesteps)), scene_tensor_validity], axis=0
        )
        scene_tensor_validity[0, :2] = 0 # invalid ego state
        keep_mask = np.concatenate([np.zeros((1, n_timesteps), dtype=bool), keep_mask], axis=0)
        keep_mask[0, 2:5] = 1
        keep_mask[0, -1] = 1
        if n_timesteps < N_TIMESTEPS:
            padding_shape = (scene_tensor.shape[0], N_TIMESTEPS - n_timesteps, scene_tensor.shape[2])
            scene_tensor = np.concatenate([scene_tensor, np.zeros(padding_shape)], axis=1)
            scene_tensor_validity = np.concatenate([scene_tensor_validity, np.zeros((scene_tensor_validity.shape[0], N_TIMESTEPS - n_timesteps))], axis=1)
            keep_mask = np.concatenate([keep_mask, np.zeros((keep_mask.shape[0], N_TIMESTEPS - n_timesteps), dtype=bool)], axis=1)
        # concatenate keep_mask to scene_tensor_validity, and unify the dtype to scene_tensor_validity.dtype
        scene_tensor_validity = np.concatenate([scene_tensor_validity, keep_mask.astype(scene_tensor_validity.dtype)], axis=-1)
        
        # normalize the scene with x-mu/2sigma
        scene_tensor = (scene_tensor - np.array(FEATURE_MEANS)) / (
            2.0 * np.array(FEATURE_STD)
        )
        scene_tensor = np.concatenate([scene_tensor, agents_id], axis=-1)
        # delete z-related channels in x,y,z, dir_x,dir_y,dir_z, map_object_type, pre_x,pre_y
        select_channels = [0, 1, 3, 4, 7, 8, 6] # x,y, dir_x,dir_y, pre_x,pre_y, map_object_type
        map_polylines = map_features.data_dict["map_polylines"][..., select_channels]
        map_polylines_mask = map_features.data_dict["map_polylines_mask"]
        road_graph = np.zeros(
            shape=(
                self._map_num_max_polylines,
                self._map_num_points_polylines,
                self._map_dim_points_polyline,
            ),
            dtype=np.float32,
        )
        road_graph_validity = np.zeros_like(road_graph)

        map_polylines_mask_ext = np.zeros((self._map_num_max_polylines, self._map_num_points_polylines), dtype=bool)
        map_polylines_mask_ext[:map_polylines_mask.shape[0]] = map_polylines_mask
        road_graph[map_polylines_mask_ext] = map_polylines[map_polylines_mask]
        road_graph_validity[map_polylines_mask_ext] = 1

        road_graph[...,:-1]= (road_graph[...,:-1] - np.array(FEATURE_MEANS[:2]*3)) / (2.0 * np.array(FEATURE_STD[:2]*3))
        road_graph[..., -1]=(road_graph[..., -1]+1)/20. # there are 20 types of map_object_type

        return SceneTensor(
            tensor=scene_tensor.astype(np.float32),
            validity=scene_tensor_validity.astype(np.float32),
            road_graph=road_graph.astype(np.float32),
            road_graph_validity=road_graph_validity.astype(np.float32),
        )
