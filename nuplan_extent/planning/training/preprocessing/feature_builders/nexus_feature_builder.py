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

from ..features.scene_tensor import (SceneTensor,
    SCENE_TENSOR_FEATURES,N_SCENE_TENSOR_FEATURES,  
    FEATURE_MEANS,FEATURE_STD,
    VEH_PARAMS,EGO_WIDTH,EGO_LENGTH,CLASSES,
    unnormalize_roadgraph,decode_scene_tensor,encode_agent,encode_ego,encode_scene_tensor)  


class NexusFeatureBuilder(HorizonVectorFeatureBuilderV2):
    def __init__(
        self,
        agent_features: List[str],
        radius: float = 150,
        longitudinal_offset: float = 0,
        past_time_horizon: float = 2.0,
        past_num_steps: int = 4,
        future_time_horizon: int = 8.0,
        future_num_steps: int = 16,
        num_max_agents: List[int] = [256, 128, 64],
        map_features: List[str] = ["LANE"],
        map_num_max_polylines: int = 200,
        map_num_points_polylines: int = 20,
        map_dim_points_polyline: int = 7,  # x,y + len(map_features),
    ) -> None:
        self._radius = radius
        builders = {}
        builders["agents"] = vb.PastCurrentAgentsVectorBuilder(
            radius=radius,
            longitudinal_offset=longitudinal_offset,
            past_time_horizon=past_time_horizon,
            past_num_steps=past_num_steps,
            future_time_horizon=future_time_horizon,
            future_num_steps=future_num_steps,
            agent_features=agent_features,
            num_max_agents=num_max_agents,
        )
        builders["ego"] = vb.PastCurrentEgoVectorBuilder(
            radius=radius,
            longitudinal_offset=longitudinal_offset,
            past_time_horizon=past_time_horizon,
            past_num_steps=past_num_steps,
            future_time_horizon=future_time_horizon,
            future_num_steps=future_num_steps,
        )
        builders["map"] = vb.MapVectorBuilder(
            image_size=0,  # not used
            radius=radius,
            longitudinal_offset=longitudinal_offset,
            map_features=map_features,
        )

        self._builders = builders
        self._num_max_agents = num_max_agents
        self._map_features = map_features
        self._map_num_points_polylines = map_num_points_polylines
        self._map_dim_points_polyline = map_dim_points_polyline
        self._map_num_max_polylines = map_num_max_polylines

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "scene_tensor"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return SceneTensor  # type: ignore

    def get_features_from_scenario(
        self, scenario:AbstractScenario, iteration: int = 0
    ) -> SceneTensor:
        if type(scenario) is WodScenario:
            return self._compute_waymo_features(scenario)
        vector_data = super().get_features_from_scenario(scenario, iteration)
        return self._compute_features(vector_data)
        
    def get_features_from_prepared_scenario(
        self, scenario: PreparedScenario, iteration: int, ego_state: NpEgoState
    ) -> AbstractModelFeature:
        vector_data = super().get_features_from_prepared_scenario(
            scenario, iteration, ego_state
        )
        return self._compute_features(vector_data)

    def _compute_waymo_features(self, scenario: WodScenario) -> SceneTensor:
        # deprecated, not used
        with open(scenario._pickle_path,'rb') as f:
            wod = pkl.load(f)
        states, dimensions, objects = [], [], []
        for track in wod.tracks:
            # Iterate over a single object's states.
            track_states, track_dimensions = [], []
            for state in track.states:
                track_states.append((state.center_x, state.center_y, state.center_z,
                            state.heading, state.valid))
                track_dimensions.append((state.length, state.width, state.height))
            # Adds to the global states.
            states.append(list(zip(*track_states)))
            dimensions.append(list(zip(*track_dimensions)))
            objects.append((track.id, track.object_type))

        # Unpack and convert to numpy arrays.
        x, y, z, heading, valid = [np.array(s) for s in zip(*states)]
        length, width, height = [np.array(s) for s in zip(*dimensions)]
        object_ids, object_types = [np.array(s) for s in zip(*objects)]

        logged_trajectories = {
            'x': x, 'y': y, 'z': z, 'heading': heading,
            'length': length, 'width': width, 'height': height,
            'valid': valid, 'object_id': object_ids,
            'object_type': object_types
        }

        # if the scenario is a test scenario, we only need to simulate part of agents
        need_predict = np.zeros(logged_trajectories['object_id'].shape, dtype=np.float32)
        if scenario.split == 'testing':
            sim_agent_ids = set(submission_specs.get_sim_agent_ids(wod))
            need_predict = np.isin(logged_trajectories['object_id'], list(sim_agent_ids)).astype(np.float32)

        # get the scene tensor and scene_tensor_validity 
        # since object_ids are necessary when submitting to waymo sim agent challenge, so we add object_id channel there
        def extend_dim(array):
            return np.tile(array[:, np.newaxis], (1, logged_trajectories['x'].shape[1])).astype(np.float32)

        scene_tensor = np.stack([ # x,y,z, heading, l,w,h, id,type, need_predict
            logged_trajectories['x'], logged_trajectories['y'], logged_trajectories['z'], 
            logged_trajectories['heading'], 
            logged_trajectories['length'], logged_trajectories['width'], logged_trajectories['height'],
            extend_dim(logged_trajectories['object_id']), extend_dim(logged_trajectories['object_type']), extend_dim(need_predict)], axis=-1)
        scene_tensor_validity = logged_trajectories['valid']

        # release memory
        del(logged_trajectories)
        # Move agents with need_predict == 1 to the front
        need_predict_indices = np.where(scene_tensor[:, 0, -1] == 1)[0]
        non_need_predict_indices = np.where(scene_tensor[:, 0, -1] == 0)[0]

        scene_tensor = np.concatenate(
            [scene_tensor[need_predict_indices], scene_tensor[non_need_predict_indices]],
            axis=0
        )
        scene_tensor_validity = np.concatenate(
            [scene_tensor_validity[need_predict_indices], scene_tensor_validity[non_need_predict_indices]],
            axis=0
        )

        # Find the ego index
        ego_id = wod.tracks[wod.sdc_track_index].id
        ego_index = np.where(scene_tensor[:, 0, -3] == ego_id)[0][0]

        # Move the ego record to the first row
        scene_tensor = np.concatenate(
            [scene_tensor[ego_index:ego_index+1], np.delete(scene_tensor, ego_index, axis=0)],
            axis=0
        )
        scene_tensor_validity = np.concatenate(
            [scene_tensor_validity[ego_index:ego_index+1], np.delete(scene_tensor_validity, ego_index, axis=0)],
            axis=0
        )
        # n_agents: the number of agents in the scene
        # n_timesteps: the number of total timesteps in the scene
        n_agents = scene_tensor.shape[0]
        n_max_timesteps = submission_specs.N_FULL_SCENARIO_STEPS
        n_max_agents = sum(self._num_max_agents)
        # to avoid the number of agents exceeding the maximum number of agents
        n_agents = min(n_agents,n_max_agents)
        # Initialize tensor and valid mask with zeros
        padded_scene_tensor = np.zeros((n_max_agents, n_max_timesteps, scene_tensor.shape[-1]), dtype=np.float32)
        padded_valid_mask = np.zeros((n_max_agents, n_max_timesteps), dtype=np.float32)

        # Copy the original data into the initialized tensors
        padded_scene_tensor[:n_agents, :scene_tensor.shape[1], :] = scene_tensor[:n_agents]
        padded_valid_mask[:n_agents, :scene_tensor_validity.shape[1]] = scene_tensor_validity[:n_agents]

        # update the scene tensor and valid mask, and change to torch.Tensor
        scene_tensor = padded_scene_tensor
        scene_tensor_validity = padded_valid_mask
        
        # dim should be 3 for x,y,z in waymo open dataset
        # if self._map_dim_points_polyline != 3:
        #     self._map_dim_points_polyline = 3
        # Initialize road graph and validity tensors
        road_graph = np.zeros(
            shape=(
            self._map_num_max_polylines,
            self._map_num_points_polylines,
            self._map_dim_points_polyline,  # only has x, y, z channels
            ),
            dtype=np.float32,
        )
        road_graph_validity = np.zeros_like(road_graph)

        index = 0
        for map_feature in wod.map_features:
            if index >= self._map_num_max_polylines:
                break
            map_type = map_feature.WhichOneof('feature_data')
            if map_type in self._map_features:
                # Determine the polyline or polygon attribute based on map type
                poly = getattr(map_feature, map_type).polyline if map_type in ['lane', 'road_edge'] else getattr(map_feature, map_type).polygon

                # Initialize coordinates and validity arrays
                poly_coords = np.zeros((self._map_num_points_polylines, 3 + len(self._map_features)), dtype=np.float32)
                poly_validity = np.zeros_like(poly_coords)
                num_points = len(poly)

                # Sample points from the polyline or polygon
                if num_points > self._map_num_points_polylines:
                    indices = np.linspace(0, num_points - 1, self._map_num_points_polylines).astype(int)
                    for j, idx in enumerate(indices):
                        poly_coords[j, :3] = [poly[idx].x, poly[idx].y, poly[idx].z]
                        poly_validity[j] = 1
                else:
                    for j in range(num_points):
                        poly_coords[j, :3] = [poly[j].x, poly[j].y, poly[j].z]
                        poly_validity[j] = 1

                # One-hot encode the map type
                type_index = self._map_features.index(map_type)
                poly_coords[:, 3 + type_index] = 1

                # Assign the sampled points to the road graph and validity tensors
                road_graph[index,:,:poly_coords.shape[1]] = poly_coords
                road_graph_validity[index] = poly_validity
                index += 1
            else:
            # Ignore other map types (e.g., stop_sign, road_line)
               pass
        del(wod)

        return SceneTensor(
            tensor=scene_tensor.astype(np.float32),
            validity=scene_tensor_validity.astype(np.float32),
            road_graph=road_graph.astype(np.float32),
            road_graph_validity=road_graph_validity.astype(np.float32),
        )

    def _compute_features(self, vector_data: HorizonVectorV2) -> SceneTensor:
        vehicles = vector_data.data["agents"]["VEHICLE"]  # type: ignore
        peds = vector_data.data["agents"]["PEDESTRIAN"]  # type: ignore
        bic = vector_data.data["agents"]["BICYCLE"]  # type: ignore
        ego = vector_data.data["ego"]  # t x 7 tensor
        n_timesteps = len(vehicles)

        # get all unique track ids
        ids_veh = np.unique(vehicles[..., 0])
        ids_ped = np.unique(peds[..., 0])
        ids_bic = np.unique(bic[..., 0])

        # n_agents = len(ids_veh) + len(ids_ped) + len(ids_bic)
        n_max_agents = sum(self._num_max_agents)

        # create nan tensor
        scene_tensor = np.zeros((n_max_agents, n_timesteps, N_SCENE_TENSOR_FEATURES))
        scene_tensor_validity = np.zeros((n_max_agents, n_timesteps))

        n_agents = 0
        for i, id_ in enumerate(ids_veh):
            if i >= self._num_max_agents[0]:
                break
            mask = vehicles[..., 0] == id_
            time_validity_ = mask.sum(axis=1).astype(bool)
            if time_validity_.sum() < 5:
                continue
            scene_tensor[n_agents, time_validity_] = encode_agent(
                "VEHICLE",
                vehicles[mask],  # type: ignore
            )
            scene_tensor_validity[n_agents] = time_validity_
            n_agents += 1

        for i, id_ in enumerate(ids_ped):
            if i >= self._num_max_agents[1]:
                break
            mask = peds[..., 0] == id_
            time_validity_ = mask.sum(axis=1).astype(bool)
            if time_validity_.sum() < 5:
                continue

            scene_tensor[n_agents, time_validity_] = encode_agent(
                "PEDESTRIAN",
                peds[mask],  # type: ignore
            )
            scene_tensor_validity[n_agents] = time_validity_
            n_agents += 1

        for i, id_ in enumerate(ids_bic):
            if i >= self._num_max_agents[2]:
                break
            mask = bic[..., 0] == id_
            time_validity_ = mask.sum(axis=1).astype(bool)
            if time_validity_.sum() < 5:
                continue
            scene_tensor[n_agents, time_validity_] = encode_agent("BICYCLE", bic[mask])  # type: ignore
            scene_tensor_validity[n_agents] = time_validity_
            n_agents += 1

        # append the ego state

        ego_state = encode_ego(ego)
        scene_tensor = np.concatenate(
            [np.expand_dims(ego_state, 0), scene_tensor], axis=0
        )
        # normalize the scene with x-mu/2sigma
        scene_tensor = (scene_tensor - np.array(FEATURE_MEANS)) / (
            2.0 * np.array(FEATURE_STD)
        )
        scene_tensor_validity = np.concatenate(
            [np.ones((1, n_timesteps)), scene_tensor_validity], axis=0
        )

        # encode road_graph
        road_graph = np.zeros(
            shape=(
                self._map_num_max_polylines,
                self._map_num_points_polylines,
                self._map_dim_points_polyline,
            ),
            dtype=np.float32,
        )
        road_graph_validity = np.zeros_like(road_graph)
        map_data = vector_data.data["map"]

        for i in range(min(self._map_num_max_polylines, len(map_data))):
            road_graph[i], road_graph_validity[i] = self._encode_map_object(map_data[i])

        # use the same normalization as for the other coordinates
        road_graph[..., :2] = (road_graph[..., :2] - np.array(FEATURE_MEANS)[:2]) / (
            2 * np.array(FEATURE_STD)[:2]
        )

        return SceneTensor(
            tensor=scene_tensor.astype(np.float32),
            validity=scene_tensor_validity.astype(np.float32),
            road_graph=road_graph.astype(np.float32),
            road_graph_validity=road_graph_validity.astype(np.float32),
        )

    def _encode_map_object(self, prepared_map_object: PreparedMapObject):
        # coords = prepared_map_object.coords
        n_points = prepared_map_object.coords.shape[0]
        coords = np.zeros((self._map_num_points_polylines, 2), dtype=np.float32)
        coords_valid = np.zeros_like(coords)

        # if n_points > self._map_num_points_polylines:
        new_points = np.linspace(0, n_points - 1, self._map_num_points_polylines)
        coords[:, 0] = np.interp(
            new_points, np.arange(0, n_points), prepared_map_object.coords[:, 0]
        )
        coords[:, 1] = np.interp(
            new_points, np.arange(0, n_points), prepared_map_object.coords[:, 1]
        )
        coords_valid[:, :] = 1
        # elif n_points < self._map_num_points_polylines:
        #     coords[:n_points] = prepared_map_object.coords
        #     coords_valid[:n_points] = 1.0

        type_ = prepared_map_object.object_type
        # one hot encode object type
        onehot = np.zeros((self._map_num_points_polylines, len(self._map_features)))
        idx = self._map_features.index(type_)
        onehot[:, idx] = 1
        onehot_valid = np.ones_like(onehot) * coords_valid[:, 0:1]  # n_points x 1
        # add speed limit
        sl = prepared_map_object.speed_limit
        speed_limit = np.full(
            shape=(self._map_num_points_polylines, 1),
            dtype=np.float32,
            fill_value=sl or -1,
        )
        speed_limit_valid = coords_valid[:, 0:1] * (1 if sl is not None else 0)

        map_array = np.concatenate([coords, onehot, speed_limit], axis=1)
        map_array_valid = np.concatenate(
            [coords_valid, onehot_valid, speed_limit_valid], axis=1
        )

        return map_array, map_array_valid