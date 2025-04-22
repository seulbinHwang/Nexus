import sys
sys.path.append('/cpfs01/user/yenaisheng/Nexus')
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan_extent.planning.scenario_builder.prepared_scenario import (
    NpEgoState,
    PreparedScenario,
)
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_vector_feature_builder_v2 import (
    HorizonVectorFeatureBuilderV2,
    HorizonVectorV2,
)
from nuplan_extent.planning.training.preprocessing.features.scene_tensor import (SceneTensor,
    SCENE_TENSOR_FEATURES,N_SCENE_TENSOR_FEATURES,  
    FEATURE_MEANS,FEATURE_STD,
    VEH_PARAMS,EGO_WIDTH,EGO_LENGTH,CLASSES,
    unnormalize_roadgraph,decode_scene_tensor,encode_agent,encode_ego) 
from typing import List, Dict
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.cuda.amp import autocast
from collections import Counter

@dataclass
class SceneTensorWrapper:
    scene_tensor: SceneTensor
    trans: np.ndarray
    center: np.ndarray
    offset: np.ndarray
    idx2token: List[str]
    log_name: str
    scenario_token: str
    map_name: str
    lidar_pc: List[str]
    agent_type: List[str]

# exp: visualize_scene_tensor(scene_tensor)
def visualize_scene_tensor(scene_tensor, output_gif="scene.gif", fps=2):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import imageio
    """
    Visualize vehicle bounding boxes in a scene and generate a GIF animation.
    
    Args:
        scene_tensor: A SceneTensor object, which contains:
            - tensor: shape (NA, 21, 8), representing the states of each vehicle across 21 time steps.
                      The dimensions are: x, y, cos(yaw), sin(yaw), vx, vy, length, width.
            - validity: shape (NA, 21), indicating whether each vehicle is valid at each time step.
            - road_graph and road_graph_validity can be extended if needed (not used here).
        output_gif: Filename for the output GIF, default is "scene.gif".
        fps: Frames per second for the GIF animation, default is 2.
    """

    # Get vehicle state and validity info
    # Assumes decode_scene_tensor returns an object with 'tensor' and 'validity' attributes
    tensor = decode_scene_tensor(scene_tensor.tensor)  # shape: (NA, 21, 8)
    validity = scene_tensor.validity  # shape: (NA, 21)

    num_vehicles, num_timesteps, _ = tensor.shape

    # Compute global coordinate bounds to keep axis consistent across frames
    all_x = tensor[:, :, 0]
    all_y = tensor[:, :, 1]
    margin = 10  # extra margin
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin

    frames = []  # store each image frame

    for t in range(num_timesteps):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Time step {t}")
        
        # Iterate over each vehicle
        for i in range(num_vehicles):
            # Skip if the vehicle is not valid at this time step
            if validity[i, t] < 0.5:
                continue

            # Extract vehicle state info
            x = tensor[i, t, 0]
            y = tensor[i, t, 1]
            cos_yaw = tensor[i, t, 2]
            sin_yaw = tensor[i, t, 3]
            # vx and vy are not used for plotting
            length = tensor[i, t, 6]
            width = tensor[i, t, 7]
            # Compute yaw angle in radians
            yaw = np.arctan2(sin_yaw, cos_yaw)
            yaw_deg = np.degrees(yaw)

            # Compute parameters for the bounding box:
            # matplotlib.patches.Rectangle requires bottom-left corner,
            # but the state provides center point, so we convert it:
            half_length = length / 2
            half_width = width / 2
            # Use rotated rectangle formula to compute bottom-left corner
            bottom_left_x = x - (half_length * np.cos(yaw) - half_width * np.sin(yaw))
            bottom_left_y = y - (half_length * np.sin(yaw) + half_width * np.cos(yaw))

            # Create a rectangle patch to represent the vehicle bbox (no fill, red border)
            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                length, width,
                angle=yaw_deg,
                edgecolor='red',
                facecolor='none',
                lw=2
            )
            ax.add_patch(rect)
            # Draw vehicle center point for reference
            ax.plot(x, y, 'bo')
        
        # Convert current plot to image frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    # Use imageio to save all frames as a GIF
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"GIF animation saved as {output_gif}")

# exp: visualize_scene_tensor_global(scene_tensor, trans, center, offset)
def visualize_scene_tensor_global(scene_tensor, local_to_global, center, global_offset, output_gif="scene_global.gif", fps=2):
    """
    Visualize a decoded scene_tensor in global coordinates and generate a GIF animation.

    Args:
        scene_tensor: A decoded SceneTensor object containing:
            - tensor: numpy array of shape (NA, 21, 8), representing the state of each vehicle
                      across 21 time steps. Dimensions: x, y, cos(yaw), sin(yaw), vx, vy, length, width.
            - validity: numpy array of shape (NA, 21), indicating vehicle presence at each timestep.
        local_to_global: A 3x3 numpy array representing the homogeneous transformation matrix
                         from local to global coordinates.
        center: A reference object in global coordinates. Must have a `heading` attribute (in radians),
                used to adjust vehicle orientation.
        global_offset: A numpy array of shape (2,), representing the x, y translation in global coordinates.
        output_gif: Filename for the output GIF. Default is "scene_global.gif".
        fps: Frames per second of the GIF. Default is 2.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import imageio

    # Get original tensor and validity (tensor data is in local coordinates)
    tensor = decode_scene_tensor(scene_tensor.tensor)  # shape: (NA, 21, 8)
    validity = scene_tensor.validity  # shape: (NA, 21)
    num_vehicles, num_timesteps, _ = tensor.shape

    # --- Coordinate Transformation ---
    # 1. Transform position (x, y) using homogeneous coordinates
    xy_local = tensor[:, :, 0:2]
    ones = np.ones((num_vehicles, num_timesteps, 1))
    xy_local_h = np.concatenate([xy_local, ones], axis=-1)  # shape: (NA, T, 3)
    xy_global_h = np.einsum('ij,ntj->nti', local_to_global, xy_local_h)
    xy_global = xy_global_h[:, :, :2]
    xy_global = xy_global + global_offset  # Apply global offset (broadcasted)

    # 2. Transform orientation (yaw) by adding center heading
    local_yaw = np.arctan2(tensor[:, :, 3], tensor[:, :, 2])  # shape: (NA, T)
    global_yaw = local_yaw + center.heading
    global_cos = np.cos(global_yaw)
    global_sin = np.sin(global_yaw)

    # 3. Optional: Transform velocity direction only (rotation without translation)
    local_vel = tensor[:, :, 4:6]
    ch = center.heading
    rot = np.array([[np.cos(ch), -np.sin(ch)],
                    [np.sin(ch),  np.cos(ch)]])
    global_vel = np.einsum('ij,ntj->nti', rot, local_vel)

    # Build the transformed tensor
    global_tensor = tensor.copy()
    global_tensor[:, :, 0:2] = xy_global      # update positions
    global_tensor[:, :, 2] = global_cos       # update cos(yaw)
    global_tensor[:, :, 3] = global_sin       # update sin(yaw)
    global_tensor[:, :, 4:6] = global_vel     # update velocity

    # --- Visualization ---
    all_x = global_tensor[:, :, 0]
    all_y = global_tensor[:, :, 1]
    margin = 10
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin

    frames = []

    for t in range(num_timesteps):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Global Time Step {t}")

        for i in range(num_vehicles):
            if validity[i, t] < 0.5:
                continue

            x = global_tensor[i, t, 0]
            y = global_tensor[i, t, 1]
            cos_yaw = global_tensor[i, t, 2]
            sin_yaw = global_tensor[i, t, 3]
            length = global_tensor[i, t, 6]
            width = global_tensor[i, t, 7]
            yaw = np.arctan2(sin_yaw, cos_yaw)
            yaw_deg = np.degrees(yaw)

            half_length = length / 2
            half_width = width / 2
            bottom_left_x = x - (half_length * np.cos(yaw) - half_width * np.sin(yaw))
            bottom_left_y = y - (half_length * np.sin(yaw) + half_width * np.cos(yaw))

            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                length, width,
                angle=yaw_deg,
                edgecolor='blue',
                facecolor='none',
                lw=2
            )
            ax.add_patch(rect)
            ax.plot(x, y, 'ro')  # Plot vehicle center

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"Global GIF saved to {output_gif}")

# exp: visualize_scene_template(scene_template)
def visualize_scene_template(scene_template, output_gif="scene_template.gif", fps=2):
    """
    Visualizes a scene_template dictionary as a GIF animation.

    Args:
        scene_template (dict): A dictionary describing a scene, containing:
            - scene_metadata: Dictionary with log name, map name, and initial token.
            - frames: List of frame data, where each frame is a dictionary containing:
                - token: The unique identifier of the frame (e.g., lidar token).
                - ego_status: Dictionary with ego_pose (x, y, heading in radians),
                              ego_velocity (vx, vy), and optionally ego_acceleration.
                - annotations: Dictionary with:
                    - boxes: numpy array of shape (N, 7), each row is [x, y, z, l, w, h, heading]
                    - velocity_3d: numpy array of shape (N, 3), each row is [vx, vy, vz]
                    - names: List of strings (e.g., "vehicle")
                    - track_tokens: List of object track tokens
        output_gif (str): Output filename for the GIF. Default is "scene_template.gif".
        fps (int): Frames per second for the GIF. Default is 2.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import imageio

    frames = scene_template["frames"]

    # Determine plot bounds from all ego and object positions
    all_x = []
    all_y = []
    for frame in frames:
        ego_pose = frame["ego_status"]["ego_pose"]
        all_x.append(ego_pose[0])
        all_y.append(ego_pose[1])
        boxes = frame["annotations"]["boxes"]
        if boxes.size > 0:
            all_x.extend(boxes[:, 0])
            all_y.extend(boxes[:, 1])

    margin = 10
    x_min = min(all_x) - margin
    x_max = max(all_x) + margin
    y_min = min(all_y) - margin
    y_max = max(all_y) + margin

    gif_frames = []
    for idx, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"Frame {idx}")

        # --- Draw ego vehicle ---
        ego_status = frame["ego_status"]
        ego_x, ego_y, ego_heading = ego_status["ego_pose"]
        ego_vel = ego_status["ego_velocity"]

        # Assume default ego dimensions if not provided
        ego_length = 4.0
        ego_width = 2.0
        half_length = ego_length / 2.0
        half_width = ego_width / 2.0

        # Compute ego's bottom-left corner from center + heading
        ego_bl_x = ego_x - (half_length * np.cos(ego_heading) - half_width * np.sin(ego_heading))
        ego_bl_y = ego_y - (half_length * np.sin(ego_heading) + half_width * np.cos(ego_heading))
        rect_ego = patches.Rectangle((ego_bl_x, ego_bl_y), ego_length, ego_width,
                                     angle=np.degrees(ego_heading),
                                     edgecolor="blue", facecolor="none", lw=2, label="ego")
        ax.add_patch(rect_ego)
        ax.plot(ego_x, ego_y, "bo")  # Plot ego center
        ax.arrow(ego_x, ego_y, ego_vel[0], ego_vel[1],
                 head_width=0.5, head_length=0.5, fc="blue", ec="blue")

        # --- Draw other vehicles ---
        ann = frame["annotations"]
        boxes = ann["boxes"]
        velocities = ann["velocity_3d"]

        if boxes.size > 0:
            for i in range(boxes.shape[0]):
                x, y, z, l, w, h, heading = boxes[i]
                half_l = l / 2.0
                half_w = w / 2.0
                bl_x = x - (half_l * np.cos(heading) - half_w * np.sin(heading))
                bl_y = y - (half_l * np.sin(heading) + half_w * np.cos(heading))
                rect = patches.Rectangle((bl_x, bl_y), l, w,
                                         angle=np.degrees(heading),
                                         edgecolor="red", facecolor="none", lw=2)
                ax.add_patch(rect)
                ax.plot(x, y, "ro")
                vx, vy, vz = velocities[i]
                ax.arrow(x, y, vx, vy, head_width=0.5, head_length=0.5, fc="red", ec="red")

        # Optional: display frame token for debugging
        ax.text(x_min + 5, y_max - 5, f"Token: {frame['token']}", color="black", fontsize=12)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(image)
        plt.close(fig)

    # Export GIF
    imageio.mimsave(output_gif, gif_frames, fps=fps)
    print(f"GIF saved to {output_gif}")

def convert_scene_tensor_wrapper_to_scene_template(scene_tensor_wrapper):
    """
    Converts the scene_tensor in the scene_tensor_wrapper from the local coordinate system to the global coordinate system,
    and constructs a dictionary in the scene_template format, with all position-related data kept in the global frame.

    Notes:
      - The ego vehicle is the agent with idx == 0 in scene_tensor, and its global info is used directly.
      - idx2token contains tokens for all valid agents (including ego), in the same order as the valid agents in scene_tensor.
      - Annotations only include non-ego agents (i >= 1) that have corresponding token mappings.

    Input: scene_tensor_wrapper structure includes:
        scene_tensor: SceneTensor object, containing:
            - tensor: np.ndarray, shape (NA, 21, 8), each vehicle's state over 21 frames, with dimensions:
                      [x, y, cos(yaw), sin(yaw), vx, vy, length, width]
            - validity: np.ndarray, shape (NA, 21), indicating whether each vehicle is valid in each frame
        trans: np.ndarray of shape (3, 3), the homogeneous transformation matrix from local to global
        center: an object with a 'heading' attribute (in radians), used for angle conversion
                (Note: ego global info is not derived from center.x/y/heading, but from the ego agent in scene_tensor)
        offset: np.ndarray of shape (2,), xy offset in the global frame
        idx2token: List[str], token list for all valid agents (including ego)
        log_name: str
        scenario_token: str
        map_name: str
        lidar_pc: List[str], length 21, corresponding lidar_pc token for each frame

    Output: scene_template format:
        {
            scene_metadata: {
                log_name: ...,
                map_name: ...,
                initial_token: ...,
            },
            frames: [
                {
                    token: <lidar_pc token>,
                    ego_status: {
                        ego_pose: np.array([x, y, heading]),         # global coordinates (ego is agent idx==0)
                        ego_velocity: np.array([vx, vy]),            # global coordinates
                        ego_acceleration: np.array([ax, ay]),        # global coordinates (currently set to 0)
                    },
                    annotations: {
                        boxes: np.array([
                            [x, y, z, l, w, h, heading],  # global coordinates, z and h set to 0
                            ...
                        ]),
                        velocity_3d: np.array([
                            [vx, vy, vz],  # global coordinates, vz set to 0
                            ...
                        ]),
                        names: [ "vehicle", ... ],
                        track_tokens: [ token, ... ]
                    }
                },
                ...
            ]
        }
    """
    # Unpack fields from scene_tensor_wrapper
    st = scene_tensor_wrapper.scene_tensor  # SceneTensor object
    trans = scene_tensor_wrapper.trans      # 3x3 transformation matrix
    center = scene_tensor_wrapper.center    # Only the 'heading' is used
    offset = scene_tensor_wrapper.offset    # Global xy offset, shape (2,)
    idx2token = scene_tensor_wrapper.idx2token
    log_name = scene_tensor_wrapper.log_name
    scenario_token = scene_tensor_wrapper.scenario_token
    map_name = scene_tensor_wrapper.map_name
    lidar_pc = scene_tensor_wrapper.lidar_pc
    agent_type = scene_tensor_wrapper.agent_type

    tensor = decode_scene_tensor(st.tensor)  # shape: (NA, 21, 8)
    validity = st.validity                   # shape: (NA, 21)
    num_vehicles, num_timesteps, _ = tensor.shape

    # --- Build token mapping for valid agents ---
    valid_indices = [i for i in range(num_vehicles) if np.any(validity[i, :] >= 0.5)]
    agent_token_mapping = {i: idx2token[k] for k, i in enumerate(valid_indices)}

    # --- 1. Convert local to global coordinates ---
    # Transform positions using homogeneous coordinates and add offset
    xy_local = tensor[:, :, 0:2]  # (NA, 21, 2)
    ones = np.ones((num_vehicles, num_timesteps, 1))
    xy_local_h = np.concatenate([xy_local, ones], axis=-1)  # (NA, 21, 3)
    xy_global_h = np.einsum('ij,ntj->nti', trans, xy_local_h)
    xy_global = xy_global_h[:, :, :2] + offset

    global_tensor = tensor.copy()
    global_tensor[:, :, 0:2] = xy_global

    # Convert heading: yaw = arctan2(sin, cos) + heading offset
    local_yaw = np.arctan2(tensor[:, :, 3], tensor[:, :, 2])
    global_yaw = local_yaw + center.heading
    global_tensor[:, :, 2] = np.cos(global_yaw)
    global_tensor[:, :, 3] = np.sin(global_yaw)

    # Rotate velocity to global frame
    local_vel = tensor[:, :, 4:6]
    ch = center.heading
    rot = np.array([[np.cos(ch), -np.sin(ch)],
                    [np.sin(ch),  np.cos(ch)]])
    global_vel = np.einsum('ij,ntj->nti', rot, local_vel)
    global_tensor[:, :, 4:6] = global_vel

    # --- 2. Construct scene_template ---
    frames_list = []
    for t in range(num_timesteps):
        # Ego information from agent idx==0
        ego_x = global_tensor[0, t, 0]
        ego_y = global_tensor[0, t, 1]
        ego_yaw = np.arctan2(global_tensor[0, t, 3], global_tensor[0, t, 2])
        ego_pose = np.array([ego_x, ego_y, ego_yaw])
        ego_velocity = global_tensor[0, t, 4:6]
        ego_acceleration = np.array([0.0, 0.0])  # Set to 0 for now

        boxes = []
        velocities = []
        names = []
        track_tokens = []

        for i in range(1, num_vehicles):
            if validity[i, t] < 0.5:
                continue
            if i not in agent_token_mapping:
                continue
            x = global_tensor[i, t, 0]
            y = global_tensor[i, t, 1]
            length = global_tensor[i, t, 6]
            width = global_tensor[i, t, 7]
            yaw = np.arctan2(global_tensor[i, t, 3], global_tensor[i, t, 2])
            box = [x, y, 0.0, length, width, 0.0, yaw]
            boxes.append(box)
            vx = global_tensor[i, t, 4]
            vy = global_tensor[i, t, 5]
            velocities.append([vx, vy, 0.0])
            names.append(agent_type[i])
            token = agent_token_mapping[i]
            track_tokens.append(token)

        frame_dict = dict(
            token=lidar_pc[t],
            ego_status=dict(
                ego_pose=ego_pose,
                ego_velocity=ego_velocity,
                ego_acceleration=ego_acceleration,
            ),
            annotations=dict(
                boxes=np.array(boxes) if boxes else np.empty((0, 7)),
                velocity_3d=np.array(velocities) if velocities else np.empty((0, 3)),
                names=names,
                track_tokens=track_tokens,
            )
        )
        frames_list.append(frame_dict)

    scene_template = dict(
        scene_metadata=dict(
            log_name=log_name,
            map_name=map_name,
            initial_token=scenario_token,
        ),
        frames=frames_list
    )

    return scene_template


def update_config_for_training(cfg: DictConfig) -> None:
    # Set the struct to False to allow changes to the config
    OmegaConf.set_struct(cfg, False)
    # You can change settings in cfg here

    # choose your scenario log name and lidar_pc token
    cfg.scenario_filter.log_names = ['log_1', 'log_2']
    cfg.scenario_filter.scenario_tokens = ['lidar_pc_1', 'lidar_pc_2']
    # specify the root directories for nuplan data and maps
    cfg.scenario_builder.scenario_mapping.data_root = 'NUPLAN_DATA_ROOT'
    cfg.scenario_builder.scenario_mapping.map_root = 'NUPLAN_MAP_ROOT'

    # Save all interpolations and remove keys that were only used for interpolation and have no further use.
    OmegaConf.resolve(cfg)
    # Freeze the config to avoid accidental changes
    OmegaConf.set_struct(cfg, True)
    return cfg

def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker) 

def get_features_from_scenario(builder: AbstractFeatureBuilder, scenario: AbstractScenario) -> SceneTensorWrapper:   
    vector_data=HorizonVectorFeatureBuilderV2.get_features_from_scenario(builder, scenario, iteration=0)
    return compute_features(builder, vector_data)

def compute_features(builder: AbstractFeatureBuilder, vector_data: HorizonVectorV2) -> SceneTensorWrapper:
    vehicles = vector_data.data["agents"]["VEHICLE"]  # type: ignore
    peds = vector_data.data["agents"]["PEDESTRIAN"]  # type: ignore
    bic = vector_data.data["agents"]["BICYCLE"]  # type: ignore
    ego = vector_data.data["ego"]  # t x 7 tensor
    track_token_id_mapping = vector_data.data['track_token_id_mapping']
    for key in track_token_id_mapping:
        track_token_id_mapping[key] = {int(v): k for k, v in track_token_id_mapping[key].items()}
    map_name = vector_data.data['map_name']
    log_name = vector_data.data['log_name']
    lidar_pc = vector_data.data['lidarpc']
    scenario_token = vector_data.data['scenario_token']
    offset = vector_data.data['offset']
    center = vector_data.data['center']
    trans = vector_data.data['trans']

    n_timesteps = len(vehicles)

    # get all unique track ids
    ids_veh = np.unique(vehicles[..., 0])
    ids_ped = np.unique(peds[..., 0])
    ids_bic = np.unique(bic[..., 0])
    idx2token = ['ego']
    agent_type = ['vehicle']
    # n_agents = len(ids_veh) + len(ids_ped) + len(ids_bic)
    # n_max_agents = sum(builder._num_max_agents)
    n_max_agents = 128
    # create nan tensor
    scene_tensor = np.zeros((n_max_agents, n_timesteps, N_SCENE_TENSOR_FEATURES))
    scene_tensor_validity = np.zeros((n_max_agents, n_timesteps))

    n_agents = 0
    for i, id_ in enumerate(ids_veh):
        if i >= builder._num_max_agents[0]:
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
        idx2token.append(track_token_id_mapping['VEHICLE'][int(id_)])
        agent_type.append('vehicle')
        n_agents += 1

    for i, id_ in enumerate(ids_ped):
        if i >= builder._num_max_agents[1]:
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
        idx2token.append(track_token_id_mapping['PEDESTRIAN'][int(id_)])
        agent_type.append('pedestrian')
        n_agents += 1

    for i, id_ in enumerate(ids_bic):
        if i >= builder._num_max_agents[2]:
            break
        mask = bic[..., 0] == id_
        time_validity_ = mask.sum(axis=1).astype(bool)
        if time_validity_.sum() < 5:
            continue
        scene_tensor[n_agents, time_validity_] = encode_agent("BICYCLE", bic[mask])  # type: ignore
        scene_tensor_validity[n_agents] = time_validity_
        idx2token.append(track_token_id_mapping['BICYCLE'][int(id_)])
        agent_type.append('bicycle')
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
            builder._map_num_max_polylines,
            builder._map_num_points_polylines,
            builder._map_dim_points_polyline,
        ),
        dtype=np.float32,
    )
    road_graph_validity = np.zeros_like(road_graph)
    map_data = vector_data.data["map"]

    for i in range(min(builder._map_num_max_polylines, len(map_data))):
        road_graph[i], road_graph_validity[i] = builder._encode_map_object(map_data[i])

    # use the same normalization as for the other coordinates
    road_graph[..., :2] = (road_graph[..., :2] - np.array(FEATURE_MEANS)[:2]) / (
        2 * np.array(FEATURE_STD)[:2]
    )

    scene_tensor = SceneTensor(
        tensor=scene_tensor.astype(np.float32),
        validity=scene_tensor_validity.astype(np.float32),
        road_graph=road_graph.astype(np.float32),
        road_graph_validity=road_graph_validity.astype(np.float32),
    )
    scene_tensor_wrapper = SceneTensorWrapper(
        scene_tensor=scene_tensor,
        trans=trans,
        center=center,
        offset=offset,
        idx2token=idx2token,
        log_name=log_name,
        scenario_token=scenario_token,
        map_name=map_name,
        lidar_pc=lidar_pc,
        agent_type=agent_type
    )
    return scene_tensor_wrapper
    # convert_scene_tensor_wrapper_to_scene_template(scene_tensor_wrapper)

def create_intent_attack_mask(scene_tensor_wrapper, n_chosen_prob=0.3):
    """
    针对意图攻击任务生成 task mask，同时对 scene_tensor_wrapper 的 tensor 进行修改：
      - 找到 ego (idx==0) 到其他 agent 第一个 timestep 距离最短的 agent，
        并交换 agent 1 与该 agent 的顺序。
      - 修改交换后的 agent 1 的最后一个 timestep 的前4个特征为 ego 的对应值，
        并确保其在最后 timestep 有效。
    """
    scene_tensor_features = scene_tensor_wrapper.scene_tensor
    task_mask = torch.zeros_like(scene_tensor_features.tensor)

    # 获取 ego 车在第一个 timestep 的位置
    ego_pos = scene_tensor_features.tensor[0, 0, :2]
    valid_mask = scene_tensor_features.validity[1:, 0]  # 取除 ego 之外的 agent

    # 计算其他 agent 在第五个 timestep 与 ego 的距离
    distances = torch.norm(scene_tensor_features.tensor[1:, 4, :2] - ego_pos, dim=1)
    distances = torch.where(valid_mask.bool(), distances, torch.tensor(1e6, dtype=torch.float32, device=scene_tensor_features.tensor.device))

    # 找到最近的 agent
    closest_index = torch.argmin(distances[:Counter(scene_tensor_wrapper.agent_type)['vehicle']-1]).item() + 1

    # 交换 agent 1 和 closest_index 对应的 agent 数据（包括 tensor 和 validity）
    temp_tensor = scene_tensor_features.tensor[1].clone()
    scene_tensor_features.tensor[1] = scene_tensor_features.tensor[closest_index]
    scene_tensor_features.tensor[closest_index] = temp_tensor

    temp_valid = scene_tensor_features.validity[1].clone()
    scene_tensor_features.validity[1] = scene_tensor_features.validity[closest_index]
    scene_tensor_features.validity[closest_index] = temp_valid

    # 更新 idx2token 和 agent_type：交换位置1与closest_index对应的条目
    if len(scene_tensor_wrapper.idx2token) > closest_index:
        scene_tensor_wrapper.idx2token[1], scene_tensor_wrapper.idx2token[closest_index] = (
            scene_tensor_wrapper.idx2token[closest_index],
            scene_tensor_wrapper.idx2token[1],
        )
    if len(scene_tensor_wrapper.agent_type) > closest_index:
        scene_tensor_wrapper.agent_type[1], scene_tensor_wrapper.agent_type[closest_index] = (
            scene_tensor_wrapper.agent_type[closest_index],
            scene_tensor_wrapper.agent_type[1],
        )

    # 应用 intent attack：将 agent 1 最后一个 timestep 的前4个特征置为 ego 对应值
    scene_tensor_features.tensor[1, -1, :4] = scene_tensor_features.tensor[0, -1, :4].clone()
    scene_tensor_features.validity[1, -1] = 1

    # 设置 task mask：前5个 timestep 全部置1，最后 timestep 的前2个 agent 置1
    task_mask[:, :5, :] = 1
    task_mask[:2, -1] = 1

    return task_mask

def create_task_mask(scene_tensor_wrapper, n_chosen_prob=0.3):
    """
    针对 scene_gen 任务生成 task mask：
      根据 agent 的有效性（任意 timestep 有效），按概率 mask 有效的 agent。
    """
    scene_tensor_features = scene_tensor_wrapper.scene_tensor
    n_valid_agents = scene_tensor_features.validity.any(dim=-1).sum().item()
    task_mask = torch.zeros_like(scene_tensor_features.tensor)

    mask = torch.rand(n_valid_agents, device=scene_tensor_features.tensor.device) < n_chosen_prob
    task_mask[:n_valid_agents] = mask.unsqueeze(-1).unsqueeze(-1)
    
    return task_mask

def unsqueeze(features: Dict):
    scene_tensor = features['scene_tensor']
    task_mask = features['task_mask']
    
    # Extract relevant tensors from scene_tensor
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    # Use unsqueeze to add an extra dimension at position 0
    tensor = tensor.unsqueeze(0)  # Correct method is unsqueeze
    validity = validity.unsqueeze(0)
    road_graph = road_graph.unsqueeze(0)
    road_graph_validity = road_graph_validity.unsqueeze(0)
    task_mask = task_mask.unsqueeze(0)

    return {
        'task_mask': task_mask,
        'scene_tensor': SceneTensor(tensor, validity, road_graph, road_graph_validity)
    }

def squeeze(features: Dict):
    scene_tensor = features['scene_tensor']
    task_mask = features['task_mask']
    
    # Extract relevant tensors from scene_tensor
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    # Use squeeze to remove dimensions of size 1
    tensor = tensor.squeeze(0)  # Removes the dimension at position 0 if it's size 1
    validity = validity.squeeze(0)
    road_graph = road_graph.squeeze(0)
    road_graph_validity = road_graph_validity.squeeze(0)
    task_mask = task_mask.squeeze(0)

    # Return the modified tensors in the same structure
    return {
        'task_mask': task_mask,
        'scene_tensor': SceneTensor(tensor, validity, road_graph, road_graph_validity)
    }

def generate_new_scene_tensor_wrapper_from_wrapper(scene_tensor_wrapper, model, task):
    """
    输入已生成的 scene_tensor_wrapper，依据 task（"bp", "scene_gen", "intent_attack"）
    构造对应的 task_mask，并生成新的 scene_tensor_wrapper。
    """
    features = {"scene_tensor": scene_tensor_wrapper.scene_tensor}
    
    if task == "bp":
        bp_mask = torch.zeros_like(scene_tensor_wrapper.scene_tensor.tensor)
        bp_mask[:, :5, :] = 1
        features["task_mask"] = bp_mask
    elif task == "scene_gen":
        features["task_mask"] = create_task_mask(scene_tensor_wrapper)
    elif task == "intent_attack":
        features["task_mask"] = create_intent_attack_mask(scene_tensor_wrapper)
    else:
        raise ValueError(f"Unknown task: {task}")

    with torch.no_grad(), autocast(enabled=False):
        model.eval()
        predictions = model.forward_inference(unsqueeze(features))
        model.train()
    predictions['scene_tensor']=features['scene_tensor']
    predictions['scene_tensor'].tensor = predictions['sampled_tensor']
    predictions = squeeze(predictions)
    # 更新 scene_tensor.tensor，同时保持其他字段不变
    new_scene_tensor_wrapper = scene_tensor_wrapper
    new_scene_tensor_wrapper.scene_tensor = predictions['scene_tensor']
    return new_scene_tensor_wrapper

def to_tensor(scene_tensor_wrapper):
    """
    将 SceneTensorWrapper 转换为 torch.Tensor 格式。
    """
    scene_tensor = scene_tensor_wrapper.scene_tensor
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    scene_tensor.tensor = torch.tensor(tensor)
    scene_tensor.validity = torch.tensor(validity)
    scene_tensor.road_graph = torch.tensor(road_graph)
    scene_tensor.road_graph_validity = torch.tensor(road_graph_validity)

    scene_tensor_wrapper.scene_tensor = scene_tensor
    return scene_tensor_wrapper

def to_numpy(scene_tensor_wrapper):
    """
    将 SceneTensorWrapper 转换为 numpy 格式。
    """
    scene_tensor = scene_tensor_wrapper.scene_tensor
    tensor = scene_tensor.tensor.numpy()
    validity = scene_tensor.validity.numpy()
    road_graph = scene_tensor.road_graph.numpy()
    road_graph_validity = scene_tensor.road_graph_validity.numpy()

    scene_tensor.tensor = tensor
    scene_tensor.validity = validity
    scene_tensor.road_graph = road_graph
    scene_tensor.road_graph_validity = road_graph_validity

    scene_tensor_wrapper.scene_tensor = scene_tensor
    return scene_tensor_wrapper
    
if __name__ == '__main__':
    
    with open('Nexus/scripts/generation/config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)
    # specify log_name, lidarpc_token here    
    # cfg = update_config_for_training(cfg)
    worker = build_worker(cfg)
    scenario_builder = build_scenario_builder(cfg)
    scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
    model = build_torch_module_wrapper(cfg.model)
    if cfg.checkpoint:
        my_ckpt = torch.load(cfg.checkpoint, map_location="cpu")
        my_ckpt["state_dict"] = {k[6:]: v for k, v in my_ckpt["state_dict"].items()}
        model.load_state_dict(my_ckpt["state_dict"], strict=True)
    
    feature_builder = model.get_list_of_required_feature()[0]

    for scenario in scenarios:
        scene_tensor_wrapper = get_features_from_scenario(feature_builder, scenario)
                
        #  scene_tensor
        visualize_scene_tensor(scene_tensor_wrapper.scene_tensor)

        scene_tensor_wrapper = to_tensor(scene_tensor_wrapper)
        task = "intent_attack"  # "bp"、"scene_gen"、"intent_attack"
        new_scene_tensor_wrapper = generate_new_scene_tensor_wrapper_from_wrapper(scene_tensor_wrapper, model, task)
        
        new_scene_tensor_wrapper = to_numpy(new_scene_tensor_wrapper)
        
        # scene_template
        new_scene_template = convert_scene_tensor_wrapper_to_scene_template(new_scene_tensor_wrapper)
        visualize_scene_template(new_scene_template)
