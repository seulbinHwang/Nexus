import sys
sys.path.append('your/path/to/Nexus')
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.modeling.types import TargetsType
from nuplan_extent.planning.training.callbacks.utils.visualization_utils import Draw
from nuplan_extent.planning.training.preprocessing.feature_builders.nexus_feature_builder import (
    SceneTensor,
    decode_scene_tensor,
    unnormalize_roadgraph,
)
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_vector_feature_builder_v2 import (
    HorizonVectorFeatureBuilderV2,
    HorizonVectorV2,
)
from nuplan_extent.planning.training.preprocessing.features.scene_tensor import (SceneTensor,
    N_SCENE_TENSOR_FEATURES,  
    FEATURE_MEANS,FEATURE_STD,
    unnormalize_roadgraph,decode_scene_tensor,encode_agent,encode_ego) 
from typing import List
from omegaconf import DictConfig, OmegaConf
import copy
import numpy as np
import torch
from torch.cuda.amp import autocast
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch.utils.data
from PIL import Image,ImageDraw

# Set up color map and canvas parameters for visualization
cmap = plt.get_cmap("tab20")
canvas_size = 800
pixel_size = 0.5

# Update configuration for training, allowing changes and setting scenario and map roots
def update_config(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    # to save time, we can only use 3 db files to test the pipeline
    cfg.scenario_filter.log_names = ['2021.06.11.12.06.26_veh-35_03726_03971', 
                                     '2021.06.11.12.09.55_veh-16_00104_00221',  
                                     '2021.10.22.18.45.52_veh-28_01175_01298']
    # please set the nuplan data root and map root, and the checkpoint path to your local machine
    cfg.scenario_builder.data_root = 'your/path/to/nuplan/dataset'
    cfg.scenario_builder.map_root = 'your/path/to/nuplan/maps'
    cfg.checkpoint = 'your/path/to/checkpoint'
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg

# Build scenarios from configuration using the scenario builder and worker pool
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

# Extract features from a scenario using the provided feature builder
def get_features_from_scenario(builder: AbstractFeatureBuilder, scenario: AbstractScenario) -> SceneTensor:   
    vector_data=HorizonVectorFeatureBuilderV2.get_features_from_scenario(builder, scenario, iteration=0)
    return compute_features(builder, vector_data)

# Compute features for all agents and the ego vehicle, normalize, and encode road graph
def compute_features(builder: AbstractFeatureBuilder, vector_data: HorizonVectorV2) -> SceneTensor:
    vehicles = vector_data.data["agents"]["VEHICLE"]  # Vehicle agent data
    peds = vector_data.data["agents"]["PEDESTRIAN"]  # Pedestrian agent data
    bic = vector_data.data["agents"]["BICYCLE"]  # Bicycle agent data
    ego = vector_data.data["ego"]  # Ego vehicle data
    track_token_id_mapping = vector_data.data['track_token_id_mapping']
    for key in track_token_id_mapping:
        track_token_id_mapping[key] = {int(v): k for k, v in track_token_id_mapping[key].items()}

    n_timesteps = len(vehicles)

    # Get all unique track ids for each agent type
    ids_veh = np.unique(vehicles[..., 0])
    ids_ped = np.unique(peds[..., 0])
    ids_bic = np.unique(bic[..., 0])
    idx2token = ['ego']
    agent_type = ['vehicle']
    n_max_agents = 128
    # Initialize tensors for scene features and validity
    scene_tensor = np.zeros((n_max_agents, n_timesteps, N_SCENE_TENSOR_FEATURES))
    scene_tensor_validity = np.zeros((n_max_agents, n_timesteps))

    n_agents = 0
    # Encode vehicle agents
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


    # Encode the ego vehicle state
    ego_state = encode_ego(ego)
    scene_tensor = np.concatenate(
        [np.expand_dims(ego_state, 0), scene_tensor], axis=0
    )
    # Normalize the scene tensor
    scene_tensor = (scene_tensor - np.array(FEATURE_MEANS)) / (
        2.0 * np.array(FEATURE_STD)
    )
    scene_tensor_validity = np.concatenate(
        [np.ones((1, n_timesteps)), scene_tensor_validity], axis=0
    )

    # Encode road graph features
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
        encoded_rg, encoded_rg_validity = builder._encode_map_object(map_data[i])
        road_graph[i,...,:encoded_rg.shape[-1]], road_graph_validity[i,...,:encoded_rg.shape[-1]] = encoded_rg, encoded_rg_validity

    # Normalize road graph coordinates
    road_graph[..., :2] = (road_graph[..., :2] - np.array(FEATURE_MEANS)[:2]) / (
        2 * np.array(FEATURE_STD)[:2]
    )

    scene_tensor = SceneTensor(
        tensor=scene_tensor.astype(np.float32),
        validity=scene_tensor_validity.astype(np.float32),
        road_graph=road_graph.astype(np.float32),
        road_graph_validity=road_graph_validity.astype(np.float32),
    )
    return scene_tensor
    # convert_scene_tensor_wrapper_to_scene_template(scene_tensor_wrapper)

# Add a batch dimension to all tensors in SceneTensor
def unsqueeze(scene_tensor: SceneTensor):
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    tensor = tensor.unsqueeze(0)
    validity = validity.unsqueeze(0)
    road_graph = road_graph.unsqueeze(0)
    road_graph_validity = road_graph_validity.unsqueeze(0)

    return SceneTensor(tensor, validity, road_graph, road_graph_validity)

# Remove the batch dimension from all tensors in SceneTensor
def squeeze(scene_tensor: SceneTensor):
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    tensor = tensor.squeeze(0)
    validity = validity.squeeze(0)
    road_graph = road_graph.squeeze(0)
    road_graph_validity = road_graph_validity.squeeze(0)

    return SceneTensor(tensor, validity, road_graph, road_graph_validity)

# Convert all arrays in SceneTensor to torch.Tensor and move to the specified device
def to_tensor(scene_tensor, device):
    """
    Convert SceneTensorWrapper to torch.Tensor format.
    """
    tensor = scene_tensor.tensor
    validity = scene_tensor.validity
    road_graph = scene_tensor.road_graph
    road_graph_validity = scene_tensor.road_graph_validity

    scene_tensor.tensor = torch.tensor(tensor, device=device)
    scene_tensor.validity = torch.tensor(validity, device=device)
    scene_tensor.road_graph = torch.tensor(road_graph, device=device)
    scene_tensor.road_graph_validity = torch.tensor(road_graph_validity, device=device)

    return scene_tensor

# Convert all arrays in SceneTensor to numpy.ndarray
def to_numpy(scene_tensor):
    """
    Convert SceneTensorWrapper to numpy format.
    """
    tensor = scene_tensor.tensor.numpy()
    validity = scene_tensor.validity.numpy()
    road_graph = scene_tensor.road_graph.numpy()
    road_graph_validity = scene_tensor.road_graph_validity.numpy()

    scene_tensor.tensor = tensor
    scene_tensor.validity = validity
    scene_tensor.road_graph = road_graph
    scene_tensor.road_graph_validity = road_graph_validity

    return scene_tensor

# Convert world coordinates to pixel coordinates for visualization
def coords_to_pixels(coords):
    return coords / pixel_size + canvas_size / 2

# Generate color gradients for different agent types for visualization
def get_color_gradients(N, alpha=128):
    # alpha is the transparency value (0 is fully transparent, 255 is fully opaque)
    # Create different colormaps
    # Green to Blue (for Sim agents)
    colors_sim = [(0, (0,232,158)), (1, (2,121,255))]
    colors_sim = [(x, tuple(y/255 for y in color)) for x, color in colors_sim]
    cmap_sim = LinearSegmentedColormap.from_list("green_to_blue", colors_sim, N=1000)

    # Orange to Yellow (for AV agents)
    colors_av = [(0, (255,132,8)), (1, (255,235,63))]
    colors_av = [(x, tuple(y/255 for y in color)) for x, color in colors_av]
    cmap_av = LinearSegmentedColormap.from_list("orange_to_yellow", colors_av, N=1000)

    # Red to Purple (for Synthetic agents)
    colors_synthetic = [(0, (255,5,5)), (1, (192,4,67))]
    colors_synthetic = [(x, tuple(y/255 for y in color)) for x, color in colors_synthetic]
    cmap_synthetic = LinearSegmentedColormap.from_list("red_to_purple", colors_synthetic, N=1000)

    # Create gradient data (from 0 to 1, with 256 steps)
    gradient = np.linspace(0, 1, N).reshape(1, -1)

    # Get RGB color values and reshape into a 2D array (N, 3)
    sim_colors = cmap_sim(gradient).reshape(-1, 4)*255  # Convert to RGB
    av_colors = cmap_av(gradient).reshape(-1, 4)*255
    synthetic_colors = cmap_synthetic(gradient).reshape(-1, 4)*255

    for color_list in [sim_colors, av_colors, synthetic_colors]:
        # Set alpha value
        color_list[:, -1] = alpha
    # Return the color lists
    sim_colors = sim_colors.astype('uint8')
    sim_colors = [tuple(color) for color in sim_colors]
    av_colors = av_colors.astype('uint8')
    av_colors = [tuple(color) for color in av_colors]
    synthetic_colors = synthetic_colors.astype('uint8')
    synthetic_colors = [tuple(color) for color in synthetic_colors]

    return av_colors,sim_colors,synthetic_colors

# Perform linear interpolation for scene tensor or mask data for smooth visualization
def linear_interpolation(data:np.ndarray,task,N=3) -> np.ndarray:
    def interpolate(data,N):
        # data: B,A,T,C
        data=data.swapaxes(-2,0) # T,A,B,C
        extend_data=[data[0:1]]
        for t in range(1,data.shape[0]):
            fore=extend_data[-1][-1] # A,B,C
            latter=data[t] # A,B,C
            slices=np.linspace(fore,latter,N+2) # N+2,A,B,C
            unvalid=np.logical_or(np.all(np.abs(fore)<1e-6,axis=-1),np.all(np.abs(latter)<1e-6,axis=-1)) # A,B
            slices[:-1,unvalid]=0
            extend_data.append(slices[1:])
        extend_data=np.concatenate(extend_data,axis=0)
        extend_data=extend_data.swapaxes(-2,0)
        return extend_data
    if task == "scene_tensor":
        extend_data=interpolate(data,N)
    elif task == 'mask':
        # there are 2 types of mask: valid_mask (B,NA,NT) and task_mask(B,NA,NT,C)
        type_='task_mask'
        if data.ndim==3:
            type_='valid_mask'
            data=np.expand_dims(data,axis=-1)
        extend_data=interpolate(data,N)
        if type_=='valid_mask':
            extend_data=np.squeeze(extend_data,axis=-1)
        extend_data[np.abs(extend_data-1)>1e-4]=0
    else:
        raise ValueError(f"task {task} not supported")

    return extend_data

# Visualize the scene tensor and predictions, generate and save GIFs for each visualization type
def visualize_scene_tensor(
    features: SceneTensor,
    predictions: TargetsType,
    task: str,
) -> None:
    """
    Visualize the scene tensor and model predictions, generate GIFs for each visualization type (prediction, ground truth, condition).
    """
    scene_tensor_features = features
    valid_mask = scene_tensor_features.validity.cpu().numpy()
    rg = scene_tensor_features.road_graph
    rg = unnormalize_roadgraph(rg.cpu())
    rgv = scene_tensor_features.road_graph_validity.cpu()
    rg_pixels = coords_to_pixels(rg[..., :2])  # b, n_lanes, n_points, n_dim
    rg_pixels_valid = rgv[..., :2]  # b, n_lanes, n_points, n_dim
    sampled_tensor = predictions['sampled_tensor']
    task_mask = predictions['task_mask']

    # Instantiate drawing functions
    draw_bev_dot = Draw.draw_bev_dot
    draw_bev_bboxes=Draw.draw_bev_bboxes
    fuse_images=Draw.fuse_images

    original_scene_tensor = decode_scene_tensor(scene_tensor_features.tensor.cpu().numpy())
    sampled_tensor = decode_scene_tensor(sampled_tensor.cpu().numpy())
    
    # Linear interpolation for smooth visualization
    N=3  # gt1, inter=1/N*gt1+(1-1/N)*gt2, gt2
    original_scene_tensor = linear_interpolation(original_scene_tensor,task="scene_tensor",N=N)
    sampled_tensor = linear_interpolation(sampled_tensor,task="scene_tensor",N=N)
    valid_mask = linear_interpolation(valid_mask,task='mask',N=N)
    task_mask = linear_interpolation(task_mask.cpu().numpy(),task='mask',N=N) # dim as scene_tensor
    
    n_timestamps = original_scene_tensor.shape[2]

    av_gradient, env_gradient, synthetic_gradient = get_color_gradients(n_timestamps)
    vehicle_color = (50, 50, 50, 255)  # Black
    line_color = (122, 120, 120,255)  # Grey, fully opaque

    for batch_idx in range(sampled_tensor.shape[0]):
        # Draw the road graph
        pred_canvas=Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
        draw=ImageDraw.Draw(pred_canvas)
        for line_idx in range(len(rg_pixels[batch_idx])):
            if not rg_pixels_valid[batch_idx, line_idx].any():
                continue
            points = rg_pixels[
                batch_idx,
                line_idx,
                rg_pixels_valid[batch_idx, line_idx].all(axis=-1),
            ].numpy().astype(np.int32)
            points = [tuple(point) for point in points]
            draw.line(points, fill=line_color , width=2)
        gt_canvas = pred_canvas.copy()
        condition_canvas = pred_canvas.copy()

        images = [[], [], []]  # Store frames for prediction, ground truth, and condition GIFs
        for t_ in range(n_timestamps):
            pred_overlay = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
            gt_overlay = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
            condition_overlay = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
            for agent_idx in range(valid_mask.shape[1]):
                if not valid_mask[batch_idx, agent_idx, t_]:
                    continue
                # Select color gradient based on agent type
                if agent_idx == 0:
                    color = av_gradient[t_]  # AV agents use orange-yellow gradient
                elif False: # whether agent is adv agent
                    color = synthetic_gradient[t_]  # Synthetic agents use red-purple gradient
                else:
                    color = env_gradient[t_]  # Environment simulation agents use green-blue gradient
                # import pdb; pdb.set_trace()

                for i, (st, canvas, overlay) in enumerate(
                    [
                        (original_scene_tensor, gt_canvas, gt_overlay),
                        (sampled_tensor, pred_canvas, pred_overlay),
                        # (original_scene_tensor, pred_canvas, pred_overlay),
                    ]
                ):
                    padding = st.shape[-1] - 6
                    x, y = coords_to_pixels(st[batch_idx, agent_idx, t_, :2])
                    l, w = st[batch_idx, agent_idx, t_, [4 + padding, 5 + padding]] / pixel_size

                    cosh = st[batch_idx, agent_idx, t_, 2]
                    sinh = st[batch_idx, agent_idx, t_, 3]
                    heading = np.arctan2(sinh, cosh)
                    is_ego = agent_idx == 0
                    if task == "scene_gen" and i == 1 and t_ < 6 * (N+1) - N and not task_mask[batch_idx, agent_idx, t_, 0] :
                        continue
                    if not i == 0 and not valid_mask[batch_idx, agent_idx, :6 * (N+1)-N].any():
                        continue
                    if task == "intent_attack" and agent_idx >= 2 and i == 1:
                        pass
                    else:
                        canvas = draw_bev_dot(
                            [
                                x,
                                y,
                                w,
                                l,
                                heading + np.pi / 2,
                            ],  # plotting function has bad weird coordinate system
                            canvas,
                            color=color,  # Multiply by 255 to convert to 0-255 range
                            fill=is_ego,
                            l_shift=-1.461 if is_ego else 0.0,
                        )
                        if i == 0:
                            gt_canvas = canvas
                        else:
                            pred_canvas = canvas
                        overlay = draw_bev_bboxes(
                            [
                                x,
                                y,
                                w,
                                l,
                                heading + np.pi / 2,
                            ],  # plotting function has bad weird coordinate system
                            overlay,
                            color=vehicle_color,# black  
                            fill=is_ego,
                            l_shift=-1.461 if is_ego else 0.0,
                        )

                    if i == 0 and task_mask[batch_idx, agent_idx, t_].any():
                        condition_canvas = draw_bev_dot(
                            [
                                x,
                                y,
                                w,
                                l,
                                heading + np.pi / 2,
                            ],  # plotting function has bad weird coordinate system
                            condition_canvas,
                            color=color,  # Multiply by 255 to convert to 0-255 range
                            fill=is_ego,
                            l_shift=-1.461 if is_ego else 0.0,
                        )
                        condition_overlay = draw_bev_bboxes(
                            [
                                x,
                                y,
                                w,
                                l,
                                heading + np.pi / 2,
                            ],  # plotting function has bad weird coordinate system
                            condition_overlay,
                            color=vehicle_color,  # Multiply by 255 to convert to 0-255 range
                            fill=is_ego,
                            l_shift=-1.461 if is_ego else 0.0,
                        )                            
                    # Concatenate the three canvases for logging

            # Prepare images for logging
            for canvas_index, cur_overlay in enumerate([pred_overlay, gt_overlay, condition_overlay]):
                images[canvas_index].append(cur_overlay)
        image = [pred_canvas, gt_canvas, condition_canvas]
        image = [canvas.convert("RGB").convert("RGBA") for canvas in image]
        for canvas_index,overlay in enumerate(images):
            for frame_index in range(len(overlay)):
                overlay[frame_index] = fuse_images(image[canvas_index].copy(), overlay[frame_index])

        fps = [N*2] * len(images)
        save_paths = [f'scripts/pipeline/{task}_{name}.gif' for name in ['pred', 'gt', 'condition']]
        generate_gif(images, fps, save_paths)
        return 

# Generate and save GIFs from lists of image frames
def generate_gif(images, fps, save_paths):
    """
    images: List of List of PIL.Image, e.g. [ [frame1, frame2, ...], [frame1, ...], [frame1, ...] ]
    fps: List of int, e.g. [fps1, fps2, fps3]
    save_paths: List of str, e.g. ['./output_0.gif', './output_1.gif', './output_2.gif']
    """
    for _, (frames, f, path) in enumerate(zip(images, fps, save_paths)):
        if not frames:
            continue
        # duration is per frame in ms
        duration = 1000 / f
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
    return

# Create a task mask for scene generation, randomly selecting agents to be conditioned
def create_sg_mask(
    scene_tensor_features: SceneTensor, n_chosen_prob: int = 0.3
) -> torch.Tensor:
    n_valid_agents_per_sample = scene_tensor_features.validity.any(dim=-1).sum(
        dim=-1
    )
    task_mask = torch.zeros_like(scene_tensor_features.tensor)
    for batch_idx in range(scene_tensor_features.tensor.shape[0]):
        n = n_valid_agents_per_sample[batch_idx]
        mask = torch.rand(n, device=n.device) < n_chosen_prob
        task_mask[batch_idx, :n] = mask.unsqueeze(-1).unsqueeze(-1)
    return task_mask

# Run model inference for a given task and return predictions
def infer_model(
    model, scene_tensor: SceneTensor, noise, task: str
) -> TargetsType:
    """
    Make an inference of the input batch features given a model.

    :param pl_module: lightning model
    :param features: model inputs
    :return: model predictions
    """
    features = {
        "scene_tensor": scene_tensor,
    }
    with torch.no_grad(), autocast(enabled=False):
        if task == "bp":
            bp_mask = torch.zeros_like(scene_tensor.tensor, device=scene_tensor.tensor.device)
            bp_mask[:, :, :5, :] = 1
            features["task_mask"] = bp_mask
        elif task == "scene_gen":
            features["task_mask"] = create_sg_mask(scene_tensor)
        predictions = model.forward_inference(features, noise)
    return predictions

# Run inference and visualization for each scenario
def infer_and_vis(
    model,
    scene_tensor: SceneTensor
) -> None:
    """
    Visualizes and logs all examples from the input dataloader.

    :param pl_module: lightning module used for inference
    :param dataloader: torch dataloader
    :param loggers: list of loggers from the trainer
    :param training_step: global step in training
    :param prefix: prefix to add to the log tag
    """
    features = copy.deepcopy(scene_tensor)
    noise = torch.randn(
            scene_tensor.tensor.shape,
            device=scene_tensor.tensor.device,
        )

    for task in ["bp"]: # "bp" for behavior prediction, "scene_gen" for scene generation
        features = copy.deepcopy(scene_tensor)
        predictions = infer_model(
            model,
            scene_tensor,
            noise,
            task,
        )

        visualize_scene_tensor(features, predictions, task)
    return 
        
if __name__ == '__main__':
    # Load configuration file
    with open('scripts/pipeline/config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)
    # Optionally update config for training
    cfg = update_config(cfg)
    worker = build_worker(cfg)
    scenario_builder = build_scenario_builder(cfg)
    scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
    del worker, scenario_builder
    model = build_torch_module_wrapper(cfg.model)
    if cfg.checkpoint:
        my_ckpt = torch.load(cfg.checkpoint, map_location="cpu")
        my_ckpt["state_dict"] = {k[6:]: v for k, v in my_ckpt["state_dict"].items()}
        model.load_state_dict(my_ckpt["state_dict"], strict=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    feature_builder = model.get_list_of_required_feature()[0]

    # Process one scenario at a time for clarity
    for scenario in scenarios[:1]:
        scene_tensor = get_features_from_scenario(feature_builder, scenario)
        scene_tensor = to_tensor(scene_tensor, device)
        scene_tensor = unsqueeze(scene_tensor)
        infer_and_vis(model, scene_tensor)

    print('done')
        
