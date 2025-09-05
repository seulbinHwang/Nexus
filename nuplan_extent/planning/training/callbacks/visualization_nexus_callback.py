import random
from typing import Any, Dict, List, Optional
from matplotlib.colors import LinearSegmentedColormap
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import wandb
from PIL import Image,ImageDraw
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    TargetsType,
    move_features_type_to_device,
)
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.cuda.amp import autocast

from nuplan_extent.planning.training.callbacks.utils.visualization_utils import (
    draw_bev_bboxes,Draw_bev_dot_interplate,generate_gif,fuse_images,Draw
)
from nuplan_extent.planning.training.preprocessing.feature_builders.nexus_feature_builder import (
    SceneTensor,
    decode_scene_tensor,
    unnormalize_roadgraph,
    encode_scene_tensor
)
import copy

cmap = plt.get_cmap("tab20")



class VisualizationNexusCallback(pl.Callback):
    """
    Callback that visualizes planner model inputs/outputs and logs them in Tensorboard.
    """

    def __init__(
        self,
        num_scene_samples_to_vis: int,
        num_noise_samples_to_vis: int,
        log_train_every_n_epochs: int,
        log_train_every_n_batches: int,
        log_val_every_n_epochs: int,
        raster_type: Dict[str, List[float]],
        canvas_size: int,
        pixel_size: int,
        dataset: str = "nuplan",
    ):
        """
        Initialize the class.

        :param images_per_tile: number of images per tiles to visualize
        :param num_train_tiles: number of tiles from the training set
        :param num_val_tiles: number of tiles from the validation set
        :param pixel_size: [m] size of pixel in meters
        :param num_frames: number of frames to visualize
        """
        super().__init__()

        self.num_scene_samples_to_vis = num_scene_samples_to_vis
        self.num_noise_samples_to_vis = num_noise_samples_to_vis
        self.dataset = dataset

        self.log_train_every_n_epochs = log_train_every_n_epochs
        self.log_train_every_n_batches = log_train_every_n_batches
        self.log_val_every_n_epochs = log_val_every_n_epochs

        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.counter = {
            "train_epochs": 0,
            "val_epochs": 0,
            "train_batches": 0,
        }

        self.noise_samples_scene_gen = [
            [None for _ in range(num_noise_samples_to_vis)]
            for _ in range(num_scene_samples_to_vis)
        ]

        self.noise_samples_bp = [
            [None for _ in range(num_noise_samples_to_vis)]
            for _ in range(num_scene_samples_to_vis)
        ]

        self.raster_type = raster_type
        self.canvas_size = canvas_size
        self.pixel_size = pixel_size

    def _initialize_dataloaders(self, datamodule: pl.LightningDataModule) -> None:
        """
        Initialize the dataloaders. This makes sure that the same examples are sampled
        every time for comparison during visualization.

        :param datamodule: lightning datamodule
        """
        train_set = datamodule.train_dataloader().dataset
        # val_set = datamodule.val_dataloader().dataset
        val_set = datamodule.train_dataloader().dataset
        self.train_dataloader = self._create_dataloader(
            train_set, self.num_scene_samples_to_vis
        )
        self.val_dataloader = self._create_dataloader(
            val_set, self.num_scene_samples_to_vis
        )

    def _create_dataloader(
        self, dataset: torch.utils.data.Dataset, num_samples: int
    ) -> torch.utils.data.DataLoader:
        dataset_size = len(dataset)
        num_keep = min(dataset_size, num_samples)
        sampled_idxs = random.sample(range(dataset_size), num_keep)
        subset = torch.utils.data.Subset(dataset=dataset, indices=sampled_idxs)
        return torch.utils.data.DataLoader(
            dataset=subset,
            batch_size=1,
            collate_fn=FeatureCollate(),
        )

    def _log_from_dataloader(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        loggers: List[Any],
        training_step: int,
        prefix: str,
    ) -> None:
        """
        Visualizes and logs all examples from the input dataloader.

        :param pl_module: lightning module used for inference
        :param dataloader: torch dataloader
        :param loggers: list of loggers from the trainer
        :param training_step: global step in training
        :param prefix: prefix to add to the log tag
        """
        for batch_idx, batch in enumerate(dataloader):
            batch[0]["scene_tensor"].tensor=batch[0]["scene_tensor"].tensor[...,:8]
            for noise_idx in range(self.num_noise_samples_to_vis):
                features = copy.deepcopy(batch[0])
                if self.noise_samples_scene_gen[batch_idx][noise_idx] is None:
                    noise_scene_gen = torch.randn(
                        features["scene_tensor"].tensor.shape,
                        device=features["scene_tensor"].tensor.device,
                    )
                    self.noise_samples_scene_gen[batch_idx][noise_idx] = noise_scene_gen
                    noise_bp = torch.randn(
                        features["scene_tensor"].tensor.shape,
                        device=features["scene_tensor"].tensor.device,
                    )
                    self.noise_samples_bp[batch_idx][noise_idx] = noise_bp

                else:
                    noise_bp = self.noise_samples_bp[batch_idx][noise_idx]
                    noise_scene_gen = self.noise_samples_scene_gen[batch_idx][noise_idx]

                for task in ["bp", "intent_attack"]:
                    features = copy.deepcopy(batch[0])
                    noise = noise_bp if task in ["bp", "intent_attack"] else noise_scene_gen
                    predictions = self._infer_model(
                        pl_module,
                        move_features_type_to_device(features, pl_module.device),
                        noise.to(pl_module.device),
                        task,
                    )

                    self._log_batch(
                        loggers,
                        features,
                        predictions,
                        batch_idx,
                        training_step,
                        f"{prefix}/{task}/{batch_idx}/{noise_idx}",
                        task,
                    )

    def _coords_to_pixels(self, coords):
        return coords / self.pixel_size + self.canvas_size / 2

    @staticmethod
    def get_color_gradients(N,alpha=128):
        # alpha is the transparency value (0 is fully transparent, 255 is fully opaque)
        # Create different colormaps
        # Green to Blue (for Sim agents)
        colors_sim = [(0, (0,232,158)), (1, (2,121,255))]
        # normalized color values to [0, 1]
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
    @staticmethod
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
    
    def _log_batch(
        self,
        loggers: List[Any],
        features: FeaturesType,
        predictions: TargetsType,
        batch_idx: int,
        training_step: int,
        plot_output_name: str,
        task: str,
    ) -> None:
        
        scene_tensor_features: SceneTensor = features["scene_tensor"]
        valid_mask = scene_tensor_features.validity.cpu().numpy()
        rg = scene_tensor_features.road_graph
        rg = unnormalize_roadgraph(rg.cpu())
        rgv = scene_tensor_features.road_graph_validity.cpu()
        rg_pixels = self._coords_to_pixels(rg[..., :2])  # b, n_lanes, n_points, n_dim
        rg_pixels_valid = rgv[..., :2]  # b, n_lanes, n_points, n_dim
        sampled_tensor = predictions["sampled_tensor"]
        task_mask = predictions["task_mask"]

        # instantiate function
        draw_bev_dot = Draw.draw_bev_dot
        draw_bev_bboxes=Draw.draw_bev_bboxes
        fuse_images=Draw.fuse_images
        generate_gif=Draw.generate_gif

        original_scene_tensor = decode_scene_tensor(scene_tensor_features.tensor.cpu().numpy())
        sampled_tensor = decode_scene_tensor(sampled_tensor.cpu().numpy())
        
        # linear interpolation
        N=3  # gt1, inter=1/N*gt1+(1-1/N)*gt2, gt2
        original_scene_tensor = self.linear_interpolation(original_scene_tensor,task="scene_tensor",N=N)
        sampled_tensor = self.linear_interpolation(sampled_tensor,task="scene_tensor",N=N)
        valid_mask = self.linear_interpolation(valid_mask,task='mask',N=N)
        task_mask = self.linear_interpolation(task_mask.numpy(),task='mask',N=N) # dim as scene_tensor
        
        n_timestamps = original_scene_tensor.shape[2]

        av_gradient, env_gradient, synthetic_gradient = self.get_color_gradients(n_timestamps)
        vehicle_color = (50, 50, 50, 255)  # Black
        line_color = (122, 120, 120,255)  # Grey ,no opacity added here, default 255

        for batch_idx in range(sampled_tensor.shape[0]):
            # plot the road graph
            pred_canvas=Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
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

            images = [[], [], []]
            for t_ in range(n_timestamps):
                pred_overlay = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
                gt_overlay = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
                condition_overlay = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))
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

                    for i, (st, canvas, overlay) in enumerate(
                        [
                            (original_scene_tensor, gt_canvas, gt_overlay),
                            (sampled_tensor, pred_canvas, pred_overlay),
                        ]
                    ):
                        padding = st.shape[-1] - 6
                        x, y = self._coords_to_pixels(st[batch_idx, agent_idx, t_, :2])
                        l, w = st[batch_idx, agent_idx, t_, [4 + padding, 5 + padding]] / self.pixel_size

                        cosh = st[batch_idx, agent_idx, t_, 2]
                        sinh = st[batch_idx, agent_idx, t_, 3]
                        heading = np.arctan2(sinh, cosh)
                        is_ego = agent_idx == 0
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
            videos = generate_gif(images,fps)
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image('image_' + plot_output_name, image, caption=["Pred", "GT", "Condition"],mode=["RGBA"]*3)
                    logger.log_video('video_' + plot_output_name, videos, fps=fps, caption=["Pred", "GT", "Condition"])

    def _create_intent_attack_mask(
        self, scene_tensor_features: SceneTensor, n_chosen_prob: int = 0.3
    ) -> torch.Tensor:

        def get_goal_point(example_traj, rg):
            example_traj = decode_scene_tensor(example_traj)
            rg = unnormalize_roadgraph(rg)[:,::2,:2]
            valid_map_lines = abs(rg).sum(dim=(1, 2)) > 0
            rg = rg[valid_map_lines].double()
            ego_traj, attack_traj = example_traj[0], example_traj[1]

            trajectory_tensor = attack_traj
            positions_tensor = rg #[num_traj, num_points, 2]

            fifth_point = trajectory_tensor[4, :2]
            yaw_cos = trajectory_tensor[4, 2]
            yaw_sin = trajectory_tensor[4, 3]

            velocity = ((trajectory_tensor[4,0]-trajectory_tensor[0,0])**2 + (trajectory_tensor[4,1]-trajectory_tensor[0,1])**2)**0.5/2
            inner_radius = torch.abs(velocity) * 7
            outer_radius = torch.abs(velocity) * 9
            angle = np.deg2rad(30)
            start_angle = torch.atan2(yaw_sin, yaw_cos) - angle / 2
            end_angle = start_angle + angle

            dx = positions_tensor[:, :, 0] - fifth_point[0]
            dy = positions_tensor[:, :, 1] - fifth_point[1]
            distances = torch.sqrt(dx**2 + dy**2)
            angles = torch.atan2(dy, dx)

            inside_sector = (angles >= start_angle) & (angles <= end_angle) & (distances >= inner_radius) & (distances <= outer_radius)
            traj_idx = inside_sector.any(dim=1)
            random_true_index = traj_idx[torch.randint(0, len(traj_idx), (1,))]

            true_indices = torch.nonzero(inside_sector)
            if true_indices.size(0) == 0:
                valid = 0
                return encode_scene_tensor(attack_traj[-1]), valid
            random_index = true_indices[torch.randint(0, true_indices.size(0), (1,))].squeeze()

            goal_point = positions_tensor[random_index[0], min(random_index[1]-1, 0)]
            next_point = positions_tensor[random_index[0], min(random_index[1]-1, 0)+1]
            goal_angle = torch.atan2(next_point[1]-goal_point[1], next_point[0]-goal_point[0])
            angle_cos, angle_sin = torch.cos(goal_angle), torch.sin(goal_angle)
            vx, vy = velocity*angle_cos, velocity*angle_sin
            dim = attack_traj[0,-2:]

            # ego_last_point
            ego_last_point = ego_traj[-1]
            dx = ego_last_point[0] - fifth_point[0]
            dy = ego_last_point[1] - fifth_point[1]
            distances = torch.sqrt(dx**2 + dy**2)
            angles = torch.atan2(dy, dx)
            is_in_sector = (angles >= start_angle) & (angles <= end_angle) & (distances >= inner_radius) & (distances <= outer_radius)
            
            if is_in_sector and torch.rand(1) < 0.5:
                
                goal_point = ego_last_point[:2]
                angle_cos, angle_sin = ego_last_point[2], ego_last_point[3]
                vx, vy = velocity*angle_cos, velocity*angle_sin
                valid = 2
            else:
                valid = 1
                pass
            
            goal_tensor = encode_scene_tensor(torch.tensor([goal_point[0], goal_point[1], angle_cos, angle_sin, vx, vy, dim[0], dim[1]]).cuda())
            return goal_tensor, valid

        task_mask = torch.zeros_like(scene_tensor_features.tensor)
        valid_mask =  scene_tensor_features.validity.bool()
        for batch_idx in range(scene_tensor_features.tensor.shape[0]):
            ego_pos = scene_tensor_features.tensor[batch_idx,0,0,:2]
            distances = torch.norm(scene_tensor_features.tensor[batch_idx,1:,0,:2] - ego_pos, dim=1)
            distances = torch.where(valid_mask[batch_idx,1:,0], distances, torch.tensor(1e6))
            values, indices = torch.topk(distances, k=4, largest=False)
            count = min(random.randint(0, len(indices)-1), max(len(indices)-1, 0))
            closest_index = indices[count] + 1
            # change to No. 1 index
            temp = scene_tensor_features.tensor[batch_idx,1].clone()
            scene_tensor_features.tensor[batch_idx,1] = scene_tensor_features.tensor[batch_idx,closest_index].clone()
            scene_tensor_features.tensor[batch_idx, closest_index] = temp
            # apply intent attack
            # goal_point = scene_tensor_features.tensor[batch_idx,0,-1,:4].clone()
            goal_point, valid = get_goal_point(scene_tensor_features.tensor[batch_idx,:2,...].clone(), scene_tensor_features.road_graph[batch_idx].clone())
            scene_tensor_features.tensor[batch_idx,1,-1,:] = goal_point
            scene_tensor_features.validity[batch_idx,1,:] = 1

        task_mask[:, :, :5, :] = 1 # set the first 5 timesteps to be conditioned on
        task_mask[:, :2, -1] = 1 # set the goal point of the ego car and attack car

        return task_mask, valid
    
    def _create_task_mask(
        self, scene_tensor_features: SceneTensor, n_chosen_prob: int = 0.3
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

    def _infer_model(
        self, pl_module: pl.LightningModule, features: FeaturesType, noise, task: str
    ) -> TargetsType:
        """
        Make an inference of the input batch features given a model.

        :param pl_module: lightning model
        :param features: model inputs
        :return: model predictions
        """
        with torch.no_grad(), autocast(enabled=False):
            pl_module.eval()
            pl_module.float()
            if task == "bp":
                bp_mask = torch.zeros_like(features["scene_tensor"].tensor)
                bp_mask[:, :, :5, :] = (
                    1  # set the first 5 timesteps to be conditioned on
                )
                features["task_mask"] = bp_mask
            # elif task == "scene_gen":
            #     features["task_mask"] = self._create_task_mask(features["scene_tensor"])
            elif task == "intent_attack":
                features["task_mask"], features["valid"] = self._create_intent_attack_mask(features["scene_tensor"])
            predictions = move_features_type_to_device(
                pl_module.model.forward_inference(features, noise), torch.device("cpu")
            )
            pl_module.train()
        # predictions = convert_features_type_to_float(predictions, torch.device('cpu'))
        return predictions

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs validation examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, "datamodule"), "Trainer missing datamodule attribute"
        assert hasattr(trainer, "global_step"), "Trainer missing global_step attribute"

        self.counter["val_epochs"] += 1
        if self.counter["val_epochs"] % self.log_val_every_n_epochs != 0:
            return

        if self.val_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(
            pl_module,
            self.val_dataloader,
            trainer.loggers,
            trainer.global_step,
            "val",
        )

    @rank_zero_only
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs validation examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, "datamodule"), "Trainer missing datamodule attribute"
        assert hasattr(trainer, "global_step"), "Trainer missing global_step attribute"

        self.counter["val_epochs"] += 1
        if self.counter["val_epochs"] % self.log_val_every_n_epochs != 0:
            return

        if self.train_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(
            pl_module,
            self.train_dataloader,
            trainer.loggers,
            trainer.global_step,
            "train",
        )