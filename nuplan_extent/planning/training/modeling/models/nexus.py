import logging
import torch
import numpy as np

from torch import nn
from typing import List, Dict

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import (
    AbstractTargetBuilder,
)

from nuplan_extent.planning.training.preprocessing.feature_builders.nexus_feature_builder import (
    SceneTensor,
    N_SCENE_TENSOR_FEATURES,
)
import wandb
import torch.distributions as tdist
import random
import time
from ...preprocessing.features.scene_tensor import decode_scene_tensor

np.set_printoptions(precision=2, suppress=True)

logger = logging.getLogger(__name__)


def is_wandb_active():
    return wandb.run is not None


class Nexus(TorchModuleWrapper):
    def __init__(
        self,
        diffuser: nn.Module,
        global_encoder: nn.Module,
        render: nn.Module,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        future_trajectory_sampling: TrajectorySampling,
        postprocessor: nn.Module = None,
        pretraining_path: str = None,
        num_paralell_scenario: int = 1,
        num_conditioned_frames: int = 4,
        downstream_task: str = "scenario_extrapolation",
        temporal_schedule_prob: float = 0.5,
        behavior_pred_task_prob: float = 0.5,
        control_agent_prob: float = 0.9,
        control_time_prob: float = 0.9,
        control_feature_prob: List[float] = [0.9] * N_SCENE_TENSOR_FEATURES,
        forcing: bool = True,
    ):
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self.diffuser = diffuser
        self.global_encoder = global_encoder
        self.render = render
        self.postprocessor = postprocessor
        self._num_paralell_scenario = num_paralell_scenario
        self._num_conditioned_frames = num_conditioned_frames
        self._downstream_task = downstream_task
        self._temporal_schedule_prob = temporal_schedule_prob
        self._behavior_pred_task_prob = behavior_pred_task_prob
        self._control_agent_prob = control_agent_prob
        self._control_time_prob = control_time_prob
        self._control_feature_prob = torch.tensor(control_feature_prob)

        self.pretraining_path = pretraining_path
        self.load_pretrained_weights()
        self.forcing = True
        # self.schedule_prob = [0.2, 0.4, 0.6, 0.8]
        # self.schedule_prob = [0.5, 1.0, 1.0, 1.0]
        self.schedule_prob = [0., 0., 0., 1.0]

    def _build_model(self):
        pass

    def set_vis_features(self, is_vis_features: bool, vis_features_path: str):
        """
        Set params for saving features, for visualization, only when simulation feature video callback is on.
        :param is_vis_features: whether to save features
        :param vis_features_path: path to save features
        """
        self._is_vis_features = is_vis_features
        self._vis_features_path = vis_features_path

    def load_pretrained_weights(self):
        logger.info(f"loading pretrained weights: {self.pretraining_path}")

        def match_incompatible_keys(checkpoint_state_dict, src, dst):
            filtered_state_dict = {}
            for name, param in checkpoint_state_dict.items():
                if name.startswith(src):
                    name = name[len(src) :]
                filtered_state_dict[name] = param
            return filtered_state_dict

        def filter_incompatible_keys(
            model_state_dict, checkpoint_state_dict, name_prefix=""
        ):
            filtered_state_dict = {}
            for name, param in checkpoint_state_dict.items():
                name = name_prefix + name
                if (
                    name in model_state_dict
                    and param.shape == model_state_dict[name].shape
                ):
                    filtered_state_dict[name] = param
                else:
                    logger.info(f"Skipping incompatible key: {name}")
            return filtered_state_dict

        if self.pretraining_path is not None:
            checkpoint = torch.load(self.pretraining_path, map_location="cpu")[
                "state_dict"
            ]
            checkpoint = match_incompatible_keys(checkpoint, src="model.", dst="")

            checkpoint_not_found_num = 0
            model_not_found_num = 0
            for k, v in checkpoint.items():
                if k not in self.state_dict() or self.state_dict()[k].shape != v.shape:
                    logger.info(f"{k} not in model")
                    print(f"{k} not in model")
                    checkpoint_not_found_num += 1
            for k, v in self.state_dict().items():
                if k not in checkpoint or checkpoint[k].shape != v.shape:
                    logger.info(f"{k} not in checkpoint")
                    model_not_found_num += 1
            self.load_state_dict(
                filter_incompatible_keys(self.state_dict(), checkpoint), strict=False
            )
            logger.info(
                f"checkpoint not found num: {checkpoint_not_found_num}/{len(checkpoint)}, model not found num: {model_not_found_num}/{len(self.state_dict())}"
            )
        # lora.mark_only_lora_as_trainable(self, bias='lora_only')

    def forward(self, input_features: FeaturesType, scenario=None) -> TargetsType:
        """
        Predict
        :param input_features: input features containing
        :return: targets: predictions from network
        """
        if self.training:
            return self.forward_train(input_features)
        else:
            with torch.no_grad():
                # import pdb; pdb.set_trace()
                if self._downstream_task is not None and 'sim_agents' in self._downstream_task :
                    # Waymo Sim Agents rollout -> 32 parallel worlds, 8s into future
                    return self.sim_agents_batched_rollout(input_features)
                if "task_mask" in input_features:
                    return self.forward_inference(input_features)
                else:
                    return self.forward_validation(input_features)

    def sim_agents_batched_rollout(self, input_features):
        # import pdb; pdb.set_trace()
        n_rollouts = 32
        # Extract the last dimension to agents_id
        agents_id = input_features['scene_tensor'].tensor[:,:,0, -1]
        # Remove the last dimension from input_features['scene_tensor'].tensor
        input_features['scene_tensor'].tensor = input_features['scene_tensor'].tensor[..., :-1]
        # repeat and batch essential features: [B, ...] => [32*B, ...]
        tmp_feature_dict = input_features.copy()
        original_bs = tmp_feature_dict["scene_tensor"].tensor.shape[0]
        # repeat the scene_tensor: [0-31: batch 1], [32-63: batch 2]....
        for key, value in vars(tmp_feature_dict['scene_tensor']).items():
            if isinstance(value, torch.Tensor):
                tmp_feature_dict['scene_tensor'].__setattr__(key, value.repeat_interleave(n_rollouts, dim=0))

        # inpainting here
        scene_tensor_features: SceneTensor = tmp_feature_dict["scene_tensor"]
        global_context = self.global_encoder(scene_tensor_features)
        # import pdb;pdb.set_trace()
        if 'test' in self._downstream_task:
            # all valid tokens in scene_tensor should be kept in test time
            # keep_mask = scene_tensor_features.validity.unsqueeze(-1).expand(-1, -1, -1, scene_tensor_features.tensor.shape[-1])
            valid_mask = scene_tensor_features.validity
            valid_mask,keep_mask = valid_mask.chunk(2,dim=-1)
            keep_mask = keep_mask.unsqueeze(-1).repeat(1,1,1,scene_tensor_features.tensor.shape[-1])
            # diffusion model only generates tokens where valid_mask is 1, 
            # in waymo, agents present at current timestep(5th) should be predicted in future timesteps(8s@2Hz)
            # valid_mask = scene_tensor_features.validity.clone()
            valid_mask[:, :, 5:] = (scene_tensor_features.validity[:, :, 4:5] == 1).expand(-1, -1, 16)

        elif 'val' in self._downstream_task:
            n_past_timesteps = 5
            if True:
                # behavior prediction task
                keep_mask = torch.zeros_like(scene_tensor_features.tensor)
                keep_mask[:, :, :n_past_timesteps] = 1
                valid_mask = scene_tensor_features.validity.clone()
            else:
                # intent prediction task
                valid_mask = scene_tensor_features.validity
                valid_mask,keep_mask = valid_mask.chunk(2,dim=-1)
                # keep_mask[...,-1] = 0
                keep_mask = keep_mask.unsqueeze(-1).repeat(1,1,1,scene_tensor_features.tensor.shape[-1])
        else:
            raise NotImplementedError(f'Unknown downstream task: {self._downstream_task}')

        self.diffuser.model.eval()
        sampled_tensor, _ = self.diffuser.sample(
            scene_tensor=scene_tensor_features.tensor,
            valid_mask=valid_mask,
            keep_mask=keep_mask,
            global_context=global_context,
            raw_map=self.extract_raw_map(scene_tensor_features)
        )  # batch, n_agents, n_timesteps, n_features
        # n_features (x, y, cos(yaw), sin(yaw), l, w)
        self.diffuser.model.train()
        sampled_tensor = decode_scene_tensor(sampled_tensor)
        # reshape into [[32 rollout for b1], [32 rollout for b2], ...] format. 
        sim_agent_rollouts = [[] for _ in range(original_bs)]
        for batch_index, tok_seq in enumerate(sampled_tensor):
            batch_number = batch_index // n_rollouts
            sim_agent_rollouts[batch_number].append(tok_seq)
        out = {"scene_tensor": input_features['scene_tensor'],
                "sim_agents_rollouts": sim_agent_rollouts,
                "agents_map": agents_id}
        return out

    def generate_task_mask(
        self, scene_tensor_features: SceneTensor, n_past_timesteps: int
    ) -> torch.Tensor:
        n_valid_agents_per_sample = scene_tensor_features.validity.any(dim=-1).sum(
            dim=-1
        )
        # sample a random number of agents to condition on, one for each sample
        task_mask_sg = torch.zeros_like(scene_tensor_features.tensor)
        for batch_idx in range(scene_tensor_features.tensor.shape[0]):
            n = n_valid_agents_per_sample[batch_idx]
            n_chosen = torch.randint(0, n, (1,), device=n.device)
            mask = torch.rand(n, device=n.device) < n_chosen / n  # NA

            task_mask_sg[batch_idx, :n] = mask.unsqueeze(-1).unsqueeze(-1)
        # always condition on ego
        task_mask_sg[:, 0] = 1.0

        task_mask_bp = torch.zeros_like(scene_tensor_features.tensor)
        task_mask_bp[:, :, :n_past_timesteps] = 1

        task_type = torch.rand(scene_tensor_features.tensor.shape[0])  # B
        task_mask = torch.zeros_like(scene_tensor_features.tensor)
        behavior_task_idx = task_type < self._behavior_pred_task_prob
        task_mask[behavior_task_idx] = task_mask_bp[behavior_task_idx]
        task_mask[~behavior_task_idx] = task_mask_sg[~behavior_task_idx]

        return task_mask

    def _generate_gaussian_diffusion_times(self, scene_tensor_features: SceneTensor):

        B, NA, NT = scene_tensor_features.tensor.shape[:3]
        device = scene_tensor_features.tensor.device
        valid_mask = scene_tensor_features.validity.bool()
        diffusion_times_gaussian = torch.rand(B, 1, 1, device=device,).repeat(1, NA, NT) # this each batch sample get a random diffusion time

        def generate_gaussian(N_min=2, N_max=3, mu_min=0, mu_max=21, std_min=0.1, std_max=2.0, strength=0.2, mesh=None, device=device):
            N = np.random.choice(np.arange(N_min, N_max+1))
            mus_x = torch.rand(N, device=device) * (mu_max - mu_min) + mu_min
            stds_x = torch.rand(N, device=device) * (std_max - std_min) + std_min
            random_values = torch.randint(0, 2, (N,), device=device) * 2 - 1
            mus_x = mus_x.view(N, 1, 1)
            stds_x = stds_x.view(N, 1, 1)
            random_values = random_values.view(N, 1, 1)
            dist_x = tdist.Normal(mus_x, stds_x)
            probs_x = strength * torch.exp(dist_x.log_prob(mesh.unsqueeze(0).repeat(N, 1, 1))) * random_values
            return probs_x 
            
        x_values = torch.arange(0, NA, device=device)
        y_values = torch.arange(0, NT, device=device)
        x_mesh, y_mesh = torch.meshgrid(x_values, y_values)

        probs_list = []
        for _ in range(B):
            probs_x = generate_gaussian(N_min=5, N_max=10, mesh=x_mesh, device=device)
            probs_y = generate_gaussian(mesh=y_mesh, device=device)
            probs_sum = torch.sum(probs_x, dim=0) + torch.sum(probs_y, dim=0)
            probs_list.append(probs_sum)

        probs_tensor = torch.stack(probs_list, dim=0)
        diffusion_times_gaussian = torch.clamp((diffusion_times_gaussian + probs_tensor), min=0, max=1)

        return diffusion_times_gaussian

    def _generate_diffusion_forcing_times(self, scene_tensor_features: SceneTensor):
        B, NA, NT = scene_tensor_features.tensor.shape[:3]
        device = scene_tensor_features.tensor.device
        valid_mask = scene_tensor_features.validity.bool()

        diffusion_times_uniform = torch.rand(B, 1, 1, device=device,).repeat(1, NA, NT)  # this each batch sample get a random diffusion time
        diffusion_times_temporal = torch.arange(1, NT+1, step=1, device=device,).unsqueeze(0).unsqueeze(0).repeat(B, NA, 1)
        # take max
        n_history_steps = 5
        n_future_steps = 16
        diffusion_times_temporal = torch.max(
            torch.tensor(0),
            (diffusion_times_temporal - n_history_steps) / n_future_steps,
        )
        diffusion_times_agent = torch.rand(B, NA, 1, device=device,).repeat(1, 1, NT) # each agent get a random diffusion time
        diffusion_times_random = torch.rand(B, NA, NT, device=device,) # each token get a random diffusion time
        diffusion_times_gaussian = self._generate_gaussian_diffusion_times(scene_tensor_features)

        diffusion_times = torch.zeros(B, NA, NT, device=device,)

        rand_index = torch.rand(B)
        time_idx = (rand_index < self.schedule_prob[0])
        diffusion_times[time_idx] = diffusion_times_uniform[time_idx]

        time_idx = (rand_index >= self.schedule_prob[0]) & (rand_index < self.schedule_prob[1])
        diffusion_times[time_idx] = diffusion_times_temporal[time_idx]

        time_idx = (rand_index >= self.schedule_prob[1]) & (rand_index < self.schedule_prob[2])
        diffusion_times[time_idx] = diffusion_times_agent[time_idx]

        time_idx = (rand_index >= self.schedule_prob[2]) & (rand_index < self.schedule_prob[3])
        diffusion_times[time_idx] = diffusion_times_random[time_idx]

        time_idx = (rand_index >= self.schedule_prob[3])
        diffusion_times[time_idx] = diffusion_times_gaussian[time_idx]

        diffusion_times = torch.where(~valid_mask, torch.full_like(diffusion_times, 1), diffusion_times)

        return diffusion_times

    def _generate_diffusion_times(
        self, scene_tensor_features: SceneTensor
    ) -> torch.Tensor:
        diffusion_times_uniform = torch.rand(
            scene_tensor_features.tensor.shape[0],
            1,
            device=scene_tensor_features.tensor.device,
        ).repeat(
            1, scene_tensor_features.tensor.shape[2]
        )  # this each sample get a random diffusion time B x A x T

        diffusion_times_temporal = (
            torch.arange(
                0,
                scene_tensor_features.tensor.shape[2],
                step=1,
                device=scene_tensor_features.tensor.device,
            )
            .unsqueeze(0)
            .repeat(scene_tensor_features.tensor.shape[0], 1)
        )  # B, T t = 0 ... 1 with the dimension
        # take max of
        n_history_steps = 5
        n_future_steps = 16
        diffusion_times_temporal = torch.max(
            torch.tensor(0),
            (diffusion_times_temporal - n_history_steps) / n_future_steps,
        )

        diffusion_times = torch.zeros(
            scene_tensor_features.tensor.shape[0],
            scene_tensor_features.tensor.shape[2],
            device=scene_tensor_features.tensor.device,
        )

        temporal_time_idx = (
            torch.rand(scene_tensor_features.tensor.shape[0])
            < self._temporal_schedule_prob
        )
        diffusion_times[temporal_time_idx] = diffusion_times_temporal[temporal_time_idx]
        diffusion_times[~temporal_time_idx] = diffusion_times_uniform[
            ~temporal_time_idx
        ]

        return diffusion_times

    def generate_diffusion_times(self, scene_tensor_features: SceneTensor):
        if self.forcing:
            return self._generate_diffusion_forcing_times(scene_tensor_features)
        else:
            return self._generate_diffusion_times(scene_tensor_features)

    def generate_control_mask(self, scene_tensor_features: SceneTensor) -> torch.Tensor:
        # TODO implement control mask
        # 1 is valid
        control_agent = (
            torch.rand(
                scene_tensor_features.tensor.shape[0],
                scene_tensor_features.tensor.shape[1],
                device=scene_tensor_features.tensor.device,
            )
            < self._control_agent_prob
        )
        control_time = (
            torch.rand(
                scene_tensor_features.tensor.shape[0],
                scene_tensor_features.tensor.shape[1],
                scene_tensor_features.tensor.shape[2],
                device=scene_tensor_features.tensor.device,
            )
            < self._control_time_prob
        )
        control_feature = (
            torch.rand(
                scene_tensor_features.tensor.shape[0],
                scene_tensor_features.tensor.shape[1],
                scene_tensor_features.tensor.shape[2],
                scene_tensor_features.tensor.shape[3],
                device=scene_tensor_features.tensor.device,
            )
            < self._control_feature_prob
        )

        control_mask = torch.ones_like(scene_tensor_features.tensor)
        # control_mask[control_agent] = 1
        control_mask[control_time] = 0
        # control_mask[control_feature] = 1
        return control_mask

    def generate_goal_mask(self, scene_tensor_features: SceneTensor, diffusion_times: torch.Tensor) -> torch.Tensor:
        self.goal_threshold = 0.15
        self.zero_prob = 0.5
        goal_mask = diffusion_times < self.goal_threshold
        zero_mask = torch.rand(diffusion_times.shape, device=diffusion_times.device) < self.zero_prob

        diffusion_times = torch.where(torch.logical_and(goal_mask, zero_mask), torch.zeros_like(diffusion_times, device=diffusion_times.device), diffusion_times)
        
        control_mask = torch.zeros_like(scene_tensor_features.tensor)
        control_mask[diffusion_times==0] = 1

        return control_mask, diffusion_times

    def forward_train(self, input_features: FeaturesType) -> Dict:
        """
        Forward pass for training
        :param input_features: input features
        :param targets: targets
        """
        # import pdb; pdb.set_trace()
        # Encode image
        input_features['scene_tensor'].tensor = input_features['scene_tensor'].tensor[...,:8] # to exclude agents id from scene_tensor
        scene_tensor_features: SceneTensor = input_features["scene_tensor"] # [B A T C]
        global_context = self.global_encoder(scene_tensor_features) # [B L C]

        n_past_timesteps = 4 + 1  # 1 current timestep + 4 past timesteps
        present_mask = scene_tensor_features.validity[:, :, :n_past_timesteps].any(
            dim=-1
        )
        for batch_idx in range(scene_tensor_features.tensor.shape[0]):
            scene_tensor_features.validity[batch_idx, ~present_mask[batch_idx], :] = 0

        # decice on the schedule
        diffusion_times = self.generate_diffusion_times(scene_tensor_features) # [B T]

        # decide on task, here is conditioned scene generation
        task_mask = self.generate_task_mask(scene_tensor_features, n_past_timesteps)

        # control_mask = self.generate_control_mask(scene_tensor_features)
        control_mask, diffusion_times = self.generate_goal_mask(scene_tensor_features, diffusion_times)

        loss, outs = self.diffuser.compute_loss(  # type: ignore
            scene_tensor=scene_tensor_features.tensor,
            valid_mask=scene_tensor_features.validity,
            global_context=global_context,
            diffusion_times=diffusion_times,
            task_mask=task_mask,
            control_mask=control_mask,
            raw_map=self.extract_raw_map(scene_tensor_features)
        )

        return dict(
            diffusion_out={
                "loss": loss,
                "scene_tensor": scene_tensor_features.tensor,
                "valid_mask": scene_tensor_features.validity,
                "task_mask": task_mask,
                **{k: v.detach() for k, v in outs.items()},
            }
        )

    def forward_validation(self, input_features: FeaturesType) -> Dict:
        # Encode image
        # global_context = self.image_encoder(input_features)  # B, L, C
        input_features['scene_tensor'].tensor = input_features['scene_tensor'].tensor[...,:8] # to exclude agents id from scene_tensor
        scene_tensor_features: SceneTensor = input_features["scene_tensor"]
        global_context = self.global_encoder(scene_tensor_features)

        n_past_timesteps = 4 + 1  # 1 current timestep + 4 past timesteps
        present_mask = scene_tensor_features.validity[:, :, :n_past_timesteps].any(
            dim=-1
        )

        for batch_idx in range(scene_tensor_features.tensor.shape[0]):
            scene_tensor_features.validity[batch_idx, ~present_mask[batch_idx], :] = 0

        task_mask_bp = torch.zeros_like(scene_tensor_features.tensor)
        task_mask_bp[:, :, :n_past_timesteps] = 1

        time_consume = 0
        time_start = time.time()
        xpred, _ = self.diffuser.sample(
            scene_tensor=scene_tensor_features.tensor,
            valid_mask=scene_tensor_features.validity,
            global_context=global_context,
            z_t=None,
            keep_mask=task_mask_bp,
            raw_map=self.extract_raw_map(scene_tensor_features)
            # use_guidance_fn=True,
        )
        time_end = time.time()
        time_consume += time_end - time_start
        # ensure that the last valid timestep is also conditioned on
        valid_mask = scene_tensor_features.validity
        intent_conditioned_task_mask_bp = torch.zeros_like(scene_tensor_features.tensor)
        intent_conditioned_task_mask_bp[:, :, :n_past_timesteps] = 1
        last_valid_indices = torch.max(
            (valid_mask == 1).float()
            * torch.arange(valid_mask.shape[-1], device=valid_mask.device).float(),
            dim=-1,
        )[1]
        batch_indices = torch.arange(valid_mask.shape[0], device=valid_mask.device)[
            :, None
        ].expand(-1, valid_mask.shape[1])
        agent_indices = torch.arange(valid_mask.shape[1], device=valid_mask.device)[
            None, :
        ].expand(valid_mask.shape[0], -1)
        intent_conditioned_task_mask_bp[
            batch_indices, agent_indices, last_valid_indices
        ] = 1.0
        time_start = time.time()
        intent_conditioned_xpred, _ = self.diffuser.sample(
            scene_tensor=scene_tensor_features.tensor,
            valid_mask=scene_tensor_features.validity,
            global_context=global_context,
            z_t=None,
            keep_mask=intent_conditioned_task_mask_bp,
            raw_map=self.extract_raw_map(scene_tensor_features)
            # use_guidance_fn=True,
        )
        time_end = time.time()
        time_consume += time_end - time_start
        time_consume /= 2
        # print(f"Time consume: {time_consume}")
        return dict(
            behavior_prediction_out={
                "behavior_pred": xpred,
                "intent_conditioned_behavior_pred": intent_conditioned_xpred,
                "scene_tensor_gt": scene_tensor_features.tensor,
                "valid_mask": scene_tensor_features.validity,
                "task_mask": task_mask_bp,
                "time_consume": time_consume,
                "road_graph": scene_tensor_features.road_graph,
                "road_graph_validity": scene_tensor_features.road_graph_validity,
            }
        )

    def extract_raw_map(self, scene_tensor_features: SceneTensor) -> torch.Tensor:
        # extract and pad map info
        road_graph,road_graph_validity=scene_tensor_features.road_graph,scene_tensor_features.road_graph_validity
        road_graph_validity=road_graph_validity.any(dim=-1)
        B,N,P=road_graph_validity.shape
        max_point_num=road_graph_validity.sum(-1).sum(-1).max() 
        padding=torch.ones((1,road_graph.shape[-1]),device=road_graph.device)*50
        raw_map=[]
        for i in range(road_graph.shape[0]):
            graph,valid=road_graph[i],road_graph_validity[i]
            temp=graph[valid]
            temp=torch.cat([temp,padding.expand(max_point_num-temp.shape[0],-1)],dim=0)
            raw_map.append(temp)
        raw_map=torch.stack(raw_map)
        return raw_map

    def forward_inference(
        self,
        input_features: FeaturesType,
        noise: torch.Tensor = None,
        return_intermidates: bool = True,
    ) -> Dict:
        """
        Forward pass for inference
        :param input_features: input features
        :return: predictions from network
        """
        # Encode image
        # global_context = self.image_encoder(input_features)
        input_features['scene_tensor'].tensor = input_features['scene_tensor'].tensor[...,:8] # to exclude agents id from scene_tensor
        scene_tensor_features: SceneTensor = input_features["scene_tensor"]
        global_context = self.global_encoder(scene_tensor_features)

        if "task_mask" in input_features:
            task_mask = input_features["task_mask"]
        else:
            return {}

        self.diffuser.model.eval()
        sampled_tensor, intermidates = self.diffuser.sample(
            scene_tensor=scene_tensor_features.tensor,
            valid_mask=scene_tensor_features.validity,
            keep_mask=task_mask,
            global_context=global_context,
            z_t=noise,
            return_intermidates=return_intermidates,
            raw_map=self.extract_raw_map(scene_tensor_features)
        )  # batch, n_agents, n_timesteps, n_features
        # n_features (x, y, cos(yaw), sin(yaw), l, w)
        self.diffuser.model.train()

        out = {"sampled_tensor": sampled_tensor, "task_mask": task_mask}
        if return_intermidates:
            out["intermidates_sampled_tensors"] = intermidates

        return out