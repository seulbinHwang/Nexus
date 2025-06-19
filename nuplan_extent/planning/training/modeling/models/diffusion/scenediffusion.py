"""
This file contains an implementation of the pseudocode from the paper
"Simple Diffusion: End-to-End Diffusion for High Resolution Images"
by Emiel Hoogeboom, Tim Salimans, and Jonathan Ho.

Reference:
Hoogeboom, E., Salimans, T., & Ho, J. (2023).
Simple Diffusion: End-to-End Diffusion for High Resolution Images.
Retrieved from https://arxiv.org/abs/2301.11093
"""

import math
from typing import Optional
import os
import torch
import torch.nn as nn
from torch.special import expm1
import numpy as np
from nuplan_extent.planning.training.preprocessing.features.scene_tensor import (
    FEATURE_STD,
    FEATURE_MEANS,
    decode_scene_tensor,
    encode_scene_tensor,
    unnormalize_roadgraph
)
from nuplan_extent.planning.training.modeling.models.tokenizers.base_tokenizer_utils import (
    check_collision,
)
from collections import namedtuple
from .models.utils import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, extract, EinopsWrapper
from torch.amp import autocast
from torch.nn import functional as F
# helper
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])

def visualize_trajectories(trajectories, mask, map_tensor,num, save_path="trajectories.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    # import pdb;pdb.set_trace()
    # visualize_trajectories(scene_tensor, mask, raw_map, num, save_path)
    # scene_tensor: [b,na,nt,ndim], mask: [b,na,nt], raw_map: [b,nm,2], num: int, save_path: str
    # scene_tensor and raw_map are both normalized while input
    trajectories = trajectories[0, :num, :, :4].cpu().detach().numpy() # na,nt,ndim
    mask = mask[0, :num, :].cpu().detach().numpy().astype(bool) # na,nt
    map_tensor = map_tensor[0,:,:2].cpu().detach().numpy() # nm,nt,2
    
    trajectories = decode_scene_tensor(trajectories)
    map_mask = ~ (map_tensor == 50).all(axis=-1) # nm,nt
    map_tensor = unnormalize_roadgraph(map_tensor)
    plt.figure(figsize=(8, 6))
    plt.scatter(map_tensor[map_mask, 0], map_tensor[map_mask, 1], marker=".", alpha=0.3, color='lightgrey')
    
    for i in range(trajectories.shape[0]):  # iterate over each trajectory 
        x = trajectories[i, mask[i], 0]  # extract x
        y = trajectories[i, mask[i], 1]  # extract y
        cos_h = trajectories[i, mask[i], 2]  # extract cos(heading)
        sin_h = trajectories[i, mask[i], 3]  # extract sin(heading) 
        if mask[i].sum() == 0:
            continue
        plt.plot(x, y, marker="o", linestyle="-", label=f"Trajectory {i+1}", alpha=0.7)  # plot trajectory
        for j in range(0, x.shape[0], 5):  # plot arrows every 5 points
            plt.arrow(x[j], y[j], 0.3 * cos_h[j], 0.3 * sin_h[j], head_width=0.1, color="black")


    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Trajectory Visualization")
    plt.legend(loc="best", fontsize=8, frameon=False)
    plt.grid(True)

    plt.savefig(save_path, dpi=300)  
    plt.close()

class simpleDiffusion(nn.Module):
    def __init__(
        self,
        model,
        # image_size,
        noise_size=64,
        pred_param="v",
        schedule="shifted_cosine",
        steps=512,
        uncertainty_scale=1.0,
        condition_time=5,
        scheduling_matrix="full_sequence",
        final_step=True,
        fill_scene_tensor=True,
        constrain_mode=['clip'],
        constrain_gamma=1.0,
        loss_factor=[1.0, 0.0, 0.0, 0.0],
        distill=False,
    ): 
        super().__init__()
        self.distill = distill
        self.constraint_gamma = constrain_gamma
        self.loss_factor = loss_factor
        self.constrain_mode=constrain_mode
        # Training objective
        assert pred_param in [
            "v",
            "eps",
        ], "Invalid prediction parameterization. Must be 'v' or 'eps'"
        self.pred_param = pred_param

        # Sampling schedule
        assert schedule in [
            "cosine",
            "shifted_cosine",
        ], "Invalid schedule. Must be 'cosine' or 'shifted_cosine'"
        self.schedule = schedule
        self.noise_d = noise_size
        self.image_d = 128
        # self.image_d = image_size

        # Model
        assert isinstance(
            model, nn.Module
        ), "Model must be an instance of torch.nn.Module."
        self.model = model

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params}")

        # Steps
        self.steps = steps
        self.uncertainty_scale = uncertainty_scale
        self.condition_time = condition_time
        self.scheduling_matrix = scheduling_matrix
        self.final_step = final_step
        self.fill_scene_tensor = fill_scene_tensor

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))


        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        logsnr_t = -2 * log(torch.tan(torch.tensor(t_min + t * (t_max - t_min))))

        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with shifted cosine:

        logSNR_shifted(t) = logSNR(t) + 2 * log(noise_d / image_d)

        Args:
        t (int): The timepoint t.
        image_d (int): The image dimension.
        noise_d (int): The noise dimension.

        Returns:
        logsnr_t_shifted (float): The logSNR value at timepoint t.
        """
        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shifted = logsnr_t + 2 * math.log(self.noise_d / self.image_d)

        return logsnr_t_shifted

    def diffuse(self, x, alpha_t, sigma_t):
        """
        Function to diffuse the input tensor x to a timepoint t with the given alpha_t and sigma_t.

        Args:
        x (torch.Tensor): The input tensor to diffuse.
        alpha_t (torch.Tensor): The alpha value at timepoint t.
        sigma_t (torch.Tensor): The sigma value at timepoint t.

        Returns:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        eps_t (torch.Tensor): The noise tensor at timepoint t.
        """
        eps_t = torch.randn_like(x)

        z_t = alpha_t * x + sigma_t * eps_t

        return z_t, eps_t

    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].

        Args:
        x (torch.Tensor): The input tensor to clip.

        Returns:
        x (torch.Tensor): The clipped tensor.
        """
        return torch.clamp(x, -1, 1)

    def constrain(self, x, mode=['clip'], window=3, gamma=0.4,
                raw_map=None, valid_mask=None, 
                threshold=2.7/(2*FEATURE_STD[0]), separation_distance=2.5,
                keep_mask=None, scene_tensor=None):
        
        if type(mode) is list:
            if type(gamma) is not list:
                gamma = [gamma] * len(mode)
            assert len(mode) == len(gamma), "mode and gamma must have the same length, but got {} and {}".format(len(mode), len(gamma))
            for m,g in zip(mode,gamma):
                x=self.constrain(x,m,window,g,raw_map,valid_mask,threshold,separation_distance,keep_mask,scene_tensor)
            return x

        B, NA, NT, D = x.shape
        
        if mode == 'keep':
            x[keep_mask.bool()] = scene_tensor[keep_mask.bool()]
            return x

        elif mode == 'clip':
            x = self.clip(x)
            return x

        elif mode == 'velocity':
            x_pred = decode_scene_tensor(x) # [b, na, nt, dim]
            
            def compute_trajectory(velocity_matrix, start_point, t=0.5):
                trajectory = torch.zeros(velocity_matrix.shape, device=velocity_matrix.device)
                trajectory[..., 0, :] = start_point
                for i in range(velocity_matrix.shape[2]-1):
                    trajectory[...,i + 1,:] = trajectory[...,i,:] + velocity_matrix[...,i,:]*t
                return trajectory          
            
            x_pred[...,4:,:2] = torch.where(valid_mask[...,4:5,None].repeat(1,1,17,2).bool(), 
                                            compute_trajectory(x_pred[...,4:,4:6], x_pred[...,4,:2]), 
                                            x_pred[...,4:,:2])
            x_pred = encode_scene_tensor(x_pred)

        elif mode == 'sma':
            if window is None or NT<window:
                raise ValueError(f"window is illegal,got {window}")
            x_pred=[]
            for t in range(NT-window+1):
                x_avg= (x[:,:,t:t+window]*valid_mask.unsqueeze(-1)[:,:,t:t+window]).sum(-2) / valid_mask[:,:,t:t+window].sum(-1).unsqueeze(-1)
                x_avg = torch.nan_to_num(x_avg, nan=0.0)
                x_pred.append(x_avg)
            x_pred=torch.cat([x[:,:,0:window-1,:],torch.stack(x_pred,dim=2)],dim=2)  

        elif mode == 'collision':
            def resolve_collision(vehicle1, vehicle2, separation_distance=3.):
                # Calculate the centers of the two vehicles
                center1 = vehicle1[:2]  # (x, y)
                center2 = vehicle2[:2]  # (x, y)
                # Calculate the vector between the two vehicles
                delta = center2 - center1
                distance = np.linalg.norm(delta)
                # If the vehicles are closer than the separation distance, adjust their positions
                if distance < separation_distance:
                    direction = delta / distance  # Normalize the direction vector
                    vehicle1[:2] -= direction * (separation_distance - distance) / 2  # Move vehicle1 away
                    vehicle2[:2] += direction * (separation_distance - distance) / 2  # Move vehicle2 away
                return vehicle1, vehicle2  # Return the updated positions of the vehicles

            # After this, you can continue processing the vehicle states in x_pred
            x_pred = np.array(x.cpu())
            x_pred = decode_scene_tensor(x_pred)
            # B,NA,NT,D
            x_pred = np.swapaxes(x_pred,1,2) # B,NT,NA,C
            x_pred = np.reshape(x_pred,(B*NT,NA,D))
            valid_mask = np.reshape(np.swapaxes(np.array(valid_mask.cpu()),1,2),(B*NT,NA))

            # Main loop for processing vehicles
            for b in range(B * NT):
                for a in range(NA):
                    if not valid_mask[b, a]:  # Skip if the vehicle is not valid
                        continue
                    vehicle1 = x_pred[b,a]
                    # Get the current vehicle's state
                    xa, ya, cos_a, sin_a, _, _, la, wa = vehicle1
                    # Check for collisions with other vehicles
                    for other_a in range(NA):
                        if other_a == a or not valid_mask[b, other_a]:  # Skip if it's the same vehicle or not valid
                            continue
                        vehicle2 = x_pred[b,other_a]
                        xb, yb, cos_b, sin_b, _, _, lb, wb = vehicle2
                        ha, hb = np.arctan2(sin_a, cos_a), np.arctan2(sin_b, cos_b)

                        if check_collision(np.array([xa, ya, ha, wa, la]), np.array([xb, yb, hb, wb, lb])):  # If a collision is detected
                            # Call the collision resolution function and receive updated positions
                            vehicle1, vehicle2 = resolve_collision(vehicle1,vehicle2,separation_distance)
                            # Update the vehicle positions in x_pred
                            x_pred[b, a] = vehicle1
                            x_pred[b, other_a] = vehicle2
            valid_mask=valid_mask.reshape(B,NT,NA)
            valid_mask=torch.tensor(valid_mask).permute(0,2,1)
            x_pred=x_pred.reshape(B,NT,NA,D)
            x_pred = encode_scene_tensor(torch.tensor(x_pred).permute(0,2,1,3))
            return x_pred.to(dtype=x.dtype, device=x.device)

        elif mode == 'map':
            if raw_map is None or 0 in raw_map.shape:
                x_pred = x
                print('No valid map provided, skip map constraint')
            else: 
                # only constrain the x,y channel
                x_pred = torch.clone(x[...,:2]) # retail the x,y feature
                x_pred = x_pred.reshape(B,-1,2)
                encoded_map = raw_map[...,:2]
                # find the closet map anchor point for agent at any 
                agent_num,map_num = x_pred.shape[1],encoded_map.shape[1]
                anchor_pred = x_pred.unsqueeze(2).repeat(1,1,map_num,1) - encoded_map.unsqueeze(1).repeat(1,agent_num,1,1)
                min_index = torch.argmin(torch.norm(anchor_pred, dim=-1), dim=-1)
                # index out closet anchor map point
                anchor_pred = encoded_map[torch.arange(min_index.shape[0]).unsqueeze(-1).expand_as(min_index),min_index]
                replace_mask = torch.norm(anchor_pred-x_pred,dim=-1)>threshold
                x_pred[replace_mask] = anchor_pred[replace_mask]
                x_pred = x_pred.reshape(B,NA,NT,2)
                x_pred = torch.cat([x_pred,x[...,2:]],dim=-1)
        else:
            raise ValueError('Unsupported mode')
        
        # if velocity < 0.5, keep token consistent
        consist_mask = torch.norm(decode_scene_tensor(x)[...,4:6],dim=-1) <= 0.7
        consist_mask = consist_mask.unsqueeze(-1).repeat(1,1,1,x_pred.shape[-1]).bool()
        x_pred[consist_mask] = x[consist_mask]

        return x + gamma * (x_pred.to(dtype=x.dtype, device=x.device) - x)

    @torch.no_grad()
    def ddpm_sampler_step(self, z_t, pred, logsnr_t, logsnr_s, grad_list=None,constrain={'mode':['clip']}) :
        """
        Function to perform a single step of the DDPM sampler.

        Args:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        pred (torch.Tensor): The predicted value from the model (v or eps).
        logsnr_t (float): The logSNR value at timepoint t.
        logsnr_s (float): The logSNR value at the sampling timepoint s.

        Returns:
        z_s (torch.Tensor): The diffused tensor at sampling timepoint s.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        recip_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t))
        recipml_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t) - 1.0)

        if self.pred_param == "v":
            x_pred = alpha_t * z_t - sigma_t * pred # pred_x_start
            pred_noise = (recip_alpha_t * z_t - x_pred) / recipml_alpha_t # pred_noise
        elif self.pred_param == "eps":
            x_pred = (z_t - sigma_t * pred) / alpha_t

        x_pred = self.constrain(x_pred, **constrain)
        if grad_list is not None:
            [grad, x_t, target] = grad_list
            mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred) - grad*300
        else:
            mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        
        variance = (sigma_s**2) * c

        return mu, variance

    def compute_start_from_v(self, z_t, pred, logsnr_t, logsnr_s):
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))
        recip_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t))
        recipml_alpha_t = torch.sqrt(1.0 / torch.sigmoid(logsnr_t) - 1.0)

        if self.pred_param == "v":
            x_pred = alpha_t * z_t - sigma_t * pred # pred_x_start
        return x_pred

    def _generate_scheduling_matrix(self, scene_tensor, zero_init=True):
        B, NA, NT = scene_tensor.shape[:3]
        if self.scheduling_matrix == "pyramid":
            return self._generate_pyramid_scheduling_matrix(zero_init, NT, self.uncertainty_scale)[:,np.newaxis,np.newaxis,:].repeat(B, 1).repeat(NA, 2)
        elif self.scheduling_matrix == "full_sequence":
            scheduling_matrix = np.linspace(1, 0, self.steps)[:,np.newaxis,np.newaxis,np.newaxis].repeat(B, 1).repeat(NA, 2).repeat(NT, 3)
            if zero_init:
                scheduling_matrix[:,:,:,:5] = 0 #[m B NA NT]
            return scheduling_matrix.astype(np.float32)
        elif self.scheduling_matrix == "half_half_sequence":
            matrix_1 = np.linspace(1, 0, self.steps)[:,np.newaxis,np.newaxis,np.newaxis].repeat(B, 1).repeat(NA, 2).repeat(NT//2, 3)
            ones_matrix_1 = np.ones((self.steps, B, NA, NT-NT//2), dtype=np.float32)
            scheduling_matrix_1 = np.concatenate([matrix_1, ones_matrix_1], axis=-1)
            zeros_matrix_2 = np.zeros_like(matrix_1)
            matrix_2 = np.linspace(1, 0, self.steps)[:,np.newaxis,np.newaxis,np.newaxis].repeat(B, 1).repeat(NA, 2).repeat(NT-NT//2, 3)
            scheduling_matrix_2 = np.concatenate([zeros_matrix_2, matrix_2], axis=-1)
            scheduling_matrix = np.concatenate([scheduling_matrix_1, scheduling_matrix_2], axis=0)
            if zero_init:
                scheduling_matrix[:,:,:,:5] = 0 #[m B NA NT]
            return scheduling_matrix.astype(np.float32)            
        elif self.scheduling_matrix == "autoregressive":
            return self._generate_pyramid_scheduling_matrix(zero_init, NT, self.steps)[:,np.newaxis,np.newaxis,:].repeat(B, 1).repeat(NA, 2)
        elif self.scheduling_matrix == "trapezoid":
            return self._generate_trapezoid_scheduling_matrix(zero_init, NT, self.uncertainty_scale)[:,np.newaxis,np.newaxis,:].repeat(B, 1).repeat(NA, 2)

    def _generate_pyramid_scheduling_matrix(self, zero_init: bool, horizon: int, uncertainty_scale: float):
        if zero_init:
            horizon = horizon - self.condition_time
        height = self.steps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.steps + int(t * uncertainty_scale) - m
        scheduling_matrix = np.concatenate([np.zeros((height, self.condition_time)), scheduling_matrix], axis=1)
        scheduling_matrix = np.clip(scheduling_matrix, 0, self.steps)/self.steps

        return scheduling_matrix.astype(np.float32)

    def _generate_trapezoid_scheduling_matrix(self, zero_init: bool, horizon: int, uncertainty_scale: float):
        if zero_init:
            horizon = horizon - self.condition_time
        extra_step = (horizon+1) % 2
        height = self.steps + int((horizon + 1) // 2 * uncertainty_scale) + extra_step
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2 + extra_step):
                scheduling_matrix[m, t] = self.steps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.steps + int(t * uncertainty_scale) - m

        scheduling_matrix = np.concatenate([np.zeros((height, self.condition_time)), scheduling_matrix], axis=1)
        scheduling_matrix = np.clip(scheduling_matrix, 0, self.steps)/self.steps

        return scheduling_matrix.astype(np.float32)

    def _filling_scene_tensor(self, scene_tensor, z_t, keep_mask, scaling_matrix):
        if not self.fill_scene_tensor:
            return scaling_matrix, z_t
        # keep_mask: [B NA NT D], scaling_matrix: [m B NA NT]
        z_t = torch.where(keep_mask.bool(), scene_tensor, z_t)
        keep_mask = keep_mask[:,:,:,0].bool().unsqueeze(0).repeat(scaling_matrix.shape[0], 1, 1, 1)
        scaling_matrix = np.where(~keep_mask.cpu().numpy(), scaling_matrix, np.full(scaling_matrix.shape, 0).astype(np.float32))
        return scaling_matrix, z_t

    def guidance_fn(self, x_start, scene_tensor, keep_mask, valid_mask, target):

        # keep_mask[:,:,:5] = 0 # keep only the goals
        mask = keep_mask.clone()
        weight = torch.full_like(x_start, 4.0, device=x_start.device)
        weight = torch.where(mask.bool(), weight, torch.full_like(weight, 1.0, device=x_start.device))
        loss = nn.functional.mse_loss(x_start, target.requires_grad_(), reduction="none")
        loss = -(weight * valid_mask.unsqueeze(-1) * loss).mean() * 10000
        # import pdb; pdb.set_trace()
        # print('target', target[0,0,:,0])
        # print('loss', (x_start-target)[0,0,:,0])

        return loss, (x_start-target)
        # return (x_start-target)

    def sampler_step(self, scene_tensor, x, keep_mask, u_t, u_s, local_context, global_context, valid_mask, schedule_func, use_guidance_fn, compute_mu, target, constrain={'mode':['clip']}):

        # u_t = np.where(
        #     u_t <= 0,
        #     np.full_like(u_t, 0.017),
        #     u_t,
        # )

        orig_x = x.clone().detach()
        logsnr_t = schedule_func(u_t)
        logsnr_t = logsnr_t.clone().detach().to(scene_tensor.device)
        alpha_t = (
            torch.sqrt(torch.sigmoid(logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        sigma_t = (
            torch.sqrt(torch.sigmoid(-logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        scaled_context, eps_t = self.diffuse(x, alpha_t, sigma_t)
        z_t = torch.where(keep_mask.bool(), scaled_context, orig_x)

        logsnr_s = schedule_func(u_s)
        logsnr_s = logsnr_s.to(scene_tensor.device)

        u_t = torch.tensor(u_t, device=x.device)
        u_s = torch.tensor(u_s, device=x.device)

        grad_list = None
        if use_guidance_fn:
            with torch.enable_grad():
                x = z_t.detach().requires_grad_()
                pred = self.model(
                    local_context=local_context,
                    diffused_scene_tensor=x,
                    valid_mask=valid_mask,
                    diffusion_times=u_t,
                    global_context=global_context,
                )
                x_start = self.compute_start_from_v(z_t, pred, logsnr_t.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3]), logsnr_s.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3]))
                guidance_loss, dist = self.guidance_fn(x_start, scene_tensor, keep_mask, valid_mask, target)
                grad_compute = -torch.autograd.grad(
                    guidance_loss,
                    x_start,
                )[0]
                grad = dist/1000
                grad = grad_compute/10
                # print('grad', grad[0,0,:,0])
                # print('grad_compute', grad_compute[0,0,:,0])

                grad_list = [grad, x, target]
            mu, variance = self.ddpm_sampler_step(
                x, pred, logsnr_t.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3]), logsnr_s.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3]), grad_list, constrain
            )
        else:
            pred = self.model(
                local_context=local_context,
                diffused_scene_tensor=z_t,
                valid_mask=valid_mask,
                diffusion_times=u_t,
                global_context=global_context,
            )
            mu, variance = self.ddpm_sampler_step(
                z_t, pred, logsnr_t.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3]), logsnr_s.unsqueeze(-1).repeat(1, 1, 1, scene_tensor.shape[3]), grad_list, constrain
            )

        intermidiates = mu.clone().detach()
        if not compute_mu:
            return mu, intermidiates

        # apply keep_mask
        mu[keep_mask.bool()] = scene_tensor[keep_mask.bool()]
        z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        return z_t, intermidiates

    @torch.no_grad()
    def compute_target(self, scene_tensor, keep_mask):
        # keep_mask[:,:,:5] = 0 # keep only the goals
        keep_flag_mask = keep_mask[:, :, :, 0] == 1
        indices = torch.nonzero(keep_flag_mask, as_tuple=False)
        first_index = indices[0] if indices.numel() > 0 else torch.tensor([0])
        target = scene_tensor.clone()

        return target

    def extrapolate(self, scene_tensor, valid_mask, keep_mask):
        batch_size, num_vehicles, num_times, num_params = scene_tensor.shape
        extrapolated_scene_tensor = torch.zeros_like(scene_tensor, device=scene_tensor.device, dtype=scene_tensor.dtype)

        for b in range(batch_size):
            for v in range(num_vehicles):
                vehicle_mask = keep_mask[b, v, :, 0] # [NT]
                vehicle_scene = scene_tensor[b, v] # [NT, D]

                keep_indices = torch.nonzero(vehicle_mask, as_tuple=False).squeeze(-1)
                if vehicle_mask.sum() == 0:
                    continue
                else:          
                    pre_idx, post_idx = keep_indices[-2], keep_indices[-1]
                    extrapolated_scene_tensor[b, v, :5] = vehicle_scene[:5]
                    extrapolated_scene_tensor[b, v, 5:] = vehicle_scene[4] + (vehicle_scene[post_idx] - vehicle_scene[pre_idx]) * (torch.arange(5, num_times, device=scene_tensor.device).float().unsqueeze(-1) - 4)/ (post_idx - pre_idx) 
                    extrapolated_scene_tensor[b, v, 5:,-2:] = vehicle_scene[4,-2:].unsqueeze(0)
        
        return extrapolated_scene_tensor, []                       
    # @torch.no_grad()
    def sample(
        self,
        scene_tensor,
        valid_mask,
        keep_mask,
        global_context,
        z_t: Optional[torch.Tensor] = None,
        return_intermidates: bool = False,
        use_guidance_fn: bool = False,
        raw_map=None
    ):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x_shape (tuple): The shape of the input tensor.
        global_context (torch.Tensor): The global context tensor.


        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        use_extrapolate = os.environ.get('EXTRAPOLATE', False)
        if use_extrapolate:
            print('extrapolate')
            return self.extrapolate(scene_tensor, valid_mask, keep_mask)

        if z_t is None:
            z_t = torch.randn(scene_tensor.shape).to(scene_tensor.device)
        fuse_mask = False
        if fuse_mask == True:
            keep_mask = keep_mask * valid_mask.unsqueeze(-1)
        local_context = scene_tensor * keep_mask
        # add the valid mask as a channel
        local_context = torch.cat([local_context, keep_mask], dim=-1)

        intermidiates = []
        schedule_func = (
            self.logsnr_schedule_cosine
            if self.schedule == "cosine"
            else self.logsnr_schedule_cosine_shifted
        )
        
        constrain={'mode':self.constrain_mode,'gamma':self.constraint_gamma,'raw_map':raw_map,'valid_mask':valid_mask,'keep_mask':keep_mask.bool(),'scene_tensor':scene_tensor}
        # Steps T -> 1
        scaling_matrix = self._generate_scheduling_matrix(scene_tensor)
        scaling_matrix, z_t = self._filling_scene_tensor(scene_tensor, z_t, keep_mask, scaling_matrix)
        # import pdb; pdb.set_trace()
        gt_replace = os.environ.get('GT_REPLACE', False)
        if gt_replace:
            res = torch.zeros_like(scene_tensor)
            gt_replace_mask = np.logical_not(scaling_matrix[0].astype(bool))
            res[gt_replace_mask] = scene_tensor[gt_replace_mask] 
            if self.fill_scene_tensor:
                res = torch.where(keep_mask.bool(), scene_tensor, res)

        # original_z_t = z_t.clone()
        target = self.compute_target(scene_tensor, keep_mask)
        for t in range(scaling_matrix.shape[0]-1):
            u_t = scaling_matrix[t]
            u_s = scaling_matrix[t+1]
            z_t, mu = self.sampler_step(scene_tensor, z_t, keep_mask, u_t, u_s, local_context, global_context, valid_mask, schedule_func, (use_guidance_fn if t < 100 else False), True, target,constrain)
            # z_t = torch.where(keep_mask.bool(), original_z_t, z_t)
            if return_intermidates:
                intermidiates.append(mu)
            if gt_replace:
                gt_replace_mask = np.logical_and(np.logical_not(u_s.astype(bool)), u_t.astype(bool))
                res[gt_replace_mask] = z_t[gt_replace_mask]
                z_t[gt_replace_mask] = scene_tensor[gt_replace_mask]
            # input() 
        if gt_replace:
            z_t = res

        if self.final_step:
            # Final step
            u_t = np.full_like(u_t, 1 / self.steps)
            u_s = np.full_like(u_s, 0)

            x_pred, _ = self.sampler_step(scene_tensor, z_t, keep_mask, u_t, u_s, local_context, global_context, valid_mask, schedule_func, use_guidance_fn, False, target, constrain)

        x_pred[keep_mask.bool()] = scene_tensor[keep_mask.bool()]
        if return_intermidates:
            intermidiates.append(x_pred)
        # visualize_trajectories(x_pred, valid_mask, raw_map, 30, '/cpfs01/user/yenaisheng/test.png')
        return x_pred, intermidiates
    
    def compute_loss(
        self,
        scene_tensor,
        valid_mask,
        global_context,
        diffusion_times,
        task_mask,
        control_mask,
        raw_map = None
    ):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        scene_tensor (torch.Tensor): The scene tensor (B, NA, NT, ND).
        valid_mask (torch.Tensor): The validity mask (B, NA, NT).
        global_context (torch.Tensor): The global context tensor (B, S, NC).
        diffusion_times (torch.Tensor): The diffusion times (B, NT).
        task_mask (torch.Tensor): The inpainting mask (B, NA, NT, ND).
        control_mask (torch.Tensor): The control mask (B, NA, NT, ND).

        Returns:
        loss (torch.Tensor): The loss value.
        """

        if False and self.distill:
            return self.compute_consist_loss(
                scene_tensor,
                valid_mask,
                global_context,
                diffusion_times,
                task_mask,
                control_mask,
                raw_map
            )

        schedule_func = (
            self.logsnr_schedule_cosine
            if self.schedule == "cosine"
            else self.logsnr_schedule_cosine_shifted
        )
        task_mask = torch.logical_or(task_mask, control_mask)
        # if self.fill_scene_tensor:
        #     keep_mask = task_mask[:,:,:,0].bool()
        #     diffusion_times = torch.where(~keep_mask, diffusion_times, torch.full(diffusion_times.shape, 0.017, device=diffusion_times.device))
        logsnr_t = schedule_func(diffusion_times)
        logsnr_t = logsnr_t.to(scene_tensor.device)
        alpha_t = (
            torch.sqrt(torch.sigmoid(logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        sigma_t = (
            torch.sqrt(torch.sigmoid(-logsnr_t))
            .view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        z_t, eps_t = self.diffuse(scene_tensor, alpha_t, sigma_t)
        # create the local context
        local_context = scene_tensor * task_mask
        # add the valid mask as a channel
        local_context = torch.cat([local_context, task_mask], dim=-1)

        pred = self.model(
            local_context=local_context,
            diffused_scene_tensor=z_t,
            valid_mask=valid_mask,
            diffusion_times=diffusion_times,
            global_context=global_context,
        )

        if self.pred_param == "v":
            eps_pred = sigma_t * z_t + alpha_t * pred
        else:
            eps_pred = pred

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t).clamp_(max=5)
        if self.pred_param == "v":
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr

        weight = weight.view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)

        # add zero weights to invalid pixels
        weight = weight * valid_mask.unsqueeze(-1)
        # weight = weight * torch.logical_not(task_mask)

        loss = torch.sum(weight * (eps_pred - eps_t) ** 2)
        # velo_loss = self.compute_traj_loss(pred, alpha_t, z_t, sigma_t, valid_mask, weight)
        # map_loss = self.compute_map_loss(pred, alpha_t, z_t, sigma_t, valid_mask, weight, raw_map)
        # coll_loss = self.compute_collision_loss(pred, alpha_t, z_t, sigma_t, valid_mask, weight)

        # loss = loss + velo_loss + map_loss + coll_loss
        # loss = loss + map_loss + velo_loss
        return (
            loss,
            dict(
            ),
        )
    
    def compute_collision_loss(self, pred, alpha_t, z_t, sigma_t, validity, weight, min_distance=2.15):

        x_pred = alpha_t * z_t - sigma_t * pred
        x_pred = decode_scene_tensor(x_pred)

        trajectories = x_pred[...,:2]

        B, NA, NT, D = trajectories.shape

        # compute pairwise L2 distance
        traj_exp1 = trajectories.unsqueeze(2)  # (B, NA, 1, NT, D)
        traj_exp2 = trajectories.unsqueeze(1)  # (B, 1, NA, NT, D)
        distances = torch.norm(traj_exp1 - traj_exp2, dim=-1)  # (B, NA, NA, NT)

        # avoid computing self-distance(i == j)
        mask_self = torch.eye(NA, device=trajectories.device).unsqueeze(0).unsqueeze(-1)  # (1, NA, NA, 1)
        distances = distances + mask_self * 1e6  

        # compute validity mask
        validity_exp1 = validity.unsqueeze(2)  # (B, NA, 1, NT)
        validity_exp2 = validity.unsqueeze(1)  # (B, 1, NA, NT)
        valid_mask = (validity_exp1 * validity_exp2).bool()  # (B, NA, NA, NT)

        # compute collision penalty(distances < min_distance)
        collision_mask = (distances < min_distance) & valid_mask  # only consider valid trajectory points
        collision_penalty = (min_distance - distances).clamp(min=0)  # only punish violations
        weight = weight.unsqueeze(1)[...,0]
        loss = torch.sum(collision_penalty * collision_mask * weight)

        return loss

    def compute_repulsion_loss(self, pred, alpha_t, z_t, sigma_t, validity, min_dist=2.15, sigma=0.5):
            x_pred = alpha_t * z_t - sigma_t * pred
            x_pred = decode_scene_tensor(x_pred)

            trajs = x_pred[...,:2]

            B, NA, NT, D = trajs.shape

            # Expand for pairwise distance computation
            traj_i = trajs.unsqueeze(2)  # [B, NA, 1, NT, D]
            traj_j = trajs.unsqueeze(1)  # [B, 1, NA, NT, D]

            # Compute pairwise Euclidean distance
            dist = torch.norm(traj_i - traj_j, dim=-1)  # [B, NA, NA, NT]

            # Compute Gaussian penalty: exp(- ((d - min_dist)² / (2σ²)))
            penalty = torch.exp(-((dist - min_dist) ** 2) / (2 * sigma ** 2))

            # Mask out self-interactions (i == j)
            mask_self = torch.eye(NA, device=trajs.device).unsqueeze(0).unsqueeze(-1)  # [1, NA, NA, 1]
            penalty = penalty * (1 - mask_self)  # [B, NA, NA, NT]

            # Apply validity mask: only consider valid agents at each timestep
            valid_mask = (validity.unsqueeze(2) * validity.unsqueeze(1)).unsqueeze(0)  # [B, NA, NA, NT]
            penalty = penalty * valid_mask

            # Compute final loss, normalized by number of valid interactions
            valid_pairs = valid_mask.sum()
            # loss = penalty.sum() / (valid_pairs + 1e-6)  # Avoid division by zero
            loss = penalty.sum()

            return loss

    def compute_traj_loss(self, pred, alpha_t, z_t, sigma_t, valid_mask, weight):
        x_pred = alpha_t * z_t - sigma_t * pred
        B, NA, NT, D = x_pred.shape
        x_pred = decode_scene_tensor(x_pred)
        freq = 2

        def expand_zeros(valid_mask):
            kernel = torch.tensor([1, 1, 1], dtype=torch.float32).view(1, 1, 3).to(valid_mask.device)
            pad_mask = torch.nn.functional.pad(valid_mask, (1, 1), mode='replicate')
            expanded_mask = torch.nn.functional.conv1d(pad_mask.view(-1, 1, 23), kernel, padding=0, groups=1).view_as(valid_mask)
            return (expanded_mask > 0).float()

        valid_mask = expand_zeros(valid_mask)

        compute_traj = x_pred[...,:2] + x_pred[...,4:6] / freq
        compute_traj = torch.cat((x_pred[...,:1,:2], compute_traj[...,:-1,:]), -2)

        traj_loss = torch.sum(weight * valid_mask.unsqueeze(-1) * encode_scene_tensor(compute_traj-x_pred[...,:2])**2)

        return traj_loss

    def compute_map_loss(self, pred, alpha_t, z_t, sigma_t, valid_mask, weight, raw_map=None):
        # map L2 loss mean
        # compute the L2 distance between the predicted_x and closet map center line point
        if raw_map is None or 0 in raw_map.shape:
            return torch.tensor(0.0).to(pred.device)
        x_pred = alpha_t * z_t - sigma_t * pred
        B, NA, NT, D = x_pred.shape
        # only constrain the x,y channel
        x_pred = x_pred[...,:2].reshape(B,-1,2)
        encoded_map = raw_map[...,:2]
        # find the closet map anchor point for agent at any time
        agent_num,map_num = x_pred.shape[1],encoded_map.shape[1]
        anchor_pred = x_pred.unsqueeze(2).repeat(1,1,map_num,1) - encoded_map.unsqueeze(1).repeat(1,agent_num,1,1)
        min_index = torch.argmin(torch.norm(anchor_pred, dim=-1), dim=-1)
        # index out closet anchor map point
        anchor_pred = encoded_map[torch.arange(min_index.shape[0]).unsqueeze(-1).expand_as(min_index),min_index]
        # if distance > threshold, then compute the l2 loss
        threshold = 3.75/(2*FEATURE_STD[0]) # 3.5m
        distance = torch.norm((anchor_pred-x_pred),dim=-1).reshape(B*NA,NT)     
        in_threshold_mask = distance < threshold
        first_true_indices = torch.argmax(valid_mask.reshape(B*NA,NT), dim=-1) 
        use_mask = in_threshold_mask[torch.arange(B*NA), first_true_indices] * torch.any(valid_mask.reshape(B*NA,NT), dim=-1)
        loss_mask = use_mask.unsqueeze(-1)
        map_loss = (distance * loss_mask * weight.reshape(B*NA,NT)).sum()
        return map_loss
 
    def compute_consist_loss(
        self,
        scene_tensor,
        valid_mask,
        global_context,
        diffusion_times,
        task_mask,
        control_mask,
        raw_map=None
    ):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        scene_tensor (torch.Tensor): The scene tensor (B, NA, NT, ND).
        valid_mask (torch.Tensor): The validity mask (B, NA, NT).
        global_context (torch.Tensor): The global context tensor (B, S, NC).
        diffusion_times (torch.Tensor): The diffusion times (B, NT).
        task_mask (torch.Tensor): The inpainting mask (B, NA, NT, ND).
        control_mask (torch.Tensor): The control mask (B, NA, NT, ND).

        Returns:
        loss (torch.Tensor): The loss value.
        """
        # common part: scene_tensor, task_mask, local_context
        schedule_func = (
            self.logsnr_schedule_cosine
            if self.schedule == "cosine"
            else self.logsnr_schedule_cosine_shifted
        )
        task_mask = torch.logical_or(task_mask, control_mask)

        # create the local context
        local_context = scene_tensor * task_mask
        # add the valid mask as a channel
        local_context = torch.cat([local_context, task_mask], dim=-1)


        # create diffusion from diffusion_times with less noise
        diffusion_times_ln = torch.clamp(diffusion_times - 0.1, 0., 1.)
        diffusion_times = torch.cat([diffusion_times, diffusion_times_ln], dim=0) # [2B, NA, NT, C]

        logsnr_t = schedule_func(diffusion_times)
        logsnr_t = logsnr_t.to(scene_tensor.device)

        alpha_t = (
            torch.sqrt(torch.sigmoid(logsnr_t))
            .view(2*scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        sigma_t = (
            torch.sqrt(torch.sigmoid(-logsnr_t))
            .view(2*scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)
            .to(scene_tensor.device)
        )
        z_t, eps_t = self.diffuse(scene_tensor.repeat(2,1,1,1), alpha_t, sigma_t)

        pred = self.model(
            diffused_scene_tensor=z_t,
            diffusion_times=diffusion_times,
            local_context=local_context.repeat(2,1,1,1),
            valid_mask=valid_mask.repeat(2,1,1),
            global_context=global_context.repeat(2,1,1),
        )

        if self.pred_param == "v":
            x_pred = alpha_t * z_t - sigma_t * pred
        else:
            x_pred = (z_t - sigma_t * pred) / alpha_t

        z_t_hn,z_t_ln = z_t.chunk(2, dim=0)
        x_pred_hn,x_pred_ln = x_pred.chunk(2, dim=0)
        logsnr_t_hn,logsnr_t_ln = logsnr_t.chunk(2, dim=0)
        alpha_t_hn,alpha_t_ln = alpha_t.chunk(2, dim=0)
        sigma_t_hn,sigma_t_ln = sigma_t.chunk(2, dim=0)
        pred_hn,pred_ln = pred.chunk(2, dim=0)

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t_hn).clamp_(max=5)
        if self.pred_param == "v":
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr

        weight = weight.view(scene_tensor.shape[0], scene_tensor.shape[1], scene_tensor.shape[2], 1)

        # add zero weights to invalid pixels
        weight = weight * valid_mask.unsqueeze(-1)
        # weight = weight * torch.logical_not(task_mask)

        loss = torch.sum(weight * (x_pred_hn - scene_tensor) ** 2)
        consist_loss = torch.sum(weight * (x_pred_hn - x_pred_ln) ** 2)
        velo_loss = self.compute_traj_loss(pred_hn, alpha_t_hn, z_t_hn, sigma_t_hn, valid_mask, weight)
        map_loss = self.compute_map_loss(pred_hn, alpha_t_hn, z_t_hn, sigma_t_hn, valid_mask, weight, raw_map)

        loss = self.loss_factor[0] * loss + self.loss_factor[1] * velo_loss + self.loss_factor[2] * map_loss + self.loss_factor[3] * consist_loss
        # loss = loss + map_loss + velo_loss
        return (
            loss,
            dict(
                # diffused_scene_tensor=z_t,
                # pred_scene_tensor=x_pred,
            ),
        )
