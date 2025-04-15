"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

from .base_pytorch_algo import BasePytorchAlgo
from .models.diffusion import Diffusion
import random
from einops import rearrange, repeat

class DiffusionForcingVideo(BasePytorchAlgo):
    def __init__(self, 
        x_shape,
        frame_stack,
        guidance_scale,
        context_frames,
        chunk_size,
        external_cond_dim,
        causal,
        uncertainty_scale,
        timesteps,
        sampling_timesteps,
        clip_noise,
        cum_snr_decay,
        frame_skip,
        diffusion
    ):
        # self.cfg = cfg

        self.x_shape = x_shape
        self.frame_stack = frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= frame_stack
        self.guidance_scale = guidance_scale
        self.context_frames = context_frames
        self.chunk_size = chunk_size
        self.external_cond_dim = external_cond_dim
        self.causal = causal

        self.uncertainty_scale = uncertainty_scale
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.clip_noise = clip_noise
        self.frame_skip = frame_skip
        self.cum_snr_decay = cum_snr_decay

        self.cum_snr_decay = self.cum_snr_decay ** (self.frame_stack * frame_skip)
        self.cfg = {}
        self.cfg['noise_level'] = 'random_all'
        self.cfg['scheduling_matrix'] = 'full_sequence' # pyramid, full_sequence, autoregressive, trapezoid
        # self.cfg['data_mean'] = [2.05, 9.8, 15.09, 7.17, 0.06, 0.52, -0.01, 1.43, 3.11, 50.94, 30.63, 0.47]
        # self.cfg['data_std'] = [0.42, 6.01, 37.96, 24.81, 1.81, 3.88, 1.87, 0.65, 2.29, 49.43, 33.32, 0.51]
        # self.cfg['data_mean'] = [15.09, 7.17, 0.06, 0.52, -0.01, 1.43, 3.11]
        # self.cfg['data_std'] = [37.96, 24.81, 1.81, 3.88, 1.87, 0.65, 2.29]

        self.cfg['data_mean'] = [0., 0., 0., 0., 0., 2., 4.5]
        self.cfg['data_std'] = [40., 40., 1., 4, 2, 0.8, 2.5]

        # self.cfg['data_mean'] = [0., 0., 0.]
        # self.cfg['data_std'] = [40., 40., 1.]


        # self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay ** (self.frame_stack * cfg.frame_skip)

        self.validation_step_outputs = []
        super().__init__()
        self.diffusion_model = diffusion


    def _build_model(self):
        # self.diffusion_model = Diffusion(
        #     x_shape=self.x_stacked_shape,
        #     external_cond_dim=self.external_cond_dim,
        #     is_causal=self.causal,
        #     cfg=self.cfg.diffusion,
        # )
        self.register_data_mean_std(self.cfg['data_mean'], self.cfg['data_std'])
        # pass


    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def _training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs))
        loss = self.reweight_loss(loss, masks)

        # log the loss
        if batch_idx % 20 == 0:
            self.log("training/loss", loss)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def get_local_context(self, xs, masks):
        local_context = xs.clone()

        masks = torch.from_numpy(masks).to(xs.device)
        masks = repeat(masks, "b t a -> t b c a 1", c=local_context.shape[2])
        discard = ~masks.bool()
        local_context = torch.where(discard, torch.full_like(local_context, 0), local_context)
        local_context[5:] = 0

        return torch.cat([local_context, masks[:,:,0][:,:,None].float()], dim=2)

    def training_step(self, feature, conditions, frame_index, token_mask):

        xs = feature
        # xs = self.pre_split(xs)
        # import pdb; pdb.set_trace()
        batch_size, n_frames, n_agents, dim = xs.shape

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        # masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        # if self.external_cond_dim:
        #     conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
        #     conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        # else:
        #     conditions = [None for _ in range(n_frames)]
        # xs = self.pre_processing(xs, token_mask)
        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) n c -> t b (fs c) n 1", fs=self.frame_stack).contiguous()
        # xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()
        all_conditions = [conditions, self.get_local_context(xs, token_mask), token_mask]
        conditions = all_conditions
        # conditions = [None for _ in range(n_frames)]

        # xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs))
        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs, token_mask))
        loss = self.reweight_loss_mask(loss, token_mask)

        xs = self._unstack(xs)
        xs_pred = self._unstack(xs_pred)
        # xs_pred = self.post_processing(xs_pred.clone(), token_mask)
        # xs_pred = self.post_split(feature, xs_pred)

        # xs_post = self.post_processing(xs.clone(), token_mask)

        # new_loss = rearrange(F.mse_loss(xs_post, xs, reduction="none"), "b t n c -> t b c n 1")
        # new_loss = self.reweight_loss_mask(new_loss, token_mask)
        # old_loss = rearrange(F.mse_loss(xs_pred, xs, reduction="none"), "b t n c -> t b c n 1")
        # old_loss = self.reweight_loss_mask(old_loss, token_mask)
        # import pdb; pdb.set_trace()

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict


    @torch.no_grad()
    def validation_step(self, feature, conditions, frame_index, token_mask):

        xs = feature
        # xs = self.pre_split(xs)
        batch_size, n_frames, n_agents, dim = xs.shape
        self.n_tokens = n_frames // self.frame_stack

        # xs = self.pre_processing(xs, token_mask)
        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) n c -> t b (fs c) n 1", fs=self.frame_stack).contiguous()
        discard = ~rearrange(torch.from_numpy(token_mask).to(xs.device), "b t a -> t b a").bool()
        # xs = torch.where(repeat(discard, "t b a -> t b (fs c) a 1", fs=self.frame_stack, c=dim), torch.clamp(torch.randn_like(xs), -self.clip_noise, self.clip_noise), xs)

        all_conditions = [conditions, self.get_local_context(xs, token_mask), token_mask]
        conditions = all_conditions
        # conditions = [None for _ in range(n_frames)]

        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames
        # masks = torch.ones(n_frames, batch_size).to(xs.device)

        # pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)
            # scheduling_matrix = np.random.randint(0, self.timesteps, size=(1, n_frames, n_agents))
            # here is for token-level noise
            scheduling_matrix = repeat(scheduling_matrix, "m t -> m t a", a=n_agents)

            # new_noise_levels = torch.where(~discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            # xs_pred = torch.cat([xs_pred, chunk], 0)
            xs_pred = xs.clone()

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            # pbar.set_postfix(
            #     {
            #         "start": start_frame,
            #         "end": curr_frame + horizon,
            #     }
            # )

            for m in range(scheduling_matrix.shape[0] - 1):
                if len(scheduling_matrix.shape) == 2:
                    from_noise_levels = np.concatenate((np.zeros((curr_frame, ), dtype=np.int64), scheduling_matrix[m]))[:, None].repeat(batch_size, axis=1)
                    to_noise_levels = np.concatenate((np.zeros((curr_frame, ), dtype=np.int64), scheduling_matrix[m + 1]))[:, None].repeat(batch_size, axis=1)
                else:
                    from_noise_levels = np.concatenate((np.zeros((curr_frame, n_agents), dtype=np.int64), scheduling_matrix[m]))[:, None].repeat(batch_size, axis=1)
                    to_noise_levels = np.concatenate((np.zeros((curr_frame, n_agents), dtype=np.int64), scheduling_matrix[m + 1]))[:, None].repeat(batch_size, axis=1)


                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # if len(scheduling_matrix.shape) == 3:
                #     from_noise_levels[:start_frame] = torch.where(discard, torch.full_like(from_noise_levels, self.sampling_timesteps - 1), from_noise_levels)[:start_frame]
                #     to_noise_levels[:start_frame] = torch.where(discard, torch.full_like(to_noise_levels, self.sampling_timesteps - 1), to_noise_levels)[:start_frame]

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    conditions,
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )
                # loss = F.mse_loss(xs_pred, xs, reduction="none")
                # loss = self.reweight_loss_mask(loss, token_mask)

            curr_frame += horizon
            # pbar.update(horizon)
            # import pdb; pdb.set_trace()

        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss_mask(loss, token_mask)

        # xs = self._unstack_and_unnormalize(xs)
        # xs_pred = self._unstack_and_unnormalize(xs_pred)
        # self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))
        # return loss

        xs = self._unstack(xs)
        xs_pred = self._unstack(xs_pred)

        metric = F.mse_loss(xs_pred, xs, reduction="none")
        metric = self.reweight_metric_mask(rearrange(metric, "b t a c -> t b c a 1"), token_mask)
        # xs_pred = self.post_processing(xs_pred.clone(), token_mask)
        # xs_pred = self.post_split(feature, xs_pred)

        # xs_pred = torch.where(repeat(discard, "t b a -> b t a c", c=xs.shape[-1]), xs, xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
            "metric": metric
        }
        # import pdb; pdb.set_trace()

        return output_dict


    @torch.no_grad()
    def _validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, masks = self._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    conditions[start_frame : curr_frame + horizon],
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )

            curr_frame += horizon
            pbar.update(horizon)

        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)
        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs, namespace="test")

    def test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        xs_gt = xs.clone()
        noise_level_list = []
        for i in range(xs.shape[1]):
            random_number = random.random()
            xs = xs_gt[:,i,...][:,None]
            if random_number > 0.8:
                noise_levels = self._generate_token_noise_levels(xs)
            elif random_number > 0.6:
                noise_levels = self._generate_frame_noise_levels(xs)
                noise_levels = repeat(noise_levels, "f b -> f b a", a=xs.shape[3])
            elif random_number > 0.4:
                noise_levels = self._generate_agent_noise_levels(xs)
            elif random_number > 0.2:
                noise_levels = self._generate_agent_noise_levels(xs)
                # noise_levels[:5] = 0
                noise_levels[5:] = torch.randint(0, self.timesteps, size=())
            else:
                noise_levels = self._generate_forward_noise_levels(xs)
            noise_levels[:5] = 0
            noise_levels[5:] = torch.randint(0, self.timesteps, size=())
            noise_level_list.append(noise_levels)
        noise_levels = torch.cat(noise_level_list, dim=1)

        if masks is not None:
            masks = torch.from_numpy(masks).to(xs.device)
            masks = rearrange(masks, "b t a -> t b a")
            discard = ~masks.bool()
            new_noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        # import pdb; pdb.set_trace()

        return noise_levels

    def _generate_token_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, dim, agent_num, _ = xs.shape
        if self.cfg['noise_level'] == "random_all": # entirely random noise levels
            noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size, agent_num), device=xs.device)

        return noise_levels

    def _generate_frame_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape
        if self.cfg['noise_level'] == "random_all": # entirely random noise levels
            noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)
        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels

    def _generate_agent_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, dim, agent_num, _ = xs.shape
        if self.cfg['noise_level'] == "random_all": # entirely random noise levels
            noise_levels = torch.randint(0, self.timesteps, (num_frames, agent_num), device=xs.device)
            noise_levels = repeat(noise_levels, "f a -> f b a", b=batch_size)

        return noise_levels

    def _generate_forward_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, dim, agent_num, _ = xs.shape

        noise_levels = torch.randint(0, self.timesteps, (num_frames, agent_num), device=xs.device)
        noise_levels = repeat(noise_levels, "f a -> f b a", b=batch_size)
        noise_levels[:5] = 0

        return noise_levels

    def _generate_scheduling_matrix(self, horizon: int):
        if self.cfg['scheduling_matrix'] == "pyramid":
            return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
        elif self.cfg['scheduling_matrix'] == "full_sequence":
            return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
        elif self.cfg['scheduling_matrix'] == "autoregressive":
            return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
        elif self.cfg['scheduling_matrix'] == "trapezoid":
            return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def reweight_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.frame_stack,
            )
            loss = loss * weight

        return loss.mean()

    def reweight_loss_mask(self, loss, weight=None, valid=True, reweight=False):
        if weight is not None:
            weight = torch.from_numpy(weight).to(loss.device)
            weight = repeat(weight, "b t a -> t b c a 1", c=loss.shape[2])
            loss = loss * weight
        if reweight:
            loss[:,:,:3] *= 9
            loss[:,:,3:] *= 1/9
            # loss[:,:,:3] *= 3
            # loss[:,:,3:] *= 1/3
        if valid:
            return loss.sum() / weight.sum()
        else:
            return loss.mean()

    def reweight_metric_mask(self, loss, weight=None, valid=True):
        if weight is not None:
            weight = torch.from_numpy(weight).to(loss.device)
            weight = repeat(weight, "b t a -> t b c a 1", c=loss.shape[2])
            print(weight.shape, loss.shape)
            loss = loss * weight / 10
            loss, weight = loss[:,:,:3], weight[:,:,:3]
        if valid:
            return loss.sum() / weight.sum()
        else:
            return loss.mean()


    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(self.device)
        std = self.data_std.reshape(shape).to(self.device)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(self.device)
        std = self.data_std.reshape(shape).to(self.device)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)

    def _unstack(self, xs):
        xs = rearrange(xs, "t b (fs c) n 1 -> b (t fs) n c", fs=self.frame_stack)
        # return xs
        return self._unnormalize_x(xs)

    def pre_processing(self, xs, mask):
        # need to be fix!! some bugs when padding zeros
        xs_pre = xs.clone()
        for i in range(1, xs.shape[1]):
            xs_pre[:,i,:,3:5] = (xs[:,i,:,:2] - xs[:,i-1,:,:2]) / 0.5
        # discard = ~torch.from_numpy(mask).to(xs.device).bool()
        # discard = repeat(discard, "b t a -> b t a c", c=xs.shape[-1])
        # xs_pre = torch.where(discard, xs, xs_pre)
        return xs_pre

    def post_processing(self, xs, mask):
        # import pdb; pdb.set_trace()
        xs_post = xs.clone()
        # use velocity x/y to calculate the waypoints
        for i in range(1, xs.shape[1]):
            xs_post[:,i,:,:2] = 0.5 * xs[:,i,:,3:5] + xs_post[:,i-1,:,:2]
        return xs_post

    def pre_split(self, xs):
        xs_pre = xs.clone()
        return xs[:,:,:,:3]

    def post_split(self, xs, xs_pred):
        xs_post = xs.clone()
        xs_post[:,:,:,:3] = xs_pred
        return xs_post