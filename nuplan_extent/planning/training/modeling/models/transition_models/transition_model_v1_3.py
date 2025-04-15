import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
from einops import rearrange, repeat
from functools import partial
from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.visual_gpt_v2 import GPTV2
from nuplan_extent.planning.training.modeling.models.modules.generative_model.llama3_model import Llama3Model, ModelArgs
# from nuplan_extent.planning.training.modeling.models.modules.generative_model.mamba_model import MambaModel, MambaConfig

from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.base_model import GPTConfig
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type_v1_1 import VocabularyStateType, PositionalStateType
from nuplan_extent.planning.training.modeling.models.transition_models.base_transition_model import BaseTransitionModel
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils
import numpy as np
# from nuplan_extent.planning.training.modeling.diffusion.df_planning import DiffusionForcingPlanning
np.set_printoptions(precision=2, suppress=True)

class TransitionModelV1_3(nn.Module):
    def __init__(self,
                 diffusion_model: nn.Module,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_embd: int = 768,
                 block_size: int = 1024,
                 bias: bool = True,
                 dropout: float = 0.1,
                 frame_dropout_rate: float = 0.2,
                 init_from: str = 'gpt2',
                 temperature: float = 1.1,
                 top_k: int = 40,
                 ):
        super().__init__()
        meta_vocab_size = VocabularyStateType.PAD_TOKEN.vocal_size
        temperature = float(os.environ.get('TEMPERATURE', temperature))
        top_k = int(os.environ.get('TOPK', top_k))

        self.use_sliding_window = True # True to enable sliding window decoding
        self.enable_qkv_cache = False  # True to enable qkv cache for llama3


        self.block_size = block_size
        self.n_embd = n_embd
        self.temperature = temperature
        self.top_k = top_k
        self.diffusion_model = diffusion_model



    def forward_train(
            self,
            image_features,
            tokenized_arrays,
            embedder,
            token_decoder,
            latent_features=None,
            last_frame_only=False):
        """
        Forward pass for training.

        Args:
            image_features (Tensor): The input image features.
            tokenized_arrays (Tensor): The tokenized input arrays.
            embedder (Embedder): The embedder object used for embedding.
            token_decoder (TokenDecoder): The token decoder object used for decoding.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The predicted agent logits, predicted agent tokens, and target tokenized state.
        """
        device = image_features.device
        
        tokenized_arrays, mask = gutils.reorgnize_array(tokenized_arrays, block_size=self.block_size)
        token_embeddings = torch.from_numpy(tokenized_arrays).to(device)[:,:,:,2:9].float()
        token_embeddings = torch.where(~repeat(torch.from_numpy(mask).to(device).bool(), "b t a -> b t a dim", dim = token_embeddings.shape[-1]), torch.zeros(token_embeddings.shape).to(device), token_embeddings)
        output_features = token_embeddings

        # output_features = repeat(token_embeddings[:,:,0,:], 'B T dim -> B T 512 dim')
        # mask = np.ones(mask.shape)

        frame_index = None

        diffusion_out = None
        if self.training:
            diffusion_out = self.diffusion_model.training_step(output_features, image_features, frame_index, mask)
        else:
            diffusion_out = self.diffusion_model.validation_step(output_features, image_features, frame_index, mask)
        output_features = diffusion_out['xs_pred']

        hidden = output_features

        return None, None, None, None, None, hidden, diffusion_out


    def forward_inference_without_cache(
                self,
                image_features,
                tokenized_arrays,
                embedder,
                token_decoder,
                render,
                latent_features=None,
                num_imagine_frames=16,
                num_conditioned_frames=4,
                update_initial_prompts=False):

        device = image_features.device

        raw_array = deepcopy(tokenized_arrays)
        
        tokenized_arrays, mask = gutils.reorgnize_array(tokenized_arrays, block_size=self.block_size)
        token_embeddings = torch.from_numpy(tokenized_arrays).to(device)[:,:,:,2:9].float()
        token_embeddings = torch.where(~repeat(torch.from_numpy(mask).to(device).bool(), "b t a -> b t a dim", dim = token_embeddings.shape[-1]), torch.zeros(token_embeddings.shape).to(device), token_embeddings)
        output_features = token_embeddings

        # output_features = repeat(token_embeddings[:,:,0,:], 'B T dim -> B T 512 dim')
        # mask = np.ones(mask.shape)

        frame_index = None

        diffusion_out = None
        if self.training:
            diffusion_out = self.diffusion_model.training_step(output_features,image_features, frame_index, mask)
        else:
            diffusion_out = self.diffusion_model.validation_step(output_features, image_features, frame_index, mask)
        output_features = diffusion_out['xs_pred']

        pred_agent_tokens = output_features.detach().cpu().numpy()
        pred_agent_tokens[mask == 0] = np.nan
        last_tokenized_arrays = rearrange(deepcopy(pred_agent_tokens), "B T seq dim -> B (T seq) dim")
        hist_tokenized_arrays = rearrange(deepcopy(tokenized_arrays), "B T seq dim -> B (T seq) dim")

        hist_tokenized_arrays[:,:,2:9] = last_tokenized_arrays

        hist_tokenized_arrays = gutils.reformat_array(hist_tokenized_arrays, raw_array)
        hist_tokenized_arrays = gutils.mark_as_generated(hist_tokenized_arrays)
        
        return hist_tokenized_arrays
