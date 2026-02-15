import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .layers.utils import *
from .layers.transformer import SpatialTemporalBlock, CrossAttentionBlock

class GestureDenoiser(nn.Module):
    def __init__(self,
        input_dim=128,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        n_seed=8,
        flip_sin_to_cos= True,
        freq_shift = 0,
        cond_proj_dim=None,
        use_exp=False,
        seq_len=32,
        embed_context_multiplier=4,
        num_body_parts=None,  # Override joint_num (2 for BVH, 3 for SMPL-X)

    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_exp = use_exp
        # Use num_body_parts if provided, else fall back to original logic
        if num_body_parts is not None:
            self.joint_num = num_body_parts
        else:
            self.joint_num = 3 if not self.use_exp else 4
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.mytimmblocks = nn.ModuleList([
            SpatialTemporalBlock(dim=self.latent_dim,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(self.num_layers)])
            
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        self.seq_len = seq_len
        self.embed_context_multiplier = embed_context_multiplier
        
        self.embed_text = nn.Linear(self.input_dim * self.joint_num * self.embed_context_multiplier, self.latent_dim)

        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

        self.rel_pos = SinusoidalEmbeddings(self.latent_dim)
        self.input_process = InputProcess(self.input_dim , self.latent_dim)
        self.input_process2 = nn.Linear(self.latent_dim*2, self.latent_dim)
        
        self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim, self.activation, cond_proj_dim=cond_proj_dim, zero_init_cond=True)
        time_dim = self.latent_dim
        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        if cond_proj_dim is not None:
            self.cond_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        
        # Null condition embedding for classifier-free guidance
        self.null_cond_embed = nn.Parameter(torch.zeros(self.seq_len, self.latent_dim*self.joint_num), requires_grad=True)

    # dropout mask
    def prob_mask_like(self, shape, prob, device):
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


    def forward(self, x, timesteps, cond_time=None, seed=None, at_feat=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        seed: [batch_size, njoints, nfeats]
        """
        
        if x.shape[2] == 1:
            x = x.squeeze(2)
            x = x.reshape(x.shape[0], self.joint_num, -1, x.shape[2])
       
        bs, njoints, nfeats, nframes = x.shape      # [bs, 3, 128, 32]
        
        # need to be an arrary, especially when bs is 1
        # timesteps = timesteps.expand(bs).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=x.dtype)

        if cond_time is not None and self.cond_proj is not None:
            cond_time = cond_time.expand(bs).clone()
            cond_emb = self.cond_proj(cond_time)
            cond_emb = cond_emb.to(dtype=x.dtype)
            emb_t = self.time_embedding(time_emb, cond_emb)
        else:
            emb_t = self.time_embedding(time_emb)
        
        if self.n_seed != 0:
            embed_text = self.embed_text(seed.reshape(bs, -1))
            emb_seed = embed_text

        xseq = self.input_process(x)

        # add the seed information
        embed_style_2 = (emb_seed + emb_t).unsqueeze(1).unsqueeze(2).expand(-1, self.joint_num, self.seq_len, -1)  # (300, 256)
        xseq = torch.cat([embed_style_2, xseq], axis=-1)  # -> [88, 300, 576]
        
        xseq = self.input_process2(xseq)
        

        # apply the positional encoding
        xseq = xseq.reshape(bs * self.joint_num, nframes, -1)
        pos_emb = self.rel_pos(xseq)
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
        xseq = xseq.reshape(bs, self.joint_num, nframes, -1)
        xseq = xseq.view(bs, self.seq_len, -1)

        
        for block in self.cross_attn_blocks:
            xseq = block(xseq, at_feat)

        xseq = xseq.view(bs, njoints, self.seq_len, -1)
        for block in self.mytimmblocks:
            xseq = block(xseq)
        
        output = xseq                

        output = self.output_process(output)
        return output