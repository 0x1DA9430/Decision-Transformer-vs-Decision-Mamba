import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from typing import Union
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba, Mamba2


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, config, index):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                        .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Convolution(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.window_size = config.window_size
        hidden_size = config.n_embd
        self.conv_proj = config.conv_proj

        self.rtg_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.obs_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.act_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)

        if config.conv_proj:
            self.fc = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #window_size = self.window_size

        # pad the input tensor with zeros along the sequence dimension
        padded_tensor = torch.nn.functional.pad(x, (0, 0, self.window_size - 1, 0)).transpose(1, 2)

        rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
        obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
        act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

        conv_tensor = torch.zeros((x.shape[0], x.shape[2], x.shape[1])).to('cuda' if torch.cuda.is_available() else 'cpu')
        conv_tensor[:, :, ::3] = rtg_conv_tensor
        conv_tensor[:, :, 1::3] = obs_conv_tensor
        conv_tensor[:, :, 2::3] = act_conv_tensor
        conv_tensor = conv_tensor.transpose(1, 2)

        if self.conv_proj:
            conv_tensor = self.dropout(self.fc(conv_tensor))

        return conv_tensor


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, index):
        super().__init__()
        self.token_mixer = config.token_mixer
        self.n_layer = config.n_layer
        self.index = index

        if 'attn' in self.token_mixer:
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config, index)
        
        if 'conv' in self.token_mixer:
            self.lnc = nn.LayerNorm(config.n_embd)
            self.conv = Convolution(config, index)

        if self.token_mixer == 'mamba':
            self.norm_mamba = nn.LayerNorm(config.n_embd)
            self.mamba = Mamba(config.n_embd)
        
        if self.token_mixer == 'mamba2':
            self.norm_mamba = nn.LayerNorm(config.n_embd)
            self.mamba = Mamba2(config.n_embd, expand=4)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp_channels = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  #GELU()
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        if self.token_mixer == 'attn':
            x = x + self.attn(self.ln1(x))
        elif self.token_mixer == 'conv':
            x = x + self.conv(self.lnc(x))
        elif self.token_mixer == 'conv-attn':
            if self.index < self.n_layer - 1:
                x = x + self.conv(self.lnc(x))
            else:
                x = x + self.attn(self.ln1(x))

        elif self.token_mixer == 'mamba' or self.token_mixer == 'mamba2':
            x = x + self.mamba(self.norm_mamba(x))
        else:
            raise NotImplementedError

        x = x + self.mlp_channels(self.ln2(x))
        return x
