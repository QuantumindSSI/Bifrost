"""Baseline attention implementations for demo comparisons."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    """Vanilla multi-head dot-product attention (baseline)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.W_o(out)
        out = self.norm(out + x)
        return out, weights
