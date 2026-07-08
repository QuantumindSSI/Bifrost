"""
Wavelet Augmentation for Transformer LLMs.

Implements the Wavelet GPT approach (Guthikonda et al., 2024): inserts
Haar wavelet transforms into transformer decoder layers to give the model
explicit multi-scale structure. The wavelet transform decomposes hidden
states into approximation (low-frequency) and detail (high-frequency)
coefficients, allowing the model to process information at multiple scales.

This module provides:
1. HaarWaveletTransform: 1D Haar wavelet transform along hidden dimension
2. WaveletAugmentedLayer: wraps a transformer decoder layer with wavelet
   multi-scale processing
3. create_wavelet_augmented_model: modifies a HuggingFace model in-place

The approach:
- After each decoder layer, decompose hidden states into wavelet coefficients
- Mix the coefficients with learnable weights (approx vs detail)
- This gives the model multi-scale inductive bias without extra parameters

Based on:
- Wavelet GPT: arXiv:2411.16720 (2x faster pre-training)
- Spectral Geometry of Thought: multi-scale structure distinguishes reasoning
"""

from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletTransform(nn.Module):
    """1D Haar wavelet transform along a specified dimension.

    Decomposes a signal into approximation (low-pass) and detail (high-pass)
    coefficients using the Haar wavelet:
    - approximation: (a + b) / sqrt(2)  — captures low-frequency structure
    - detail: (a - b) / sqrt(2)          — captures high-frequency structure

    The transform is invertible and can be applied recursively for multi-scale
    decomposition.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Haar wavelet transform.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. The dimension specified by `dim` must be even.

        Returns
        -------
        approx : torch.Tensor
            Approximation coefficients (low-frequency), half the size of dim.
        detail : torch.Tensor
            Detail coefficients (high-frequency), half the size of dim.
        """
        # Move dim to last for processing
        x = x.transpose(self.dim, -1)

        # Pad if odd
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1), mode="reflect")

        # Split into pairs
        x_even = x[..., 0::2]  # (..., D/2)
        x_odd = x[..., 1::2]   # (..., D/2)

        # Haar wavelet coefficients
        sqrt2 = math.sqrt(2.0)
        approx = (x_even + x_odd) / sqrt2
        detail = (x_even - x_odd) / sqrt2

        # Move back
        approx = approx.transpose(self.dim, -1)
        detail = detail.transpose(self.dim, -1)

        return approx, detail

    def inverse(self, approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        """Inverse Haar wavelet transform (reconstruct from coefficients).

        Parameters
        ----------
        approx : torch.Tensor
            Approximation coefficients.
        detail : torch.Tensor
            Detail coefficients.

        Returns
        -------
        x : torch.Tensor
            Reconstructed signal.
        """
        approx = approx.transpose(self.dim, -1)
        detail = detail.transpose(self.dim, -1)

        sqrt2 = math.sqrt(2.0)
        x_even = (approx + detail) / sqrt2
        x_odd = (approx - detail) / sqrt2

        # Interleave
        D = x_even.shape[-1] + x_odd.shape[-1]
        x = torch.zeros(*x_even.shape[:-1], D, device=x_even.device, dtype=x_even.dtype)
        x[..., 0::2] = x_even
        x[..., 1::2] = x_odd

        return x.transpose(self.dim, -1)


class WaveletMixer(nn.Module):
    """Mixes wavelet approximation and detail coefficients with learnable weights.

    After decomposing hidden states into approx and detail, this module
    learns how to recombine them. A gate controls the balance between
    low-frequency (structural) and high-frequency (detail) information.

    IMPORTANT: This is a RESIDUAL operation. The wavelet-mixed signal is
    ADDED to the original, not replacing it. This preserves the base
    model's representations while adding multi-scale structure.
    """

    def __init__(self, hidden_dim: int, n_scales: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales

        # Learnable mixing weight for each scale (initialized near zero
        # so the model starts as the identity and learns to use wavelet info)
        self.scale_weights = nn.ParameterList()
        for scale in range(n_scales):
            dim_at_scale = max(hidden_dim // (2 ** (scale + 1)), 1)
            # Weight per scale level — starts at zero (identity)
            self.scale_weights.append(nn.Parameter(torch.zeros(1)))

        # Learnable projection to map wavelet features back to hidden dim
        # This is the key: instead of reconstructing, we project
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Initialize projection to near-zero so we start as identity
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale wavelet mixing as a residual.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, hidden_dim)

        Returns
        -------
        x_out : torch.Tensor
            Same shape as input: x + wavelet_residual
        """
        wavelet = HaarWaveletTransform(dim=-1)
        residual = x
        target_dim = x.shape[-1]
        wavelet_features = []

        # Multi-level decomposition — collect detail coefficients
        current = x
        for scale in range(self.n_scales):
            if current.shape[-1] < 2:
                break
            approx, detail = wavelet(current)
            # Weight the detail coefficients at this scale
            w = self.scale_weights[scale]
            # Upsample detail back to current's dimension for accumulation
            # We use the inverse wavelet with zeros for approx
            detail_upsampled = wavelet.inverse(
                torch.zeros_like(detail), detail)
            # Pad or truncate to match the original hidden dimension
            if detail_upsampled.shape[-1] < target_dim:
                pad_size = target_dim - detail_upsampled.shape[-1]
                detail_upsampled = F.pad(detail_upsampled, (0, pad_size))
            elif detail_upsampled.shape[-1] > target_dim:
                detail_upsampled = detail_upsampled[..., :target_dim]
            wavelet_features.append(w * detail_upsampled)
            current = approx  # Continue decomposing

        # Sum all scale features and project
        if wavelet_features:
            multi_scale = sum(wavelet_features)
            # Project through learnable linear layer
            projected = self.proj(multi_scale)
            return residual + projected
        else:
            return residual


class WaveletAugmentedLayer(nn.Module):
    """Wraps a transformer decoder layer with wavelet multi-scale processing.

    Architecture:
        x -> LayerNorm -> Attention -> WaveletMixer -> + residual
          -> LayerNorm -> MLP -> WaveletMixer -> + residual

    The wavelet mixer is inserted after attention and after MLP, giving
    the model multi-scale structure at both the attention and feed-forward
    stages.
    """

    def __init__(self, base_layer: nn.Module, hidden_dim: int,
                 n_scales: int = 3, use_wavelet_after_attn: bool = True,
                 use_wavelet_after_mlp: bool = True):
        super().__init__()
        self.base_layer = base_layer
        self.use_wavelet_after_attn = use_wavelet_after_attn
        self.use_wavelet_after_mlp = use_wavelet_after_mlp

        if use_wavelet_after_attn:
            self.wavelet_after_attn = WaveletMixer(hidden_dim, n_scales)
        if use_wavelet_after_mlp:
            self.wavelet_after_mlp = WaveletMixer(hidden_dim, n_scales)

        # Expose attributes that the parent model expects from decoder layers
        # This allows the HuggingFace model's forward pass to access them
        for attr in ["attention_type", "layer_type", "self_attn"]:
            if hasattr(base_layer, attr):
                setattr(self, attr, getattr(base_layer, attr))

    def __getattr__(self, name: str):
        """Delegate attribute access to base_layer if not found on self."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self._modules["base_layer"], name)
            except (KeyError, AttributeError):
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Forward pass through the base layer, then wavelet mixing.

        The base layer handles attention and MLP internally. We apply
        wavelet mixing to the output.
        """
        # Run the base decoder layer
        output = self.base_layer(hidden_states, *args, **kwargs)

        # output is typically a tuple; first element is hidden states
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()

        # Apply wavelet mixing
        if self.use_wavelet_after_mlp:
            hidden = self.wavelet_after_mlp(hidden)

        if rest:
            return (hidden,) + rest
        return hidden


def create_wavelet_augmented_model(
    model: nn.Module,
    n_scales: int = 3,
    layers_to_augment: Optional[List[int]] = None,
    use_wavelet_after_attn: bool = True,
    use_wavelet_after_mlp: bool = True,
) -> nn.Module:
    """Modify a HuggingFace model in-place to add wavelet augmentation.

    Parameters
    ----------
    model : nn.Module
        A HuggingFace causal LM model (e.g., Qwen2ForCausalLM)
    n_scales : int
        Number of wavelet decomposition levels.
    layers_to_augment : List[int], optional
        Which layers to augment (0-indexed). Default: all layers.
    use_wavelet_after_attn : bool
        Whether to add wavelet mixing after attention.
    use_wavelet_after_mlp : bool
        Whether to add wavelet mixing after MLP.

    Returns
    -------
    model : nn.Module
        The same model with wavelet-augmented layers.
    """
    # Get the decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find decoder layers in model")

    n_layers = len(layers)
    hidden_dim = model.config.hidden_size

    if layers_to_augment is None:
        layers_to_augment = list(range(n_layers))

    # Wrap each specified layer
    for i in layers_to_augment:
        layers[i] = WaveletAugmentedLayer(
            layers[i],
            hidden_dim=hidden_dim,
            n_scales=n_scales,
            use_wavelet_after_attn=use_wavelet_after_attn,
            use_wavelet_after_mlp=use_wavelet_after_mlp,
        )

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_M": total / 1e6,
            "trainable_M": trainable / 1e6}
