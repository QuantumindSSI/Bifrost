"""
S2_SpectralBinding — Semantic Frequency Binding via Resonance Attention.

Wraps ResonanceAttention to operate directly on SpectralTensor objects,
producing bound spectral candidates ready for attractor identification (S3).

Responsibilities (per Engineering Script S2):
    - Compute phase-coherence attention maps.
    - Bind spectral modes into semantic candidates.
    - Generate signals for attractor identification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spectral_tensor import SpectralTensor
from .attention import ResonanceAttention


class S2SpectralBinding(nn.Module):
    """
    Stage S2: SpectralTensor → bound spectral embedding + coherence map.

    Parameters
    ----------
    d_model : int
        Feature dimension (must match S1 output n_freq or be projected).
    n_heads : int
        Number of resonance attention heads.
    n_bands : int
        Spectral bands for coherence scoring.
    dropout : float
        Attention dropout.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_bands: int = 8,
        dropout: float = 0.1,
        n_freq_in: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_freq_in = n_freq_in  # if set, adds a persistent input projection

        # Persistent input projection when n_freq_in != d_model
        in_dim = n_freq_in if n_freq_in is not None else d_model
        self.amp_proj = nn.Linear(in_dim, d_model)
        self.phase_proj = nn.Linear(in_dim, d_model)

        self.resonance = ResonanceAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_bands=n_bands,
            dropout=dropout,
        )

    def forward(
        self,
        st: SpectralTensor,
        input_proj: Optional[nn.Linear] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Parameters
        ----------
        st : SpectralTensor
            Output of S1.  ``st.amplitude`` shape: ``(batch, channels, n_freq)``.
        input_proj : nn.Linear, optional
            External projection layer if n_freq != d_model.

        Returns
        -------
        bound_st : SpectralTensor
            Coherence-weighted spectral embedding.
        coherence : torch.Tensor
            (batch, n_heads, seq, seq) coherence weights for diagnostics.
        """
        amp = st.amplitude   # (B, T, n_freq) or (B, n_freq)
        phase = st.phase

        # Ensure 3-D: (batch, seq_len, features)
        needs_squeeze = False
        if amp.dim() == 1:
            amp = amp.unsqueeze(0).unsqueeze(0)
            phase = phase.unsqueeze(0).unsqueeze(0)
            needs_squeeze = True
        elif amp.dim() == 2:
            amp = amp.unsqueeze(1)
            phase = phase.unsqueeze(1)
            needs_squeeze = True

        # Apply external projection if provided (S1→S2 bridge)
        if input_proj is not None:
            amp = input_proj(amp)
            phase = input_proj(phase)

        # Project to d_model using persistent projections
        amp = self.amp_proj(amp)
        phase = self.phase_proj(phase)

        # Resonance Attention
        bound, coherence = self.resonance(amp, phase=phase)

        if needs_squeeze:
            bound = bound.squeeze(0)

        # Package back into SpectralTensor
        # If amp was projected (n_freq → d_model), scale/uncertainty must
        # be resampled to match the new spectral dimension.
        orig_d = st.amplitude.shape[-1]
        new_d = bound.shape[-1]

        if orig_d != new_d:
            # Resample scale linearly along the feature dim
            scale_out = F.interpolate(
                st.scale.reshape(-1, 1, orig_d),
                size=new_d, mode="linear", align_corners=False,
            ).reshape(*st.scale.shape[:-1], new_d)

            # Resample uncertainty similarly
            unc_out = F.interpolate(
                st.uncertainty.reshape(-1, 1, orig_d),
                size=new_d, mode="linear", align_corners=False,
            ).reshape(*st.uncertainty.shape[:-1], new_d)
        else:
            scale_out = st.scale
            unc_out = st.uncertainty

        # coherence: (B, H, S, S) → scalar confidence per element in [0, 1]
        coh_score = coherence.mean(dim=1).mean(dim=-1)  # (B, S)
        if needs_squeeze:
            coh_score = coh_score.squeeze(0)  # (S,)
        # Expand to match uncertainty shape
        while coh_score.dim() < unc_out.dim():
            coh_score = coh_score.unsqueeze(-1)
        coh_score = coh_score.expand_as(unc_out).clamp(0, 1)

        bound_st = SpectralTensor(
            amplitude=bound.abs(),
            phase=torch.atan2(bound.sin(), bound.cos()),
            scale=scale_out,
            uncertainty=unc_out * (1.0 - coh_score),
            metadata={
                **st.metadata,
                "stage": "S2",
                "n_heads": self.resonance.n_heads,
                "n_bands": self.resonance.n_bands,
            },
        )

        return bound_st, coherence
