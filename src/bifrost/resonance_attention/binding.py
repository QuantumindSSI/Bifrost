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


class SpectralBinding(nn.Module):
    """
    Binding stage: SpectralTensor → bound spectral embedding + coherence map.

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
        # Phase is NOT projected - we compute coherence in original n_freq space
        # to preserve harmonic relationships (440Hz, 880Hz, etc. remain distinct)
        self.use_original_phase = n_freq_in is not None and n_freq_in != d_model

        if self.use_original_phase:
            # Separate resonance attention for original-phase coherence
            # Use n_heads=1 since 513 (n_freq) may not be divisible by n_heads
            # We only need coherence computation, not multi-head attention
            self.resonance_orig = ResonanceAttention(
                d_model=in_dim,  # n_freq dimension for phase coherence
                n_heads=1,  # Single head for coherence in original frequency space
                n_bands=min(n_bands, in_dim),  # can't have more bands than features
                dropout=0.0,  # No dropout for coherence computation
            )
            # Linear layer to map coherence from n_freq space to d_model for aggregation
            self.coherence_proj = nn.Linear(in_dim, d_model)

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
        canonical_phase: Optional[torch.Tensor] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Parameters
        ----------
        st : SpectralTensor
            Output of decomposition.  ``st.amplitude`` shape: ``(batch, channels, n_freq)``.
        input_proj : nn.Linear, optional
            External projection layer if n_freq != d_model.
        canonical_phase : torch.Tensor, optional
            Raw STFT phase from S0 Canonicaliser (B, n_freq) or (B, T, n_freq).
            When supplied, this is used for coherence computation instead of
            st.phase — it has passed through zero learned projections and
            cannot collapse to a constant, making it collapse-proof.

        Returns
        -------
        bound_st : SpectralTensor
            Coherence-weighted spectral embedding.
        coherence : torch.Tensor
            (batch, n_heads, seq, seq) coherence weights for diagnostics.
        """
        amp = st.amplitude   # (B, T, n_freq) or (B, n_freq)
        phase = st.phase
        scale = st.scale
        uncertainty = st.uncertainty

        # Ensure 3-D: (batch, seq_len, features)
        needs_squeeze = False
        if amp.dim() == 1:
            amp = amp.unsqueeze(0).unsqueeze(0)
            phase = phase.unsqueeze(0).unsqueeze(0)
            scale = scale.unsqueeze(0).unsqueeze(0)
            uncertainty = uncertainty.unsqueeze(0).unsqueeze(0)
            needs_squeeze = True
        elif amp.dim() == 2:
            amp = amp.unsqueeze(1)
            phase = phase.unsqueeze(1)
            scale = scale.unsqueeze(1)
            uncertainty = uncertainty.unsqueeze(1)
            needs_squeeze = True

        # Store original amplitude and phase before any projection (needed for harmonic coherence)
        amp_orig = amp
        # If canonical_phase (raw S0 STFT phase) is supplied, use it for coherence.
        # It has passed through ZERO learned parameters — it cannot collapse.
        # decomposed.phase passes through output_proj (a learned complex linear layer)
        # which collapses to a constant at equilibrium, making all phase differences → 0.
        # Resize to (B, T_amp, d_model) to match the amplitude tensor's temporal resolution.
        if canonical_phase is not None:
            cp = canonical_phase
            # Normalise to exactly 3D (B, T_s0, n_freq_s0) regardless of input dims.
            if cp.dim() == 2:
                cp = cp.unsqueeze(1)          # (B, 1, n_freq) → treat single frame as T=1
            elif cp.dim() == 4:
                cp = cp.mean(dim=1)           # (B, C, T, n_freq) → average channels → (B, T, n_freq)
            # cp is now (B, T_s0, n_freq_s0)
            T_target = amp.shape[1]
            F_target = self.d_model
            # Reshape to (B, 1, T_s0, F_s0) for F.interpolate which expects (N, C, H, W)
            cp_4d = cp.unsqueeze(1)  # (B, 1, T_s0, F_s0)
            cp_resized = torch.nn.functional.interpolate(
                cp_4d.float(),
                size=(T_target, F_target),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)  # (B, T_target, F_target)
            phase_orig = cp_resized
        else:
            phase_orig = phase

        # Apply external projection if provided (S1→S2 bridge)
        if input_proj is not None:
            amp = input_proj(amp)
            # When input_proj is provided, amp is already in d_model space — skip self.amp_proj
            amp_proj = amp
        else:
            # Project amplitude to d_model for values
            amp_proj = self.amp_proj(amp)

        B_sz, T_sz, _ = amp_proj.shape
        d_head = self.resonance.d_head
        n_heads = self.resonance.n_heads

        # Coherence computation strategy depends on whether canonical_phase is supplied.
        #
        # When canonical_phase IS available (normal training/inference via pipeline):
        #   Compute coherence directly from raw S0 STFT phase — zero learned parameters,
        #   architecturally collapse-proof. W_q/W_k/W_k cannot drive it to uniform.
        #   Gradients flow only through W_v and W_o (value routing), not coherence.
        #   The loss (var_real - var_noise) still has grad_fn through W_v: when the
        #   coherence matrix is peaked for real signals, the weighted-sum output of W_v
        #   differs from the noise case, so var_real carries grad via W_v → amp_proj.
        #
        # When canonical_phase is NOT available (standalone / legacy use):
        #   Fall through to resonance.forward with phase_orig (may collapse eventually
        #   but keeps the module usable independently of the pipeline).
        if canonical_phase is not None:
            # phase_orig already set to resized canonical_phase (B, T_target, d_model)
            # Compute coherence in (B, 1, T, T) using _phase_coherence with single band
            # — no learned parameters involved.
            ph = self.resonance._reshape_heads(phase_orig.float())  # (B, H, T, d_head)
            precomp_coh = self.resonance._phase_coherence(ph, ph)   # (B, H, T, T)
            bound, coherence = self.resonance(
                amp_proj,
                precomputed_coherence=precomp_coh,
            )
        else:
            # Legacy path: use phase_orig through the internal attention mechanism.
            if phase_orig.shape[-1] != self.d_model:
                phase_for_attn = torch.nn.functional.interpolate(
                    phase_orig.reshape(-1, 1, phase_orig.shape[-1]),
                    size=self.d_model,
                    mode='linear',
                    align_corners=False,
                ).reshape(B_sz, T_sz, self.d_model)
            else:
                phase_for_attn = phase_orig
            bound, coherence = self.resonance(amp_proj, phase=phase_for_attn)

        # Compute V projections for diagnostics / harmonic blend re-aggregation
        V_proj = self.resonance.W_v(amp_proj)  # (B, T, d_model)
        # (B, n_heads, T, d_head)
        V_heads = V_proj.view(B_sz, T_sz, n_heads, d_head).transpose(1, 2)
        # coherence: (B, n_heads, T, T) - returned for diagnostics

        # Compute coherence in ORIGINAL n_freq space to preserve harmonic structure
        if self.use_original_phase and phase_orig.shape[-1] != self.d_model:
            # Compute phase coherence in original frequency space (n_freq dimension)
            # This preserves harmonic relationships (440Hz, 880Hz, etc. are distinct)
            _, coherence_orig = self.resonance_orig(amp, phase=phase_orig)
            # coherence_orig: (B, 1, T, T) computed from full 513-dim phase

            # Broadcast original-phase coherence to all heads
            coherence_orig_expanded = coherence_orig.expand(-1, n_heads, -1, -1)

            # Blend: 70% harmonic-space coherence (structure) + 30% projected (learned)
            coherence = 0.7 * coherence_orig_expanded + 0.3 * coherence

            # Re-normalise blended weights and re-aggregate V so bound reflects harmonics
            # Clamp to prevent -inf from propagating through softmax (produces NaN)
            coherence = torch.clamp(coherence, min=-50, max=50)
            blended_weights = F.softmax(coherence, dim=-1)
            blended_weights = self.resonance.attn_dropout(blended_weights)
            # (B, n_heads, T, d_head)
            out_heads = torch.matmul(blended_weights, V_heads)
            # (B, T, d_model)
            out_flat = out_heads.transpose(1, 2).contiguous().view(B_sz, T_sz, self.d_model)
            bound = self.resonance.norm(self.resonance.W_o(out_flat) + amp_proj)

        if needs_squeeze:
            bound = bound.squeeze(0)

        # Package back into SpectralTensor
        # If amp was projected (n_freq → d_model), scale/uncertainty must
        # be resampled to match the new spectral dimension.
        orig_d = amp.shape[-1]  # use post-unsqueeze amp shape
        new_d = bound.shape[-1]

        if orig_d != new_d:
            # Resample scale linearly along the feature dim
            scale_out = F.interpolate(
                scale.reshape(-1, 1, orig_d),
                size=new_d, mode="linear", align_corners=False,
            ).reshape(*scale.shape[:-1], new_d)

            # Resample uncertainty similarly
            unc_out = F.interpolate(
                uncertainty.reshape(-1, 1, orig_d),
                size=new_d, mode="linear", align_corners=False,
            ).reshape(*uncertainty.shape[:-1], new_d)
        else:
            scale_out = scale
            unc_out = uncertainty

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
                "stage": "bind",
                "n_heads": self.resonance.n_heads,
                "n_bands": self.resonance.n_bands,
            },
        )
        bound_st.validate()

        return bound_st, coherence


S2SpectralBinding = SpectralBinding
