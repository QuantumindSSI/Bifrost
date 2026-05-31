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
from .harmonic_coherence import HarmonicCoherenceDetector


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
        harmonic_blend_ratio: float = 0.7,  # Empirically validated default
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_freq_in = n_freq_in  # if set, adds a persistent input projection
        # Harmonic coherence blend ratio - validated empirically on 440+880+1320Hz signals
        # Range [0.0, 1.0]: 0.0 = all learned coherence, 1.0 = all harmonic coherence
        # See scripts/validate_blend_ratio.py for validation methodology
        self.harmonic_blend_ratio = harmonic_blend_ratio

        # Persistent input projection when n_freq_in != d_model
        in_dim = n_freq_in if n_freq_in is not None else d_model
        self.amp_proj = nn.Linear(in_dim, d_model)
        # Phase is NOT projected - we compute coherence in original n_freq space
        # to preserve harmonic relationships (440Hz, 880Hz, etc. remain distinct)
        self.use_original_phase = n_freq_in is not None and n_freq_in != d_model

        if self.use_original_phase:
            # Harmonic coherence detector: measures energy at harmonic frequency bins
            # This distinguishes harmonic signals (440Hz, 880Hz, 1320Hz, etc.)
            # from inharmonic signals (energy spread across non-harmonic frequencies)
            self.harmonic_coherence = HarmonicCoherenceDetector(
                n_freq=in_dim,
                n_harmonics=5,  # f, 2f, 3f, 4f, 5f
                base_freq=None,  # Auto-detect from amplitude spectrum
                sample_rate=16000.0,  # Default sample rate
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
        canonical_amplitude: Optional[torch.Tensor] = None,
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
        canonical_amplitude : torch.Tensor, optional
            Raw STFT amplitude from S0 Canonicaliser (B, n_freq) or (B, T, n_freq).
            Used for harmonic coherence detection to preserve harmonic structure.
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
        #
        # CRITICAL: For harmonic coherence preservation, we must compute coherence in the
        # ORIGINAL frequency space (n_freq), NOT in the projected d_model space.
        # Interpolating phase from n_freq to d_model destroys harmonic frequency relationships
        # (440Hz, 880Hz, etc. no longer align with distinct bins).
        #
        # Solution: Keep canonical_phase at original n_freq dimensions for coherence_orig,
        # only interpolate if needed for the legacy path (when use_original_phase is False).
        if canonical_phase is not None:
            cp = canonical_phase
            # Normalise to exactly 3D (B, T_s0, n_freq_s0) regardless of input dims.
            if cp.dim() == 2:
                cp = cp.unsqueeze(1)          # (B, 1, n_freq) → treat single frame as T=1
            elif cp.dim() == 4:
                cp = cp.mean(dim=1)           # (B, C, T, n_freq) → average channels → (B, T, n_freq)
            # cp is now (B, T_s0, n_freq_s0)

            # CRITICAL: phase_orig must match amp's temporal dimension (T) for matmul
            # but keep n_freq dimensions for harmonic coherence.
            # Interpolate temporally (T_s0 → T_amp) but keep frequency dimension n_freq.
            T_target = amp.shape[1]  # Target temporal dimension from amp
            if cp.shape[1] != T_target:
                # Temporal interpolation: (B, T_s0, n_freq) → (B, T_target, n_freq)
                # Using linear interpolation along time axis
                cp_f = cp.float()
                # Reshape for interpolate: (B, C=1, T_s0, n_freq)
                cp_reshaped = cp_f.unsqueeze(1)  # (B, 1, T_s0, n_freq)
                cp_interp = F.interpolate(
                    cp_reshaped, size=(T_target, cp.shape[2]),
                    mode='bilinear', align_corners=False
                )  # (B, 1, T_target, n_freq)
                phase_orig = cp_interp.squeeze(1)  # (B, T_target, n_freq)
            else:
                phase_orig = cp.float()

            # For learned coherence path, interpolate BOTH temporal AND frequency dimensions
            if cp.shape[-1] != self.d_model or cp.shape[1] != T_target:
                F_target = self.d_model
                cp_f = cp.float()
                # Phase is circular (wraps at ±π). Naive linear interpolation destroys
                # variance at wrap boundaries (e.g. interpolating +π and -π gives 0).
                # Correct approach: interpolate cos(φ) and sin(φ) (both smooth, bounded
                # in [-1,1]), then recover angle via atan2. Preserves circular structure.
                cos_cp = torch.cos(cp_f).unsqueeze(1)  # (B, 1, T_s0, F_s0)
                sin_cp = torch.sin(cp_f).unsqueeze(1)
                cos_r = F.interpolate(
                    cos_cp, size=(T_target, F_target), mode='bilinear', align_corners=False,
                ).squeeze(1)  # (B, T_target, F_target)
                sin_r = F.interpolate(
                    sin_cp, size=(T_target, F_target), mode='bilinear', align_corners=False,
                ).squeeze(1)
                phase = torch.atan2(sin_r, cos_r)  # (B, T_target, F_target) - for learned path
            else:
                phase = phase_orig
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
            # Use interpolated phase (d_model dims) for learned coherence computation
            # phase_orig (n_freq dims) is reserved for harmonic coherence computation only
            # Compute coherence in (B, 1, T, T) using _phase_coherence with single band
            # — no learned parameters involved.
            ph = self.resonance._reshape_heads(phase.float())  # (B, H, T, d_head)
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
            # Compute harmonic coherence: energy concentration at harmonic frequency bins
            # This distinguishes harmonic signals (440Hz, 880Hz, 1320Hz, etc.)
            # from inharmonic signals (energy spread across non-harmonic frequencies)
            # CRITICAL: Use canonical_amplitude (raw STFT amplitude from S0), not decomposed amplitude
            # The decomposed amplitude has been transformed by S1 and may not preserve harmonic structure
            if canonical_amplitude is not None:
                # Use raw STFT amplitude for harmonic detection
                amp_for_harmonic = canonical_amplitude
                # Temporally interpolate to match phase_orig dimensions if needed
                if amp_for_harmonic.shape[1] != phase_orig.shape[1]:
                    amp_for_harmonic = F.interpolate(
                        amp_for_harmonic.unsqueeze(1), size=(phase_orig.shape[1], amp_for_harmonic.shape[-1]),
                        mode='bilinear', align_corners=False
                    ).squeeze(1)
            else:
                # Fallback to amp_orig if canonical_amplitude not available
                amp_for_harmonic = amp_orig
            _, coherence_orig = self.harmonic_coherence(amp_for_harmonic, phase=phase_orig, n_fft=512)
            # coherence_orig: (B, 1, T, T) computed from harmonic energy concentration

            # Broadcast original-phase coherence to all heads
            coherence_orig_expanded = coherence_orig.expand(-1, n_heads, -1, -1)

            # Blend: harmonic_blend_ratio% harmonic-space coherence + (1-ratio)% projected
            # This preserves harmonic structure while allowing learned adaptation
            # Ratio is empirically validated on ground-truth harmonic signals
            h_ratio = self.harmonic_blend_ratio
            # DEBUG: Print coherence stats to verify blend is working
            print(f"[DEBUG] blend_ratio={h_ratio:.2f}, coherence_orig_mean={coherence_orig.mean():.4f}, "
                  f"coherence_learned_mean={coherence.mean():.4f}, "
                  f"coherence_orig_std={coherence_orig.std():.4f}, "
                  f"coherence_learned_std={coherence.std():.4f}")
            coherence = h_ratio * coherence_orig_expanded + (1.0 - h_ratio) * coherence

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
        # CRITICAL: Capture original dimension BEFORE any projection happened.
        # st.shape[-1] is the canonical S0/S1 frequency dimension (n_freq).
        orig_d = st.shape[-1]  # Original frequency dimension from input SpectralTensor
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
