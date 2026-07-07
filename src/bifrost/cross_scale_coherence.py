"""
Cross-Scale Coherence — computes phase coherence between different
analysis scales.

This is the generalization of CBMPC from cross-band (across frequency
bands at one scale) to cross-scale (across analysis scales).

For audio: measures whether modulation phase at scale s1 (e.g., 2Hz)
is coherent with modulation phase at scale s2 (e.g., 4Hz).
For images: measures whether phase congruency at fine scale is
coherent with phase congruency at coarse scale.

The module produces two types of features:
    1. PLV between scale pairs — are phases at different scales locked?
    2. Harmonic deviation — if scale s2 = 2 * scale s1, does phase at s2
       follow the harmonic relationship? (wavelet analog of harmonic binding)

References:
    - Grinsted et al. (2004): wavelet coherence
    - Kovesi (1999): phase congruency across scales
    - Bruna & Mallat (2013): wavelet scattering across scales
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleCoherence(nn.Module):
    """Computes phase coherence between different analysis scales.

    Parameters
    ----------
    n_scales : int
        Number of analysis scales.
    dyadic : bool
        If True, scales are assumed to be dyadic (s_{i+1} = 2 * s_i).
        Enables harmonic deviation computation.
    """

    def __init__(self, n_scales: int = 6, dyadic: bool = True) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.dyadic = dyadic

        # Scale pairs: (i, j) where i < j
        self.scale_pairs = [(i, j) for i in range(n_scales)
                            for j in range(i + 1, n_scales)]
        self.n_pairs = len(self.scale_pairs)

        # Feature dimension:
        # - PLV per scale pair: n_pairs
        # - Harmonic deviation per scale pair: n_pairs (if dyadic)
        # - Mean cross-scale coherence: 1
        # - Max cross-scale coherence: 1
        self.feature_dim = self.n_pairs * (2 if dyadic else 1) + 2

    def forward(
        self,
        multi_scale_phases: List[torch.Tensor],
        multi_scale_amplitudes: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute cross-scale coherence features.

        Parameters
        ----------
        multi_scale_phases : List[torch.Tensor]
            Phase tensors at each scale. Each tensor has shape (B, ...).
            Expected: [phase_s0, phase_s1, ..., phase_sn]
        multi_scale_amplitudes : List[torch.Tensor], optional
            Amplitude tensors at each scale. If provided, used for
            amplitude-weighted PLV. If None, unweighted PLV is used.

        Returns
        -------
        torch.Tensor
            Cross-scale coherence feature vector. Shape (B, feature_dim)
        """
        assert len(multi_scale_phases) == self.n_scales, \
            f"Expected {self.n_scales} scales, got {len(multi_scale_phases)}"

        # Determine batch size
        B = multi_scale_phases[0].shape[0]

        # Compute per-sample PLV for each scale pair
        plv_per_sample = []  # each: (B,)
        harmonic_per_sample = []  # each: (B,)

        for (i, j) in self.scale_pairs:
            phase_i = multi_scale_phases[i]  # (B, ...)
            phase_j = multi_scale_phases[j]  # (B, ...)

            # Flatten spatial dims, keep batch
            phase_i_flat = phase_i.reshape(B, -1)  # (B, N)
            phase_j_flat = phase_j.reshape(B, -1)  # (B, N)

            # PLV per sample: |mean_N exp(i*(phase_i - phase_j))|
            diff = phase_i_flat - phase_j_flat  # (B, N)
            plv = torch.abs(torch.mean(torch.exp(1j * diff), dim=-1)).real  # (B,)
            plv_per_sample.append(plv)

            if self.dyadic:
                ratio = 2 ** (j - i)
                expected_phase = phase_i_flat * ratio
                expected_phase = torch.atan2(
                    torch.sin(expected_phase), torch.cos(expected_phase)
                )
                deviation = torch.angle(
                    torch.exp(1j * (phase_j_flat - expected_phase))
                )  # (B, N)
                harmonic_per_sample.append(deviation.abs().mean(dim=-1))  # (B,)

        # Stack: (B, n_pairs)
        plv_tensor = torch.stack(plv_per_sample, dim=-1)  # (B, n_pairs)
        features = [plv_tensor]

        if self.dyadic:
            harmonic_tensor = torch.stack(harmonic_per_sample, dim=-1)  # (B, n_pairs)
            features.append(harmonic_tensor)

        # Summary statistics per sample
        features.append(plv_tensor.mean(dim=-1, keepdim=True))  # (B, 1)
        features.append(plv_tensor.max(dim=-1, keepdim=True).values)  # (B, 1)

        return torch.cat(features, dim=-1)  # (B, feature_dim)

    def _plv(self, phase_a: torch.Tensor, phase_b: torch.Tensor) -> torch.Tensor:
        """Phase Locking Value between two phase tensors."""
        diff = phase_a - phase_b
        return torch.abs(torch.mean(torch.exp(1j * diff))).real

    def _weighted_plv(
        self,
        phase_a: torch.Tensor,
        phase_b: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Amplitude-weighted PLV."""
        diff = phase_a - phase_b
        weighted = weight * torch.exp(1j * diff)
        return torch.abs(weighted.sum() / (weight.sum() + 1e-8)).real

    def extra_repr(self) -> str:
        return (f"n_scales={self.n_scales}, dyadic={self.dyadic}, "
                f"n_pairs={self.n_pairs}, feature_dim={self.feature_dim}")
