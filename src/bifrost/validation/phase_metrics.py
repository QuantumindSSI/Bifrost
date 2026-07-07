"""
Phase Coherence Metrics — pure signal processing metrics computable at
any pipeline layer to measure phase coherence.

These metrics are NOT learned — they are deterministic functions of phase.
They can be computed before, during, and after processing to track how
much phase coherence exists at each stage.

Metrics:
    - phase_locking_value:    PLV = |mean(exp(i * (phase_a - phase_b)))|
    - phase_entropy:          Shannon entropy of phase distribution
    - phase_congruency:       Kovesi (1999) phase congruency across scales
    - cross_frequency_coupling: phase-amplitude coupling between bands
    - phase_stability:        temporal stability of phase

References:
    - Kovesi (1999): phase congruency as image feature
    - Lachaux et al. (1999): phase locking value measurement
    - Hyafil (2015): cross-frequency coupling mechanisms
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PhaseCoherenceSignalMetrics(nn.Module):
    """Phase coherence metrics computable at any pipeline layer.

    All methods are differentiable and can be used in loss functions.
    All methods operate on phase tensors (radians, [-pi, pi]).

    Note: This is distinct from complex_training.PhaseCoherenceMetrics
    which measures SSM-specific coherence. This module measures signal-level
    phase coherence using pure signal processing.
    """

    def phase_locking_value(
        self,
        phases_a: torch.Tensor,
        phases_b: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Phase Locking Value (PLV).

        PLV = |mean(exp(i * (phase_a - phase_b)))|

        Range: [0, 1]. 1 = perfect phase locking, 0 = no locking.

        Parameters
        ----------
        phases_a, phases_b : torch.Tensor
            Phase tensors of the same shape.
        dim : int
            Dimension along which to compute the mean (default: last).

        Returns
        -------
        torch.Tensor
            PLV values with dim reduced.
        """
        diff = phases_a - phases_b
        return torch.abs(torch.mean(torch.exp(1j * diff), dim=dim)).real

    def weighted_plv(
        self,
        phases_a: torch.Tensor,
        phases_b: torch.Tensor,
        weight: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Amplitude-weighted Phase Locking Value.

        PLV_w = |sum(w * exp(i * (phase_a - phase_b)))| / sum(w)

        High-amplitude phase relationships contribute more.
        """
        diff = phases_a - phases_b
        weighted = weight * torch.exp(1j * diff)
        return torch.abs(weighted.sum(dim=dim) / (weight.sum(dim=dim) + 1e-8)).real

    def phase_entropy(self, phases: torch.Tensor, n_bins: int = 32,
                      dim: int = -1) -> torch.Tensor:
        """Shannon entropy of phase distribution.

        Low entropy = concentrated phases = high coherence.
        High entropy = dispersed phases = low coherence.

        Range: [0, log(n_bins)]. Normalized to [0, 1] by dividing by log(n_bins).
        """
        # Histogram of phases
        phases_flat = phases.transpose(dim, -1).reshape(-1, phases.shape[dim])
        # Use torch.histc per sample
        n_samples = phases_flat.shape[0] if phases_flat.dim() > 1 else 1
        entropies = []
        for i in range(phases_flat.shape[0] if phases_flat.dim() > 1 else 1):
            if phases_flat.dim() > 1:
                p = phases_flat[i]
            else:
                p = phases_flat
            hist = torch.histc(p, bins=n_bins, min=-torch.pi, max=torch.pi)
            hist = hist / (hist.sum() + 1e-8)
            # Shannon entropy: -sum(p * log(p))
            entropy = -(hist * torch.log(hist + 1e-8)).sum()
            # Normalize by max entropy
            entropies.append(entropy / torch.log(torch.tensor(float(n_bins))))
        return torch.stack(entropies) if entropies else torch.tensor(0.0)

    def phase_congruency(
        self,
        multi_scale_phases: torch.Tensor,
        amplitudes: torch.Tensor,
        scale_dim: int = 1,
    ) -> torch.Tensor:
        """Kovesi (1999) phase congruency across scales.

        PC(x) = |sum_s A_s cos(phi_s - phi_bar)| / sum_s A_s

        where phi_bar is the amplitude-weighted mean phase.

        Parameters
        ----------
        multi_scale_phases : torch.Tensor
            Phases at multiple scales. Shape (..., n_scales, ...).
        amplitudes : torch.Tensor
            Amplitudes at multiple scales. Same shape as multi_scale_phases.
        scale_dim : int
            Dimension indexing scales.

        Returns
        -------
        torch.Tensor
            Phase congruency at each spatial/temporal location.
        """
        # Weighted mean phase: phi_bar = atan2(sum A sin(phi), sum A cos(phi))
        sum_sin = (amplitudes * torch.sin(multi_scale_phases)).sum(dim=scale_dim)
        sum_cos = (amplitudes * torch.cos(multi_scale_phases)).sum(dim=scale_dim)
        mean_phase = torch.atan2(sum_sin, sum_cos)

        # Phase deviation from mean
        phase_diff = multi_scale_phases - mean_phase.unsqueeze(scale_dim)

        # Energy: |sum A cos(phi - phi_bar)|
        energy = (amplitudes * torch.cos(phase_diff)).sum(dim=scale_dim)
        total_amplitude = amplitudes.sum(dim=scale_dim) + 1e-8

        return energy.abs() / total_amplitude

    def cross_frequency_coupling(
        self,
        phases: torch.Tensor,
        low_freq_idx: list[int],
        high_freq_idx: list[int],
        freq_dim: int = -1,
    ) -> torch.Tensor:
        """Phase-amplitude coupling between low and high frequency bands.

        Measures whether high-frequency phase is locked to low-frequency phase.
        This is the theta-gamma coupling mechanism from neuroscience.

        Computes mean PLV between each low-freq band and the mean phase
        of all high-freq bands.

        Parameters
        ----------
        phases : torch.Tensor
            Phase tensor with frequency bands along freq_dim.
        low_freq_idx : list of int
            Indices of low-frequency bands.
        high_freq_idx : list of int
            Indices of high-frequency bands.
        freq_dim : int
            Dimension indexing frequency bands.

        Returns
        -------
        torch.Tensor
            Mean cross-frequency coupling value.
        """
        # Mean phase of high-frequency bands
        high_phases = phases.index_select(freq_dim, torch.tensor(high_freq_idx,
                                                                  device=phases.device))
        # Compute mean phase via circular mean
        high_sum_sin = torch.sin(high_phases).sum(dim=freq_dim)
        high_sum_cos = torch.cos(high_phases).sum(dim=freq_dim)
        high_mean_phase = torch.atan2(high_sum_sin, high_sum_cos)

        # PLV between each low-freq band and high-freq mean phase
        coupling_values = []
        for lf_idx in low_freq_idx:
            low_phase = phases.index_select(freq_dim, torch.tensor(lf_idx,
                                                                    device=phases.device))
            # Squeeze the selected dimension
            low_phase = low_phase.squeeze(freq_dim if freq_dim >= 0 else freq_dim + 1)
            plv = self.phase_locking_value(low_phase, high_mean_phase, dim=0)
            coupling_values.append(plv)

        return torch.stack(coupling_values).mean()

    def phase_stability(self, phases_over_time: torch.Tensor,
                        time_dim: int = -2) -> torch.Tensor:
        """Temporal stability of phase.

        stability = 1 - var(phase) / (2*pi)

        Stable phase = consistent semantic structure over time.
        Unstable phase = changing structure.

        Uses circular variance for proper phase statistics.

        Parameters
        ----------
        phases_over_time : torch.Tensor
            Phase tensor with time along time_dim.
        time_dim : int
            Dimension indexing time.

        Returns
        -------
        torch.Tensor
            Phase stability in [0, 1].
        """
        # Circular variance: R = |mean(exp(i*phase))|
        # var = 1 - R^2 (range [0, 1])
        # stability = 1 - var = R^2
        R = torch.abs(torch.mean(torch.exp(1j * phases_over_time), dim=time_dim))
        return R ** 2

    def coherence_profile(self, st_phases: torch.Tensor,
                          freq_dim: int = -1) -> torch.Tensor:
        """Compute a comprehensive coherence profile from phase tensor.

        Returns a feature vector capturing multiple aspects of phase coherence:
        - Mean PLV across all band pairs
        - Phase entropy
        - Phase stability (if temporal dimension exists)
        - Cross-frequency coupling (low vs high bands)

        Parameters
        ----------
        st_phases : torch.Tensor
            Phase tensor from SpectralTensor.
        freq_dim : int
            Dimension indexing frequency bands.

        Returns
        -------
        torch.Tensor
            Coherence profile feature vector.
        """
        n_bands = st_phases.shape[freq_dim]
        features = []

        # 1. Mean pairwise PLV across all band pairs
        plv_sum = 0.0
        count = 0
        for i in range(n_bands):
            for j in range(i + 1, n_bands):
                phase_i = st_phases.index_select(freq_dim, torch.tensor(i,
                                                    device=st_phases.device))
                phase_j = st_phases.index_select(freq_dim, torch.tensor(j,
                                                    device=st_phases.device))
                # Average over non-frequency dimensions
                phase_i_flat = phase_i.reshape(-1)
                phase_j_flat = phase_j.reshape(-1)
                plv = self.phase_locking_value(phase_i_flat, phase_j_flat, dim=0)
                plv_sum = plv_sum + plv
                count += 1
        mean_plv = plv_sum / max(count, 1)
        features.append(mean_plv.unsqueeze(0))

        # 2. Phase entropy (averaged over bands)
        entropy = self.phase_entropy(st_phases, dim=freq_dim)
        features.append(entropy.mean().unsqueeze(0))

        # 3. Phase stability (if temporal dimension exists)
        if st_phases.dim() >= 2 and time_dim != freq_dim:
            stability = self.phase_stability(st_phases, time_dim=-2 if freq_dim == -1 else -1)
            features.append(stability.mean().unsqueeze(0))

        # 4. Cross-frequency coupling (first third vs last third of bands)
        n_low = n_bands // 3
        n_high_start = (2 * n_bands) // 3
        low_idx = list(range(n_low))
        high_idx = list(range(n_high_start, n_bands))
        if low_idx and high_idx:
            cfc = self.cross_frequency_coupling(st_phases, low_idx, high_idx,
                                                 freq_dim=freq_dim)
            features.append(cfc.unsqueeze(0))

        return torch.cat([f.reshape(-1) for f in features])
