"""
Phase Ablation Harness — systematically destroys phase information to
measure its contribution to semantic task performance.

This module implements the ablation framework for Claim C1 of the
Structured Resonance Thesis: phase coherence captures semantic structure.

Each ablation targets a specific aspect of phase:
    - phase_zero:           set all phase to 0 (magnitude-only spectrogram)
    - phase_randomize:      shuffle phase across time (destroys temporal structure)
    - phase_noise:          add Gaussian noise to phase (degrades gradually)
    - phase_quantize:       reduce phase precision (measures precision sensitivity)
    - cross_band_scramble:  scramble phase between frequency bands
                            (directly tests CBMPC's cross-band hypothesis)
    - cross_scale_scramble: scramble phase between analysis scales
                            (tests cross-scale coherence hypothesis)

All ablations preserve amplitude — only phase is modified.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..spectral_tensor import SpectralTensor


class PhaseAblationHarness(nn.Module):
    """
    Systematically destroys phase information to measure its contribution
    to semantic task performance.

    Usage:
        harness = PhaseAblationHarness()
        ablated = harness.phase_zero(spectral_tensor)
        # run classifier on ablated tensor, compare with baseline
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = seed

    def phase_zero(self, st: SpectralTensor) -> SpectralTensor:
        """Set all phase to zero. Equivalent to magnitude-only spectrogram.
        This is what most audio ML systems actually use."""
        return SpectralTensor(
            amplitude=st.amplitude.clone(),
            phase=torch.zeros_like(st.phase),
            scale=st.scale.clone(),
            uncertainty=st.uncertainty.clone(),
            metadata={**st.metadata, "ablation": "phase_zero"},
        )

    def phase_randomize(self, st: SpectralTensor) -> SpectralTensor:
        """Shuffle phase across the time axis. Preserves amplitude distribution
        and per-frame phase values, but destroys temporal phase relationships."""
        g = torch.Generator(device=st.phase.device).manual_seed(self.seed)
        phase = st.phase.clone()
        # Shuffle along time axis (dim=-2 assumed to be time)
        if phase.dim() >= 2:
            time_dim = -2
            perm = torch.randperm(phase.shape[time_dim], generator=g,
                                  device=phase.device)
            phase = phase.index_select(time_dim, perm)
        return SpectralTensor(
            amplitude=st.amplitude.clone(),
            phase=phase,
            scale=st.scale.clone(),
            uncertainty=st.uncertainty.clone(),
            metadata={**st.metadata, "ablation": "phase_randomize"},
        )

    def phase_noise(self, st: SpectralTensor, sigma: float = 0.5) -> SpectralTensor:
        """Add Gaussian noise to phase: phi' = phi + N(0, sigma).
        Degrades phase gradually to measure phase precision sensitivity."""
        g = torch.Generator(device=st.phase.device).manual_seed(self.seed)
        noise = torch.randn(st.phase.shape, generator=g,
                            device=st.phase.device,
                            dtype=st.phase.dtype) * sigma
        phase = st.phase + noise
        # Wrap to [-pi, pi]
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
        return SpectralTensor(
            amplitude=st.amplitude.clone(),
            phase=phase,
            scale=st.scale.clone(),
            uncertainty=st.uncertainty.clone(),
            metadata={**st.metadata, "ablation": f"phase_noise_{sigma}"},
        )

    def phase_quantize(self, st: SpectralTensor, n_levels: int = 4) -> SpectralTensor:
        """Quantize phase to n_levels. Measures how much phase precision
        is needed to preserve semantic structure."""
        step = 2 * torch.pi / n_levels
        phase = torch.round(st.phase / step) * step
        # Wrap to [-pi, pi]
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
        return SpectralTensor(
            amplitude=st.amplitude.clone(),
            phase=phase,
            scale=st.scale.clone(),
            uncertainty=st.uncertainty.clone(),
            metadata={**st.metadata, "ablation": f"phase_quantize_{n_levels}"},
        )

    def cross_band_scramble(self, st: SpectralTensor) -> SpectralTensor:
        """Scramble phase relationships between frequency bands while
        preserving within-band phase. Directly tests CBMPC's hypothesis:
        if cross-band phase relationships carry semantic structure,
        scrambling them should destroy it."""
        g = torch.Generator(device=st.phase.device).manual_seed(self.seed)
        phase = st.phase.clone()
        # Scramble band order (last dim assumed to be frequency bands)
        if phase.dim() >= 2:
            band_dim = -1
            perm = torch.randperm(phase.shape[band_dim], generator=g,
                                  device=phase.device)
            phase = phase.index_select(band_dim, perm)
        return SpectralTensor(
            amplitude=st.amplitude.clone(),
            phase=phase,
            scale=st.scale.clone(),
            uncertainty=st.uncertainty.clone(),
            metadata={**st.metadata, "ablation": "cross_band_scramble"},
        )

    def cross_scale_scramble(
        self,
        multi_scale_st: list[SpectralTensor],
    ) -> list[SpectralTensor]:
        """Scramble phase between analysis scales while preserving
        within-scale phase. Tests cross-scale coherence hypothesis.

        Adds an independent random phase offset to each scale, breaking
        cross-scale phase relationships while preserving within-scale structure.
        """
        g = torch.Generator(device=multi_scale_st[0].phase.device
                            ).manual_seed(self.seed)
        result = []
        for i, st in enumerate(multi_scale_st):
            # Each scale gets a different random phase offset
            offset = torch.rand(1, generator=g,
                                device=st.phase.device,
                                dtype=st.phase.dtype) * 2 * torch.pi - torch.pi
            phase = st.phase + offset
            phase = torch.atan2(torch.sin(phase), torch.cos(phase))
            result.append(SpectralTensor(
                amplitude=st.amplitude.clone(),
                phase=phase,
                scale=st.scale.clone(),
                uncertainty=st.uncertainty.clone(),
                metadata={**st.metadata, "ablation": f"cross_scale_scramble_{i}"},
            ))
        return result

    def run_all_ablations(
        self,
        st: SpectralTensor,
        feature_extractor: nn.Module,
        classifier: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Run all single-tensor ablations and return logits for each.

        Parameters
        ----------
        st : SpectralTensor
            The input spectral tensor.
        feature_extractor : nn.Module
            Feature extractor that takes SpectralTensor and returns features.
        classifier : nn.Module
            Classifier that takes features and returns logits.

        Returns
        -------
        dict[str, torch.Tensor]
            Logits for each ablation condition.
        """
        results = {}

        # Baseline (no ablation)
        features = feature_extractor(st)
        results["baseline"] = classifier(features)

        # Phase zero
        ablated = self.phase_zero(st)
        features = feature_extractor(ablated)
        results["phase_zero"] = classifier(features)

        # Phase randomize
        ablated = self.phase_randomize(st)
        features = feature_extractor(ablated)
        results["phase_randomize"] = classifier(features)

        # Phase noise (moderate)
        ablated = self.phase_noise(st, sigma=0.5)
        features = feature_extractor(ablated)
        results["phase_noise_0.5"] = classifier(features)

        # Phase noise (severe)
        ablated = self.phase_noise(st, sigma=2.0)
        features = feature_extractor(ablated)
        results["phase_noise_2.0"] = classifier(features)

        # Phase quantize (coarse)
        ablated = self.phase_quantize(st, n_levels=4)
        features = feature_extractor(ablated)
        results["phase_quantize_4"] = classifier(features)

        # Cross-band scramble
        ablated = self.cross_band_scramble(st)
        features = feature_extractor(ablated)
        results["cross_band_scramble"] = classifier(features)

        return results
