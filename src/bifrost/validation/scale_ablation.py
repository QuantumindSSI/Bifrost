"""
Scale Ablation Harness — tests whether cross-scale coherence matters
beyond single-scale processing.

This module implements the ablation framework for Claim C2 of the
Structured Resonance Thesis: multi-scale coherence is necessary.

Ablations:
    - single_scale:        use only one scale, discard all others
    - scale_subset:        use k randomly chosen scales
    - scale_shuffle:       shuffle scale assignments (same scales, wrong labels)
    - cross_scale_destroy: keep all scales but break cross-scale phase
                            relationships (add independent random phase offset
                            per scale)

The critical ablation is cross_scale_destroy: if performance drops when
cross-scale relationships are broken but individual scales are preserved,
it proves that the *relationships between scales* carry semantic structure.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..utils.spectral_utils import wrap_phase


class ScaleAblationHarness(nn.Module):
    """Tests whether cross-scale coherence matters beyond single-scale.

    Parameters
    ----------
    n_scales : int
        Number of analysis scales.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_scales: int = 6, seed: int = 42) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.seed = seed

    def single_scale(
        self,
        multi_scale_phases: List[torch.Tensor],
        multi_scale_amplitudes: List[torch.Tensor],
        scale_idx: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Use only one scale. Discard all others.

        Parameters
        ----------
        scale_idx : int
            Index of the scale to keep.

        Returns
        -------
        (phases, amplitudes)
            Lists containing only the selected scale.
        """
        return (
            [multi_scale_phases[scale_idx]],
            [multi_scale_amplitudes[scale_idx]],
        )

    def best_single_scale(
        self,
        multi_scale_phases: List[torch.Tensor],
        multi_scale_amplitudes: List[torch.Tensor],
        feature_extractor: nn.Module,
        classifier: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        """Find and use the best single scale by classification accuracy.

        Runs the classifier with each scale individually and selects
        the one with highest accuracy.
        """
        best_acc = -1.0
        best_idx = 0
        for s in range(self.n_scales):
            phases = [multi_scale_phases[s]]
            amps = [multi_scale_amplitudes[s]]
            features = feature_extractor(phases, amps)
            logits = classifier(features)
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_idx = s
        return self.single_scale(
            multi_scale_phases, multi_scale_amplitudes, best_idx
        ) + (best_idx,)

    def scale_subset(
        self,
        multi_scale_phases: List[torch.Tensor],
        multi_scale_amplitudes: List[torch.Tensor],
        k: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Use k randomly chosen scales.

        Parameters
        ----------
        k : int
            Number of scales to keep.
        """
        g = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(self.n_scales, generator=g)[:k].tolist()
        return (
            [multi_scale_phases[i] for i in indices],
            [multi_scale_amplitudes[i] for i in indices],
        )

    def scale_shuffle(
        self,
        multi_scale_phases: List[torch.Tensor],
        multi_scale_amplitudes: List[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Shuffle scale assignments. Same scales, wrong scale labels.

        This tests whether the scale ordering matters. If the system
        relies on knowing which scale is fine vs coarse, shuffling
        should degrade performance.
        """
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(self.n_scales, generator=g).tolist()
        return (
            [multi_scale_phases[i] for i in perm],
            [multi_scale_amplitudes[i] for i in perm],
        )

    def cross_scale_destroy(
        self,
        multi_scale_phases: List[torch.Tensor],
        multi_scale_amplitudes: List[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Keep all scales but break cross-scale phase relationships.

        Adds independent random phase noise to each time point in each scale,
        breaking the temporal alignment of phases across scales. This is
        the critical ablation: if performance drops here but not with
        single_scale, it proves cross-scale *relationships* matter.

        Note: PLV is invariant to constant phase offsets (|exp(i*offset)|=1),
        so we must add per-element noise, not per-scale offsets.
        """
        g = torch.Generator(device=multi_scale_phases[0].device
                            ).manual_seed(self.seed)
        shuffled_phases = []
        for i in range(self.n_scales):
            # Add independent random noise to each element
            noise = torch.randn(multi_scale_phases[i].shape,
                                generator=g,
                                device=multi_scale_phases[i].device,
                                dtype=multi_scale_phases[i].dtype) * torch.pi
            phase = multi_scale_phases[i] + noise
            phase = wrap_phase(phase)
            shuffled_phases.append(phase)
        return shuffled_phases, multi_scale_amplitudes

    def run_all_ablations(
        self,
        multi_scale_phases: List[torch.Tensor],
        multi_scale_amplitudes: List[torch.Tensor],
        feature_extractor: nn.Module,
        classifier: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Run all scale ablations and return logits for each.

        Parameters
        ----------
        multi_scale_phases, multi_scale_amplitudes : List[torch.Tensor]
            Multi-scale phase and amplitude tensors.
        feature_extractor : nn.Module
            Takes (phases, amplitudes) lists and returns features.
        classifier : nn.Module
            Takes features and returns logits.

        Returns
        -------
        dict[str, torch.Tensor]
            Logits for each ablation condition.
        """
        results = {}

        # Baseline (all scales, no ablation)
        features = feature_extractor(multi_scale_phases, multi_scale_amplitudes)
        results["baseline"] = classifier(features)

        # Single scale (each scale individually)
        for s in range(min(self.n_scales, 3)):  # test first 3 scales
            phases, amps = self.single_scale(
                multi_scale_phases, multi_scale_amplitudes, s
            )
            features = feature_extractor(phases, amps)
            results[f"single_scale_{s}"] = classifier(features)

        # Scale subset (half the scales)
        phases, amps = self.scale_subset(
            multi_scale_phases, multi_scale_amplitudes,
            k=max(1, self.n_scales // 2)
        )
        features = feature_extractor(phases, amps)
        results["scale_subset_half"] = classifier(features)

        # Scale shuffle
        phases, amps = self.scale_shuffle(
            multi_scale_phases, multi_scale_amplitudes
        )
        features = feature_extractor(phases, amps)
        results["scale_shuffle"] = classifier(features)

        # Cross-scale destroy (the critical ablation)
        phases, amps = self.cross_scale_destroy(
            multi_scale_phases, multi_scale_amplitudes
        )
        features = feature_extractor(phases, amps)
        results["cross_scale_destroy"] = classifier(features)

        return results
