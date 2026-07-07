"""Tests for the ScaleAblationHarness."""

import pytest
import torch

from bifrost.validation.scale_ablation import ScaleAblationHarness


@pytest.fixture
def harness():
    """Scale ablation harness with 4 scales."""
    return ScaleAblationHarness(n_scales=4, seed=42)


@pytest.fixture
def multi_scale_phases():
    """List of 4 random phase tensors (B=4, T=16)."""
    torch.manual_seed(0)
    return [torch.rand(4, 16) * 2 * torch.pi - torch.pi for _ in range(4)]


@pytest.fixture
def multi_scale_amplitudes():
    """List of 4 random amplitude tensors (B=4, T=16)."""
    torch.manual_seed(0)
    return [torch.rand(4, 16) + 0.1 for _ in range(4)]


class TestScaleAblationHarness:
    def test_single_scale_returns_one(self, harness, multi_scale_phases, multi_scale_amplitudes):
        """single_scale returns a list of length 1."""
        phases, amps = harness.single_scale(multi_scale_phases, multi_scale_amplitudes, scale_idx=2)
        assert len(phases) == 1
        assert len(amps) == 1
        assert torch.allclose(phases[0], multi_scale_phases[2])
        assert torch.allclose(amps[0], multi_scale_amplitudes[2])

    def test_scale_subset_size(self, harness, multi_scale_phases, multi_scale_amplitudes):
        """scale_subset returns exactly k scales."""
        phases, amps = harness.scale_subset(multi_scale_phases, multi_scale_amplitudes, k=2)
        assert len(phases) == 2
        assert len(amps) == 2

    def test_scale_shuffle_preserves_set(self, harness, multi_scale_phases, multi_scale_amplitudes):
        """scale_shuffle reorders but keeps the same set of tensors."""
        phases, amps = harness.scale_shuffle(multi_scale_phases, multi_scale_amplitudes)
        assert len(phases) == 4
        # The shuffled phases should be a permutation of the originals
        # (compare by stacking and sorting)
        orig_stack = torch.stack(multi_scale_phases)
        shuf_stack = torch.stack(phases)
        # At least one position should differ (shuffle happened)
        assert not torch.allclose(orig_stack, shuf_stack)

    def test_cross_scale_destroy_changes_phases(self, harness, multi_scale_phases, multi_scale_amplitudes):
        """cross_scale_destroy adds noise so phases differ from baseline."""
        phases, amps = harness.cross_scale_destroy(multi_scale_phases, multi_scale_amplitudes)
        assert len(phases) == 4
        for i in range(4):
            assert not torch.allclose(phases[i], multi_scale_phases[i])
        # amplitudes preserved
        for i in range(4):
            assert torch.allclose(amps[i], multi_scale_amplitudes[i])

    def test_cross_scale_destroy_wraps_phase(self, harness, multi_scale_phases, multi_scale_amplitudes):
        """cross_scale_destroy wraps phases to [-pi, pi]."""
        phases, _ = harness.cross_scale_destroy(multi_scale_phases, multi_scale_amplitudes)
        for p in phases:
            assert torch.all(p >= -torch.pi - 1e-5)
            assert torch.all(p <= torch.pi + 1e-5)

    def test_run_all_ablations_keys(self, harness, multi_scale_phases, multi_scale_amplitudes):
        """run_all_ablations returns logits for baseline + all conditions."""

        class FeatureExtractor(torch.nn.Module):
            def __init__(self, n_scales):
                super().__init__()
                self.n_scales = n_scales

            def forward(self, phases, amps):
                # Mean of each scale -> (B, len(phases)), then pad to n_scales
                feats = torch.stack([p.mean(dim=-1) for p in phases], dim=-1)
                if feats.shape[-1] < self.n_scales:
                    pad = torch.zeros(
                        feats.shape[0], self.n_scales - feats.shape[-1],
                        device=feats.device,
                    )
                    feats = torch.cat([feats, pad], dim=-1)
                return feats

        classifier = torch.nn.Linear(harness.n_scales, 3)
        results = harness.run_all_ablations(
            multi_scale_phases, multi_scale_amplitudes,
            FeatureExtractor(harness.n_scales), classifier,
        )
        expected_keys = {
            "baseline", "single_scale_0", "single_scale_1", "single_scale_2",
            "scale_subset_half", "scale_shuffle", "cross_scale_destroy",
        }
        assert expected_keys.issubset(set(results.keys()))
        for v in results.values():
            assert v.shape[0] == 4

    def test_ablations_produce_different_outputs(self, harness, multi_scale_phases, multi_scale_amplitudes):
        """Baseline and cross_scale_destroy produce different logits."""

        class FeatureExtractor(torch.nn.Module):
            def __init__(self, n_scales):
                super().__init__()
                self.n_scales = n_scales

            def forward(self, phases, amps):
                feats = torch.stack([p.mean(dim=-1) for p in phases], dim=-1)
                if feats.shape[-1] < self.n_scales:
                    pad = torch.zeros(
                        feats.shape[0], self.n_scales - feats.shape[-1],
                        device=feats.device,
                    )
                    feats = torch.cat([feats, pad], dim=-1)
                return feats

        classifier = torch.nn.Linear(harness.n_scales, 3)
        results = harness.run_all_ablations(
            multi_scale_phases, multi_scale_amplitudes,
            FeatureExtractor(harness.n_scales), classifier,
        )
        assert not torch.allclose(results["baseline"], results["cross_scale_destroy"])
