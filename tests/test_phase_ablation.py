"""Tests for the PhaseAblationHarness."""

import pytest
import torch

from bifrost.spectral_tensor import SpectralTensor
from bifrost.validation.phase_ablation import PhaseAblationHarness


@pytest.fixture
def harness():
    """Phase ablation harness with fixed seed."""
    return PhaseAblationHarness(seed=42)


@pytest.fixture
def spectral_tensor():
    """A small SpectralTensor (B=4, T=8, F=16) with valid invariants.

    Shape convention: dim=-2 is time, dim=-1 is frequency bands.
    """
    torch.manual_seed(0)
    B, T, F = 4, 8, 16
    amplitude = torch.rand(B, T, F) + 0.1
    phase = torch.rand(B, T, F) * 2 * torch.pi - torch.pi
    scale = torch.ones(B, T, F)
    uncertainty = torch.rand(B, T, F) * 0.1
    return SpectralTensor(
        amplitude=amplitude,
        phase=phase,
        scale=scale,
        uncertainty=uncertainty,
        metadata={"source": "test"},
    )


def _phases_differ(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True if the two phase tensors are not elementwise equal."""
    return not torch.allclose(a, b)


class TestPhaseAblationHarness:
    def test_phase_zero_sets_phase_to_zero(self, harness, spectral_tensor):
        """phase_zero produces an all-zero phase tensor."""
        out = harness.phase_zero(spectral_tensor)
        assert torch.all(out.phase == 0)
        # amplitude preserved
        assert torch.allclose(out.amplitude, spectral_tensor.amplitude)

    def test_phase_zero_changes_phase(self, harness, spectral_tensor):
        """phase_zero changes the phase relative to baseline."""
        out = harness.phase_zero(spectral_tensor)
        assert _phases_differ(out.phase, spectral_tensor.phase)

    def test_phase_randomize_changes_phase(self, harness, spectral_tensor):
        """phase_randomize shuffles phase so it differs from baseline."""
        out = harness.phase_randomize(spectral_tensor)
        assert _phases_differ(out.phase, spectral_tensor.phase)
        # amplitude preserved
        assert torch.allclose(out.amplitude, spectral_tensor.amplitude)

    def test_phase_noise_changes_phase(self, harness, spectral_tensor):
        """phase_noise adds noise so the phase differs from baseline."""
        out = harness.phase_noise(spectral_tensor, sigma=0.5)
        assert _phases_differ(out.phase, spectral_tensor.phase)
        # wrapped to [-pi, pi]
        assert torch.all(out.phase >= -torch.pi - 1e-5)
        assert torch.all(out.phase <= torch.pi + 1e-5)

    def test_phase_quantize_changes_phase(self, harness, spectral_tensor):
        """phase_quantize reduces precision so phase differs from baseline."""
        out = harness.phase_quantize(spectral_tensor, n_levels=4)
        assert _phases_differ(out.phase, spectral_tensor.phase)
        # wrapped to [-pi, pi]
        assert torch.all(out.phase >= -torch.pi - 1e-5)
        assert torch.all(out.phase <= torch.pi + 1e-5)

    def test_cross_band_scramble_changes_phase(self, harness, spectral_tensor):
        """cross_band_scramble reorders bands so phase differs from baseline."""
        out = harness.cross_band_scramble(spectral_tensor)
        assert _phases_differ(out.phase, spectral_tensor.phase)
        # amplitude preserved
        assert torch.allclose(out.amplitude, spectral_tensor.amplitude)

    def test_cross_scale_scramble_changes_phase(self, harness, spectral_tensor):
        """cross_scale_scramble adds per-scale offsets so phases differ."""
        st2 = SpectralTensor(
            amplitude=spectral_tensor.amplitude.clone(),
            phase=spectral_tensor.phase.clone(),
            scale=spectral_tensor.scale.clone(),
            uncertainty=spectral_tensor.uncertainty.clone(),
        )
        out_list = harness.cross_scale_scramble([spectral_tensor, st2])
        assert len(out_list) == 2
        assert _phases_differ(out_list[0].phase, spectral_tensor.phase)
        assert _phases_differ(out_list[1].phase, st2.phase)

    def test_ablation_metadata_tagged(self, harness, spectral_tensor):
        """Each ablation tags the metadata with its name."""
        assert harness.phase_zero(spectral_tensor).metadata["ablation"] == "phase_zero"
        assert harness.phase_randomize(spectral_tensor).metadata["ablation"] == "phase_randomize"
        assert harness.cross_band_scramble(spectral_tensor).metadata["ablation"] == "cross_band_scramble"

    def test_run_all_ablations_keys(self, harness, spectral_tensor):
        """run_all_ablations returns logits for baseline + all conditions."""
        flat_dim = spectral_tensor.amplitude.shape[-1] * spectral_tensor.amplitude.shape[-2]

        class FlattenExtractor(torch.nn.Module):
            def forward(self, st):
                return st.amplitude.reshape(st.amplitude.shape[0], -1)

        classifier = torch.nn.Linear(flat_dim, 3)
        results = harness.run_all_ablations(
            spectral_tensor, FlattenExtractor(), classifier
        )
        expected_keys = {
            "baseline", "phase_zero", "phase_randomize",
            "phase_noise_0.5", "phase_noise_2.0",
            "phase_quantize_4", "cross_band_scramble",
        }
        assert expected_keys.issubset(set(results.keys()))
        # each value is a logits tensor (B, n_classes)
        for k, v in results.items():
            assert v.shape[0] == 4
