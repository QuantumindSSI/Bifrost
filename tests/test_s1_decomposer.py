"""Tests for S1 Spectral Decomposer."""

import pytest
import torch

from bifrost.spectral_tensor import SpectralTensor
from bifrost.canonicalizer import SpectralCanonicalizer
from bifrost.decomposer import SpectralDecomposer


@pytest.fixture
def s0():
    return SpectralCanonicalizer(n_fft=256, normalize_input=True)


@pytest.fixture
def s1():
    # n_freq from canonicalizer with n_fft=256 is 129; use same for decomposer d_model
    return SpectralDecomposer(n_fft=256, n_scales=4, d_model=129, wavelet_kernel=15)


@pytest.fixture
def s0_output(s0):
    """Run S0 on a synthetic sine wave to produce a SpectralTensor."""
    sr = 16000
    t = torch.linspace(0, 1.0, sr)
    signal = torch.sin(2 * 3.14159 * 440 * t)
    return s0(signal, {"sample_rate": sr})


class TestSpectralDecomposer:
    def test_output_is_spectral_tensor(self, s1, s0_output):
        st = s1(s0_output)
        assert isinstance(st, SpectralTensor)

    def test_output_shape(self, s1, s0_output):
        st = s1(s0_output)
        assert st.amplitude.shape[-1] == 129  # n_fft=256 -> 129 bins

    def test_metadata_stage(self, s1, s0_output):
        st = s1(s0_output)
        assert st.metadata["stage"] == "decompose"
        assert st.metadata["n_scales"] == 4

    def test_uncertainty_reduced(self, s1, s0_output):
        st = s1(s0_output)
        # S1 uncertainty should generally be less than S0's uniform 1.0
        assert st.uncertainty.mean().item() < s0_output.uncertainty.mean().item()

    def test_gradient_flows(self, s1, s0_output):
        s0_output.amplitude.requires_grad_(True)
        st = s1(s0_output)
        loss = st.amplitude.sum()
        loss.backward()
        # At least one parameter should have a gradient
        grads = [p.grad for p in s1.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_multichannel(self, s0, s1):
        stereo = torch.randn(2, 16000)
        st0 = s0(stereo, {"sample_rate": 16000})
        st1 = s1(st0)
        assert st1.amplitude.shape[0] == 2
