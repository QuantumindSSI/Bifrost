"""Tests for S0 Canonicalizer."""

import pytest
import numpy as np
import torch

from bifrost.canonicalizer import SpectralCanonicalizer
from bifrost.spectral_tensor import SpectralTensor


@pytest.fixture
def canonicalizer():
    return SpectralCanonicalizer(n_fft=256, normalize_input=True)


@pytest.fixture
def sine_signal():
    """440 Hz sine wave at 16 kHz, 1 second."""
    sr = 16000
    t = torch.linspace(0, 1.0, sr)
    return torch.sin(2 * 3.14159 * 440 * t), {"sample_rate": sr}


class TestSpectralCanonicalizer:
    def test_output_is_spectral_tensor(self, canonicalizer, sine_signal):
        signal, meta = sine_signal
        st = canonicalizer(signal, meta)
        assert isinstance(st, SpectralTensor)

    def test_output_shape(self, canonicalizer, sine_signal):
        signal, meta = sine_signal
        st = canonicalizer(signal, meta)
        n_freq = 256 // 2 + 1  # 129
        assert st.amplitude.shape[-1] == n_freq

    def test_amplitude_normalized(self, canonicalizer, sine_signal):
        signal, meta = sine_signal
        st = canonicalizer(signal, meta)
        assert st.amplitude.max().item() <= 1.0 + 1e-6

    def test_phase_range(self, canonicalizer, sine_signal):
        signal, meta = sine_signal
        st = canonicalizer(signal, meta)
        assert st.phase.min().item() >= -3.15
        assert st.phase.max().item() <= 3.15

    def test_metadata_propagated(self, canonicalizer, sine_signal):
        signal, meta = sine_signal
        st = canonicalizer(signal, meta)
        assert st.metadata["stage"] == "canonicalize"
        assert st.metadata["sample_rate"] == 16000

    def test_validate_passes(self, canonicalizer, sine_signal):
        signal, meta = sine_signal
        st = canonicalizer(signal, meta)
        st.validate()

    def test_short_signal_padded(self, canonicalizer):
        short = torch.randn(100)
        st = canonicalizer(short, {"sample_rate": 8000})
        assert st.amplitude.shape[-1] == 129

    def test_multichannel(self, canonicalizer):
        stereo = torch.randn(2, 16000)
        st = canonicalizer(stereo, {"sample_rate": 16000})
        assert st.amplitude.shape[0] == 2

    def test_from_numpy(self, canonicalizer):
        arr = np.random.randn(16000).astype(np.float32)
        st = canonicalizer.from_numpy(arr, {"sample_rate": 16000})
        assert isinstance(st, SpectralTensor)

    def test_batch_dimension(self, canonicalizer):
        batch = torch.randn(4, 1, 16000)
        st = canonicalizer(batch, {"sample_rate": 16000})
        assert st.amplitude.shape[0] == 4
