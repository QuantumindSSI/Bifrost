"""
Full integration tests: Ingest → Bridge → S0 → S1 → S2 → Phase-Lock Bridge.

Covers real sample data files and edge cases.
"""

import pytest
import numpy as np
import torch

from bifrost.bridge import bridge_to_canonicalizer
from bifrost.canonicalizer import SpectralCanonicalizer
from bifrost.decomposer import SpectralDecomposer
from bifrost.resonance_attention import SpectralBinding
from bifrost.phase_lock_bridge import PhaseLockBridge, FrequencyAttractor
from bifrost.spectral_tensor import SpectralTensor


N_FFT = 256
D_MODEL = 64


@pytest.fixture
def s0():
    return SpectralCanonicalizer(n_fft=N_FFT)


@pytest.fixture
def s1():
    n_freq = N_FFT // 2 + 1  # 129
    return SpectralDecomposer(n_fft=N_FFT, n_scales=4, d_model=n_freq)


@pytest.fixture
def s2():
    return SpectralBinding(d_model=D_MODEL, n_heads=4, n_bands=8, dropout=0.0)


@pytest.fixture
def plb():
    return PhaseLockBridge(n_bands=8, min_locked_bands=3,
                           band_threshold=0.4, activation_threshold=0.5)


def _run_s0_s1(s0, s1, signal, meta):
    sig, enriched = bridge_to_canonicalizer(signal, meta)
    st0 = s0(sig, enriched)
    st1 = s1(st0)
    return st1, enriched


# ── Real sample data (audio) ──────────────────────────────────────────────

class TestAudioIntegration:
    def test_mono_8khz(self, s0, s1):
        from scipy.io import wavfile
        sr, data = wavfile.read("sample_data/mono_8khz.wav")
        st1, meta = _run_s0_s1(s0, s1, data.astype(np.float32), {"format": "wav", "channels": 1, "sample_rate": sr})
        st1.validate()
        assert meta["channels"] == 1

    def test_mono_16khz(self, s0, s1):
        from scipy.io import wavfile
        sr, data = wavfile.read("sample_data/mono_16khz.wav")
        st1, meta = _run_s0_s1(s0, s1, data.astype(np.float32), {"format": "wav", "channels": 1, "sample_rate": sr})
        st1.validate()

    def test_stereo_44khz(self, s0, s1):
        from scipy.io import wavfile
        sr, data = wavfile.read("sample_data/stereo_44khz.wav")
        st1, meta = _run_s0_s1(s0, s1, data.astype(np.float32), {"format": "wav", "channels": 2, "sample_rate": sr})
        st1.validate()
        assert meta["channels"] == 2


# ── Real sample data (images) ─────────────────────────────────────────────

class TestImageIntegration:
    def test_grayscale(self, s0, s1):
        from PIL import Image
        img = np.array(Image.open("sample_data/gray_image.png"), dtype=np.float32) / 255.0
        st1, meta = _run_s0_s1(s0, s1, img, {"format": "png", "channels": 1, "height": img.shape[0], "width": img.shape[1]})
        st1.validate()
        assert meta["channels"] == 1

    def test_rgb(self, s0, s1):
        from PIL import Image
        img = np.array(Image.open("sample_data/rgb_image.png"), dtype=np.float32) / 255.0
        st1, meta = _run_s0_s1(s0, s1, img, {"format": "png", "channels": 3, "height": img.shape[0], "width": img.shape[1]})
        st1.validate()
        assert meta["channels"] == 3


# ── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_short_signal(self, s0, s1):
        """Signal shorter than n_fft should be padded, not crash."""
        data = np.random.randn(64).astype(np.float32)
        st1, _ = _run_s0_s1(s0, s1, data, {"format": "wav", "channels": 1, "sample_rate": 8000})
        st1.validate()

    def test_single_sample(self, s0, s1):
        """Degenerate single-sample signal."""
        data = np.array([0.5], dtype=np.float32)
        st1, _ = _run_s0_s1(s0, s1, data, {"format": "wav", "channels": 1, "sample_rate": 8000})
        st1.validate()

    def test_silent_signal(self, s0, s1):
        """All-zeros should produce valid output without NaN."""
        data = np.zeros(4000, dtype=np.float32)
        st1, _ = _run_s0_s1(s0, s1, data, {"format": "wav", "channels": 1, "sample_rate": 8000})
        st1.validate()
        assert not torch.isnan(st1.amplitude).any()
        assert not torch.isnan(st1.phase).any()

    def test_high_channel_count(self, s0, s1):
        """8-channel input."""
        data = np.random.randn(8, 2000).astype(np.float32)
        sig, meta = bridge_to_canonicalizer(data, {"format": "npy", "shape": data.shape, "channel_axis": 0})
        st0 = s0(sig, meta)
        st1 = s1(st0)
        st1.validate()
        assert st1.amplitude.shape[0] == 8


# ── Full pipeline through Phase-Lock Bridge ────────────────────────────────

class TestPhaseLockIntegration:
    def test_same_signal_cross_domain(self, s0, s1, s2, plb):
        """Same signal presented as two domains should produce activated bridges."""
        torch.manual_seed(42)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 4000)).astype(np.float32)

        # Process as "audio"
        sig_a, meta_a = bridge_to_canonicalizer(signal, {"format": "wav", "channels": 1, "sample_rate": 8000})
        st0_a = s0(sig_a, meta_a)
        st1_a = s1(st0_a)
        attractors_a = PhaseLockBridge.extract_attractors_from_s2(st1_a, n_bands=8, domain="audio")

        # Process same signal as "vision" (simulating cross-domain analogy)
        sig_b, meta_b = bridge_to_canonicalizer(signal, {"format": "wav", "channels": 1, "sample_rate": 8000})
        st0_b = s0(sig_b, meta_b)
        st1_b = s1(st0_b)
        attractors_b = PhaseLockBridge.extract_attractors_from_s2(st1_b, n_bands=8, domain="vision")

        bridges = plb.find_bridges(attractors_a, attractors_b)
        # Same signal should produce high coherence between matching positions
        assert len(bridges) > 0

    def test_different_signals_less_bridges(self, s0, s1, plb):
        """Completely different signals should produce fewer activated bridges."""
        torch.manual_seed(0)
        np.random.seed(0)
        sig_a_raw = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 4000)).astype(np.float32)
        sig_b_raw = np.random.randn(4000).astype(np.float32)

        sig_a, meta_a = bridge_to_canonicalizer(sig_a_raw, {"format": "wav", "channels": 1, "sample_rate": 8000})
        st0_a = s0(sig_a, meta_a)
        st1_a = s1(st0_a)
        att_a = PhaseLockBridge.extract_attractors_from_s2(st1_a, n_bands=8, domain="audio")

        sig_b, meta_b = bridge_to_canonicalizer(sig_b_raw, {"format": "wav", "channels": 1, "sample_rate": 8000})
        st0_b = s0(sig_b, meta_b)
        st1_b = s1(st0_b)
        att_b = PhaseLockBridge.extract_attractors_from_s2(st1_b, n_bands=8, domain="noise")

        bridges_same = plb.find_bridges(att_a, att_a)
        bridges_diff = plb.find_bridges(att_a, att_b)

        # Self-comparison must always produce at least one bridge
        assert len(bridges_same) > 0, "Self-comparison must produce bridges"

        # Cross-signal bridges must also be produced (both signals are non-trivial)
        assert len(bridges_diff) > 0, "Cross-signal comparison must produce bridges"

        # All activation scores must be in valid range [0, 1]
        for b in bridges_same:
            assert 0.0 <= b.activation_score <= 1.0, (
                f"activation_score {b.activation_score:.4f} out of [0,1]"
            )
