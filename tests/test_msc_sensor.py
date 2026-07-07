"""Tests for the WaveletCoherenceExtractor (sensor MSC instance)."""

import pytest
import torch

from bifrost.msc_sensor import WaveletCoherenceExtractor


@pytest.fixture
def extractor():
    """Small wavelet coherence extractor for 4-channel sensor data."""
    return WaveletCoherenceExtractor(
        n_scales=4,
        n_channels=4,
        sample_rate=50.0,
        wavelet="morlet",
        smoothing_window=3,
    )


@pytest.fixture
def random_sensor():
    """Batch of random multi-channel sensor signals (B=4, C=4, T=128)."""
    torch.manual_seed(42)
    return torch.randn(4, 4, 128)


class TestWaveletCoherenceExtractor:
    def test_forward_shape(self, extractor, random_sensor):
        """Forward pass produces (B, feature_dim)."""
        out = extractor(random_sensor)
        assert out.shape[0] == 4
        assert out.shape[1] == extractor.feature_dim

    def test_feature_dim_components(self, extractor):
        """feature_dim matches the documented component sum."""
        n_pairs = extractor.n_pairs
        expected = (
            2 * extractor.n_scales * n_pairs
            + 4
            + extractor.n_scales
            + n_pairs
        )
        assert extractor.feature_dim == expected

    def test_n_pairs(self, extractor):
        """n_pairs = C*(C-1)/2 for 4 channels."""
        assert extractor.n_pairs == 4 * 3 // 2

    def test_output_is_finite(self, extractor, random_sensor):
        """Output must not contain NaN or Inf."""
        out = extractor(random_sensor)
        assert torch.isfinite(out).all()

    def test_channel_assertion(self, extractor):
        """Wrong channel count raises an AssertionError."""
        bad = torch.randn(4, 3, 128)
        with pytest.raises(AssertionError):
            extractor(bad)

    def test_different_inputs_different_outputs(self, extractor):
        """Different sensor signals produce different feature vectors."""
        torch.manual_seed(0)
        x1 = torch.randn(2, 4, 128)
        torch.manual_seed(1)
        x2 = torch.randn(2, 4, 128)
        out1 = extractor(x1)
        out2 = extractor(x2)
        assert not torch.allclose(out1, out2)
