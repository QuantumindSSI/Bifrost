"""Tests for the CBMPC (Cross-Band Modulation Phase Coherence) extractor."""

import pytest
import torch

from bifrost.cbmpc import CBMPCClassifier, CBMPCExtractor


@pytest.fixture
def extractor_compact():
    """Small CBMPC extractor in compact feature mode."""
    return CBMPCExtractor(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=16,
        modulation_freqs=[1.0, 2.0, 4.0],
        duration_seconds=1.0,
        feature_mode="compact",
    )


@pytest.fixture
def extractor_rich():
    """Small CBMPC extractor in rich feature mode."""
    return CBMPCExtractor(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=16,
        modulation_freqs=[1.0, 2.0, 4.0],
        duration_seconds=1.0,
        feature_mode="rich",
    )


@pytest.fixture
def random_audio():
    """Batch of random audio waveforms (B=4, 1 second @ 16kHz)."""
    torch.manual_seed(42)
    return torch.randn(4, 16000)


class TestCBMPCExtractor:
    def test_compact_forward_shape(self, extractor_compact, random_audio):
        """Compact forward pass produces (B, 2*n_mod_freqs)."""
        out = extractor_compact(random_audio)
        assert out.shape[0] == 4
        assert out.shape[1] == extractor_compact.feature_dim
        assert out.shape[1] == 2 * extractor_compact.n_mod_freqs

    def test_rich_forward_shape(self, extractor_rich, random_audio):
        """Rich forward pass produces (B, n_mels*n_mod_freqs + 2*n_mod_freqs)."""
        out = extractor_rich(random_audio)
        assert out.shape[0] == 4
        assert out.shape[1] == extractor_rich.feature_dim
        expected = extractor_rich.n_mels * extractor_rich.n_mod_freqs + 2 * extractor_rich.n_mod_freqs
        assert out.shape[1] == expected

    def test_compact_vs_rich_dimension(self, extractor_compact, extractor_rich):
        """Compact and rich modes report different feature dimensions."""
        assert extractor_compact.feature_dim < extractor_rich.feature_dim

    def test_output_is_finite(self, extractor_compact, random_audio):
        """Output must not contain NaN or Inf."""
        out = extractor_compact(random_audio)
        assert torch.isfinite(out).all()

    def test_feature_dim_attribute(self, extractor_compact):
        """feature_dim attribute matches compact spec."""
        assert extractor_compact.feature_dim == 2 * len(extractor_compact.modulation_freqs)

    def test_classifier_forward(self, extractor_compact, random_audio):
        """CBMPCClassifier produces logits of shape (B, n_classes)."""
        clf = CBMPCClassifier(extractor_compact, n_classes=5)
        logits = clf(random_audio)
        assert logits.shape == (4, 5)
        assert torch.isfinite(logits).all()
