"""Tests for the UnifiedCoherenceMetric module."""

import pytest
import torch

from bifrost.unified_coherence import (
    CoherenceClassifier,
    CrossModalCoherenceLoss,
    UnifiedCoherenceClassifier,
    UnifiedCoherenceMetric,
)


@pytest.fixture
def metric():
    """Small unified coherence metric."""
    return UnifiedCoherenceMetric(
        audio_dim=8,
        image_dim=8,
        sensor_dim=8,
        target_dim=16,
        hidden_dim=32,
    )


@pytest.fixture
def random_audio_features():
    """Random audio coherence features (B=4, audio_dim)."""
    torch.manual_seed(42)
    return torch.randn(4, 8)


@pytest.fixture
def random_image_features():
    """Random image coherence features (B=4, image_dim)."""
    torch.manual_seed(43)
    return torch.randn(4, 8)


@pytest.fixture
def random_sensor_features():
    """Random sensor coherence features (B=4, sensor_dim)."""
    torch.manual_seed(44)
    return torch.randn(4, 8)


class TestUnifiedCoherenceMetric:
    def test_audio_forward_shape(self, metric, random_audio_features):
        """Audio projection produces (B, target_dim) unit-norm vectors."""
        out = metric(random_audio_features, "audio")
        assert out.shape == (4, 16)
        # L2-normalized => norm ~1
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_image_forward_shape(self, metric, random_image_features):
        """Image projection produces (B, target_dim)."""
        out = metric(random_image_features, "image")
        assert out.shape == (4, 16)

    def test_sensor_forward_shape(self, metric, random_sensor_features):
        """Sensor projection produces (B, target_dim)."""
        out = metric(random_sensor_features, "sensor")
        assert out.shape == (4, 16)

    def test_output_is_finite(self, metric, random_audio_features):
        """Output must not contain NaN or Inf."""
        out = metric(random_audio_features, "audio")
        assert torch.isfinite(out).all()

    def test_unknown_modality_raises(self, metric, random_audio_features):
        """Unknown modality string raises ValueError."""
        with pytest.raises(ValueError):
            metric(random_audio_features, "video")

    def test_coherence_similarity_range(self, metric, random_audio_features):
        """Cosine similarity is in [-1, 1]."""
        a = metric(random_audio_features, "audio")
        b = metric(random_audio_features, "audio")
        sim = metric.coherence_similarity(a, b)
        assert sim.shape == (4,)
        assert (sim >= -1.0 - 1e-5).all()
        assert (sim <= 1.0 + 1e-5).all()

    def test_cross_modal_distance_nonneg(self, metric, random_audio_features, random_image_features):
        """Cross-modal distance is non-negative."""
        a = metric(random_audio_features, "audio")
        b = metric(random_image_features, "image")
        dist = metric.cross_modal_distance(a, b)
        assert dist.shape == (4,)
        assert (dist >= 0).all()


class TestCrossModalCoherenceLoss:
    def test_loss_scalar(self, metric, random_audio_features, random_image_features):
        """Contrastive loss returns a scalar tensor."""
        loss_fn = CrossModalCoherenceLoss(temperature=0.07)
        a = metric(random_audio_features, "audio")
        b = metric(random_image_features, "image")
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(a, b, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_triplet_loss(self, metric, random_audio_features):
        """Triplet loss returns a non-negative scalar."""
        loss_fn = CrossModalCoherenceLoss()
        a = metric(random_audio_features, "audio")
        p = metric(random_audio_features, "audio")
        n = metric(random_image_features if False else torch.randn(4, 8), "audio")
        loss = loss_fn.forward_triplet(a, p, n, margin=0.5)
        assert loss.dim() == 0
        assert loss >= 0


class TestCoherenceClassifier:
    def test_forward_shape(self):
        """CoherenceClassifier produces (B, n_classes)."""
        clf = CoherenceClassifier(feature_dim=8, n_classes=3)
        x = torch.randn(4, 8)
        out = clf(x)
        assert out.shape == (4, 3)


class TestUnifiedCoherenceClassifier:
    def test_forward_shape(self, metric, random_audio_features):
        """UnifiedCoherenceClassifier produces (B, n_classes)."""
        clf = UnifiedCoherenceClassifier(metric, n_classes=3)
        out = clf(random_audio_features, "audio")
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all()
