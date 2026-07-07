"""Tests for the PhaseCongruencyExtractor (image MSC instance)."""

import pytest
import torch

from bifrost.msc_image import PhaseCongruencyExtractor


@pytest.fixture
def extractor():
    """Small phase congruency extractor for 16x16 images."""
    return PhaseCongruencyExtractor(
        n_scales=3,
        n_orientations=4,
        base_wavelength=3.0,
        scale_factor=2.0,
        image_size=16,
        n_pc_bins=8,
    )


@pytest.fixture
def random_images():
    """Batch of random grayscale images (B=4, 1, 16, 16)."""
    torch.manual_seed(42)
    return torch.randn(4, 1, 16, 16)


@pytest.fixture
def random_rgb_images():
    """Batch of random RGB images (B=4, 3, 16, 16)."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 16, 16)


class TestPhaseCongruencyExtractor:
    def test_forward_shape(self, extractor, random_images):
        """Forward pass produces (B, feature_dim)."""
        out = extractor(random_images)
        assert out.shape[0] == 4
        assert out.shape[1] == extractor.feature_dim

    def test_feature_dim_components(self, extractor):
        """feature_dim equals the documented sum of components."""
        expected = (
            extractor.n_pc_bins
            + extractor.n_scales
            + extractor.n_orientations
            + extractor.n_scales * extractor.n_orientations
            + 4
            + 64
        )
        assert extractor.feature_dim == expected

    def test_rgb_input_shape(self, extractor, random_rgb_images):
        """Multi-channel images are converted to grayscale and still work."""
        out = extractor(random_rgb_images)
        assert out.shape[0] == 4
        assert out.shape[1] == extractor.feature_dim

    def test_output_is_finite(self, extractor, random_images):
        """Output must not contain NaN or Inf."""
        out = extractor(random_images)
        assert torch.isfinite(out).all()

    def test_different_inputs_different_outputs(self, extractor):
        """Different images should produce different feature vectors."""
        torch.manual_seed(0)
        x1 = torch.randn(2, 1, 16, 16)
        torch.manual_seed(1)
        x2 = torch.randn(2, 1, 16, 16)
        out1 = extractor(x1)
        out2 = extractor(x2)
        assert not torch.allclose(out1, out2)
