"""Unit tests for normalization module."""

import pytest
import numpy as np
from spectral_encoder.ingest.normalize import Normalizer


class TestNormalizer:
    """Test normalization functions."""

    def test_normalize_audio_float32_in_range(self):
        """Test normalization of float32 audio already in [-1, 1]."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        normalized = Normalizer.normalize_audio(audio, {})
        
        assert normalized.dtype == np.float32
        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)

    def test_normalize_audio_clipping(self):
        """Test that audio outside [-1, 1] is clipped."""
        audio = np.array([0.0, 2.0, -3.0], dtype=np.float32)
        normalized = Normalizer.normalize_audio(audio, {})
        
        assert normalized[1] <= 1.0
        assert normalized[2] >= -1.0

    def test_normalize_audio_very_quiet(self):
        """Test scaling up very quiet audio."""
        audio = np.array([0.0, 0.0001, -0.0001], dtype=np.float32)
        normalized = Normalizer.normalize_audio(audio, {})
        
        # Should scale up by 10x
        assert np.abs(normalized).max() > 0.0005

    def test_normalize_image_uint8_to_float32(self):
        """Test conversion of uint8 image to float32."""
        image = np.array([0, 128, 255], dtype=np.uint8).reshape(3, 1, 1)
        normalized = Normalizer.normalize_image(image, {})
        
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)
        assert abs(normalized[1, 0, 0] - 0.502) < 0.01  # 128/255 ≈ 0.502

    def test_normalize_image_clipping(self):
        """Test that image values > 1 are clipped."""
        image = np.array([0.0, 0.5, 2.0], dtype=np.float32).reshape(3, 1, 1)
        normalized = Normalizer.normalize_image(image, {})
        
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    def test_normalize_tensor_large_values(self):
        """Test normalization of large-valued tensor."""
        tensor = np.array([0, 1000, 2000], dtype=np.float32)
        normalized = Normalizer.normalize_tensor(tensor)
        
        assert normalized.dtype == np.float32
        assert np.abs(normalized).max() <= 1.0

    def test_normalize_tensor_small_values(self):
        """Test scaling up small-valued tensor."""
        tensor = np.array([0, 0.0001, 0.0002], dtype=np.float32)
        normalized = Normalizer.normalize_tensor(tensor)
        
        # Should scale up
        assert np.abs(normalized).max() > 0.0005

    def test_normalize_tensor_zero(self):
        """Test handling of all-zero tensor."""
        tensor = np.zeros(10, dtype=np.float32)
        normalized = Normalizer.normalize_tensor(tensor)
        
        assert np.allclose(normalized, 0.0)
