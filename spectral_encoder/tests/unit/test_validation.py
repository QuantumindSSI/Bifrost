"""Unit tests for validation modules."""

import pytest
import numpy as np
from spectral_encoder.ingest.validation.audio import AudioValidator
from spectral_encoder.ingest.validation.image import ImageValidator


class TestAudioValidator:
    """Test audio validation logic."""

    def test_valid_audio(self):
        """Test validation of valid audio."""
        audio = np.random.randn(16000).astype(np.float32)
        metadata = {
            "sample_rate": 16000,
            "bit_depth": 16,
            "channels": 1,
            "num_samples": 16000,
            "duration_sec": 1.0,
        }
        is_valid, msg = AudioValidator.validate(audio, metadata)
        assert is_valid, msg

    def test_invalid_sample_rate_low(self):
        """Test rejection of too-low sample rate."""
        audio = np.random.randn(1000).astype(np.float32)
        metadata = {
            "sample_rate": 4000,
            "bit_depth": 16,
            "channels": 1,
            "num_samples": 1000,
            "duration_sec": 0.25,
        }
        is_valid, msg = AudioValidator.validate(audio, metadata)
        assert not is_valid

    def test_invalid_sample_rate_high(self):
        """Test rejection of too-high sample rate."""
        audio = np.random.randn(96000).astype(np.float32)
        metadata = {
            "sample_rate": 96000,
            "bit_depth": 16,
            "channels": 1,
            "num_samples": 96000,
            "duration_sec": 1.0,
        }
        is_valid, msg = AudioValidator.validate(audio, metadata)
        assert not is_valid

    def test_invalid_duration_too_short(self):
        """Test rejection of very short duration."""
        audio = np.random.randn(100).astype(np.float32)
        metadata = {
            "sample_rate": 16000,
            "bit_depth": 16,
            "channels": 1,
            "num_samples": 100,
            "duration_sec": 0.005,
        }
        is_valid, msg = AudioValidator.validate(audio, metadata)
        assert not is_valid

    def test_nan_detection(self):
        """Test detection of NaN values."""
        audio = np.random.randn(16000).astype(np.float32)
        audio[1000] = np.nan
        metadata = {
            "sample_rate": 16000,
            "bit_depth": 16,
            "channels": 1,
            "num_samples": 16000,
            "duration_sec": 1.0,
        }
        is_valid, msg = AudioValidator.validate(audio, metadata)
        assert not is_valid
        assert "NaN" in msg

    def test_inf_detection(self):
        """Test detection of Inf values."""
        audio = np.random.randn(16000).astype(np.float32)
        audio[500] = np.inf
        metadata = {
            "sample_rate": 16000,
            "bit_depth": 16,
            "channels": 1,
            "num_samples": 16000,
            "duration_sec": 1.0,
        }
        is_valid, msg = AudioValidator.validate(audio, metadata)
        assert not is_valid
        assert "Inf" in msg


class TestImageValidator:
    """Test image validation logic."""

    def test_valid_image(self):
        """Test validation of valid image."""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        metadata = {
            "width": 224,
            "height": 224,
            "channels": 3,
            "bit_depth": 8,
            "color_space": "rgb",
        }
        is_valid, msg = ImageValidator.validate(image, metadata)
        assert is_valid, msg

    def test_valid_grayscale(self):
        """Test validation of grayscale image."""
        image = np.random.rand(224, 224).astype(np.float32)
        metadata = {
            "width": 224,
            "height": 224,
            "channels": 1,
            "bit_depth": 8,
            "color_space": "grayscale",
        }
        is_valid, msg = ImageValidator.validate(image, metadata)
        assert is_valid, msg

    def test_invalid_width_too_small(self):
        """Test rejection of too-small width."""
        image = np.random.rand(10, 224).astype(np.float32)
        metadata = {
            "width": 10,
            "height": 224,
            "channels": 1,
            "bit_depth": 8,
            "color_space": "grayscale",
        }
        is_valid, msg = ImageValidator.validate(image, metadata)
        assert not is_valid

    def test_invalid_channels(self):
        """Test rejection of invalid channel count."""
        image = np.random.rand(224, 224, 5).astype(np.float32)
        metadata = {
            "width": 224,
            "height": 224,
            "channels": 5,
            "bit_depth": 8,
            "color_space": "unknown",
        }
        is_valid, msg = ImageValidator.validate(image, metadata)
        assert not is_valid

    def test_nan_detection(self):
        """Test detection of NaN in image."""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        image[100, 100, 0] = np.nan
        metadata = {
            "width": 224,
            "height": 224,
            "channels": 3,
            "bit_depth": 8,
            "color_space": "rgb",
        }
        is_valid, msg = ImageValidator.validate(image, metadata)
        assert not is_valid
