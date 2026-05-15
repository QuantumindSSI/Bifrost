"""Pytest configuration and fixtures for testing."""

import pytest
import numpy as np
import os
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_wav_bytes(test_data_dir):
    """Load sample WAV file."""
    wav_path = test_data_dir / "sample.wav"
    if wav_path.exists():
        with open(wav_path, "rb") as f:
            return f.read()
    return None


@pytest.fixture
def sample_png_bytes(test_data_dir):
    """Load sample PNG file."""
    png_path = test_data_dir / "sample.png"
    if png_path.exists():
        with open(png_path, "rb") as f:
            return f.read()
    return None


@pytest.fixture
def synthetic_audio():
    """Generate synthetic audio (1 second, 16kHz, mono)."""
    sr = 16000
    duration = 1.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    frequency = 440  # A4
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio, {"sample_rate": sr, "channels": 1, "num_samples": samples}


@pytest.fixture
def synthetic_image():
    """Generate synthetic image (224x224 RGB)."""
    img = np.random.rand(224, 224, 3).astype(np.uint8)
    metadata = {
        "width": 224,
        "height": 224,
        "channels": 3,
        "bit_depth": 8,
        "color_space": "rgb",
    }
    return img, metadata
