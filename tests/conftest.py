"""Pytest configuration and shared fixtures for Bifröst tests."""

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_audio_tensor():
    """Generate a sample audio tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(1, 16000)  # 1 second at 16kHz


@pytest.fixture
def sample_spectral_input():
    """Generate sample spectral input (features, samples) format."""
    torch.manual_seed(42)
    return torch.randn(128, 100)  # 128 features, 100 time steps


@pytest.fixture
def device():
    """Return device for testing (CPU for determinism)."""
    return torch.device("cpu")
