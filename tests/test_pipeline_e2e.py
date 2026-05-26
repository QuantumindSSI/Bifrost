"""End-to-end integration tests for the FBC pipeline."""

import pytest
import numpy as np
import torch

from fbc.pipeline import FBCPipeline
from fbc.spectral_tensor import SpectralTensor


@pytest.fixture
def pipeline():
    return FBCPipeline(
        n_fft_s0=256,
        n_fft_s1=256,
        n_scales=4,
        d_model=64,
        n_heads=4,
        n_bands=8,
        dropout=0.0,
    )


class TestFBCPipelineE2E:
    def test_sine_wave(self, pipeline):
        sr = 16000
        t = torch.linspace(0, 1.0, sr)
        signal = torch.sin(2 * 3.14159 * 440 * t)
        bound_st, coh = pipeline(signal, {"sample_rate": sr})
        assert isinstance(bound_st, SpectralTensor)
        assert bound_st.metadata["stage"] == "bind"

    def test_stereo_signal(self, pipeline):
        signal = torch.randn(2, 16000)
        bound_st, coh = pipeline(signal, {"sample_rate": 16000})
        assert isinstance(bound_st, SpectralTensor)

    def test_batch_signal(self, pipeline):
        signal = torch.randn(4, 1, 16000)
        bound_st, coh = pipeline(signal, {"sample_rate": 16000})
        assert isinstance(bound_st, SpectralTensor)
        assert bound_st.amplitude.shape[0] == 4

    def test_numpy_input(self, pipeline):
        arr = np.random.randn(16000).astype(np.float32)
        bound_st, coh = pipeline.process_numpy(arr, {"sample_rate": 16000})
        assert isinstance(bound_st, SpectralTensor)

    def test_gradient_through_pipeline(self, pipeline):
        signal = torch.randn(1, 1, 8000, requires_grad=True)
        bound_st, coh = pipeline(signal, {"sample_rate": 8000})
        loss = bound_st.amplitude.sum()
        loss.backward()
        param_grads = [p.grad for p in pipeline.parameters() if p.grad is not None]
        assert len(param_grads) > 0

    def test_metadata_flows_through(self, pipeline):
        signal = torch.randn(8000)
        bound_st, _ = pipeline(signal, {"sample_rate": 8000, "source": "test"})
        assert bound_st.metadata["source"] == "test"
        assert "n_fft" in bound_st.metadata  # from canonicalizer
        assert "n_scales" in bound_st.metadata  # from decomposer
        assert "n_heads" in bound_st.metadata  # from binding

    def test_coherence_shape(self, pipeline):
        signal = torch.randn(2, 8000)
        _, coh = pipeline(signal, {"sample_rate": 8000})
        # coherence: (batch, n_heads, seq, seq) — batch was added by binding
        assert coh.dim() == 4
        assert coh.shape[1] == 4  # n_heads
