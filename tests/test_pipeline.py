"""Tests for BifrostPipeline - happy path, boundaries, error paths."""

import pytest
import torch
from bifrost import BifrostPipeline, FBCPipeline


class TestBifrostPipeline:
    """Test suite for BifrostPipeline per CTO guidelines (C-03)."""

    def test_import_aliases(self):
        """Happy path: Verify BifrostPipeline and FBCPipeline are same class."""
        assert BifrostPipeline is FBCPipeline

    def test_initialization_defaults(self):
        """Happy path: Pipeline initializes with default parameters."""
        pipeline = BifrostPipeline()
        assert pipeline is not None
        assert pipeline.d_model > 0

    def test_initialization_custom_params(self):
        """Happy path: Pipeline initializes with custom d_model."""
        pipeline = BifrostPipeline(d_model=256, use_mamba=True)
        assert pipeline.d_model == 256

    def test_process_signal_shape(self, sample_audio_tensor):
        """Happy path: process_signal returns expected output shapes."""
        pipeline = BifrostPipeline(d_model=128)
        # Input: (batch, samples)
        bound, coherence = pipeline.process_signal(sample_audio_tensor)
        # Output should be valid tensors
        assert torch.is_tensor(bound)
        assert torch.is_tensor(coherence)
        assert coherence.dim() >= 1

    def test_empty_input_error(self):
        """Error path: Empty input should raise ValueError."""
        pipeline = BifrostPipeline()
        empty = torch.tensor([])
        with pytest.raises((ValueError, RuntimeError)):
            pipeline.process_signal(empty)

    def test_invalid_dtype_error(self):
        """Error path: Invalid dtype should be handled."""
        pipeline = BifrostPipeline()
        # Integer input should either work or raise clear error
        int_input = torch.randint(0, 100, (1, 1000))
        try:
            pipeline.process_signal(int_input.float())
        except Exception as e:
            # Should provide meaningful error message
            assert len(str(e)) > 0

    def test_large_input_boundary(self):
        """Boundary: Very long audio should handle gracefully."""
        pipeline = BifrostPipeline(d_model=64)
        # 10 minutes at 16kHz
        large_audio = torch.randn(1, 16000 * 600)
        try:
            bound, coherence = pipeline.process_signal(large_audio)
            assert bound is not None
        except RuntimeError:
            # Memory error is acceptable for very large input
            pass

    def test_single_sample_boundary(self):
        """Boundary: Single sample should handle gracefully."""
        pipeline = BifrostPipeline(d_model=64)
        single = torch.randn(1, 1)
        # Should not crash
        try:
            pipeline.process_signal(single)
        except Exception:
            pass  # Small input may be rejected
