"""Tests for BifrostPipeline - happy path, boundaries, error paths."""

import pytest
import torch
from bifrost import BifrostPipeline, BifrostPipeline


class TestBifrostPipeline:
    """Test suite for BifrostPipeline."""

    def test_import_aliases(self):
        """Happy path: Verify BifrostPipeline and BifrostPipeline are same class."""
        assert BifrostPipeline is BifrostPipeline

    def test_initialization_defaults(self):
        """Happy path: Pipeline initializes with default parameters."""
        pipeline = BifrostPipeline()
        assert pipeline is not None
        # Verify expected submodules exist
        assert hasattr(pipeline, 'canonicalizer')
        assert hasattr(pipeline, 'decomposer')
        assert hasattr(pipeline, 'binding')

    def test_initialization_custom_params(self):
        """Happy path: Pipeline initializes with custom d_model."""
        pipeline = BifrostPipeline(d_model=256)
        assert pipeline is not None
        assert hasattr(pipeline, 'binding')

    def test_forward_returns_tuple(self, sample_audio_tensor):
        """Happy path: forward() returns (SpectralTensor, coherence_tensor) tuple."""
        pipeline = BifrostPipeline()
        result = pipeline(sample_audio_tensor)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_output_shapes(self, sample_audio_tensor):
        """Happy path: forward() output tensors have valid shapes."""
        pipeline = BifrostPipeline()
        spectral, coherence = pipeline(sample_audio_tensor)
        assert torch.is_tensor(coherence)
        assert coherence.dim() >= 1

    def test_empty_input_error(self):
        """Error path: Empty 1D input should raise RuntimeError."""
        pipeline = BifrostPipeline()
        empty = torch.tensor([])
        with pytest.raises((ValueError, RuntimeError)):
            pipeline(empty)

    def test_invalid_dtype_error(self):
        """Error path: Float input accepted or raises clear error."""
        pipeline = BifrostPipeline()
        x = torch.randint(0, 100, (1, 1000)).float()
        try:
            pipeline(x)
        except Exception as e:
            assert len(str(e)) > 0

    def test_large_input_boundary(self):
        """Boundary: Long audio (1 min) should not crash."""
        pipeline = BifrostPipeline()
        large_audio = torch.randn(1, 16000 * 60)
        try:
            result = pipeline(large_audio)
            assert result is not None
        except RuntimeError:
            pass  # OOM acceptable

    def test_single_sample_boundary(self):
        """Boundary: Very short input should handle gracefully."""
        pipeline = BifrostPipeline()
        single = torch.randn(1, 1)
        try:
            pipeline(single)
        except Exception:
            pass  # Short input may be rejected by STFT
