"""Tests for demo utilities and demo_1_antiphase functionality."""

import pytest
import torch
import sys
import os

# Add demos to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demos'))


class TestDemoUtils:
    """Test suite for demo utility functions."""

    def test_hilbert_antiphase_shape(self):
        """Happy path: hilbert_antiphase returns correct shapes."""
        from demos.utils import hilbert_antiphase

        signal = torch.randn(1, 128)
        a, b = hilbert_antiphase(signal, n_samples=64)

        assert a.shape == (64,)
        assert b.shape == (64,)

    def test_attention_l1_distance_non_negative(self):
        """Property: L1 distance is always non-negative."""
        from demos.utils import attention_l1_distance

        w1 = torch.rand(2, 4, 10, 10)
        w2 = torch.rand(2, 4, 10, 10)
        dist = attention_l1_distance(w1, w2)

        assert dist >= 0.0

    def test_attention_l1_distance_symmetry(self):
        """Property: L1 distance is symmetric."""
        from demos.utils import attention_l1_distance

        w1 = torch.rand(2, 4, 10, 10)
        w2 = torch.rand(2, 4, 10, 10)
        dist_ab = attention_l1_distance(w1, w2)
        dist_ba = attention_l1_distance(w2, w1)

        assert abs(dist_ab - dist_ba) < 1e-6

    def test_attention_l1_distance_zero_for_identical(self):
        """Property: Identical inputs give zero distance."""
        from demos.utils import attention_l1_distance

        w = torch.rand(2, 4, 10, 10)
        dist = attention_l1_distance(w, w)

        assert abs(dist) < 1e-6


class TestDemo1AntiPhase:
    """Test suite for demo_1_antiphase."""

    def test_discriminate_pair_assertions(self):
        """Happy path: discriminate_pair passes NASA Rule 5 assertions."""
        # Import the function directly
        import demos.demo_1_antiphase as demo

        # Create synthetic anti-phase pair
        signal = torch.randn(1, 128)
        label, feat_a, feat_b, phase_a, phase_b = demo._create_antiphase_pair(signal)

        # Should not raise assertion error
        result = demo.discriminate_pair(label, feat_a, feat_b, phase_a, phase_b)

        assert result.passes is True
        assert result.ratio >= 1.0
        assert result.res_distance > result.dot_distance

    def test_load_wav_file_not_found(self):
        """Error path: Loading non-existent file raises error."""
        import demos.demo_1_antiphase as demo

        with pytest.raises((FileNotFoundError, OSError)):
            demo._load_wav("nonexistent_file.wav")

    def test_pass_ratio_constant(self):
        """Happy path: PASS_RATIO is reasonable threshold."""
        import demos.demo_1_antiphase as demo

        assert demo.PASS_RATIO >= 1.0
        assert isinstance(demo.PASS_RATIO, float)
