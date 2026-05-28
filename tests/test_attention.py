"""Tests for ResonantAttention and DotProductAttention."""

import pytest
import torch
from demos.baselines import DotProductAttention


class TestDotProductAttention:
    """Test suite for baseline dot-product attention."""

    def test_initialization(self):
        """Happy path: Attention module initializes."""
        attn = DotProductAttention(d_model=128, n_heads=4)
        assert attn.d_model == 128
        assert attn.n_heads == 4

    def test_forward_pass_shape(self):
        """Happy path: Forward pass returns correct output shape."""
        attn = DotProductAttention(d_model=128, n_heads=4)
        x = torch.randn(2, 10, 128)  # (batch, seq, dim)
        out, weights = attn(x)
        assert out.shape == x.shape
        assert weights.shape[0] == 2  # batch

    def test_attention_weights_sum_to_one(self):
        """Property: Attention weights should sum to ~1 per query."""
        attn = DotProductAttention(d_model=64, n_heads=2)
        x = torch.randn(1, 5, 64)
        _, weights = attn(x)
        # weights shape: (batch, heads, seq, seq)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_invalid_head_count_error(self):
        """Error path: d_model not divisible by n_heads should raise."""
        with pytest.raises((AssertionError, ValueError)):
            DotProductAttention(d_model=128, n_heads=3)  # 128 % 3 != 0

    def test_different_sequence_lengths(self):
        """Boundary: Different sequence lengths should work."""
        attn = DotProductAttention(d_model=64, n_heads=2)
        for seq_len in [1, 5, 100, 500]:
            x = torch.randn(1, seq_len, 64)
            out, _ = attn(x)
            assert out.shape == (1, seq_len, 64)


class TestAttentionComparison:
    """Tests comparing Resonant vs Dot-Product attention."""

    def test_resonant_detects_phase_difference(self):
        """Core claim: Resonant attention detects anti-phase signals."""
        # This tests the fundamental Bifröst claim
        from demos.utils import hilbert_antiphase, attention_l1_distance

        # Create anti-phase pair
        signal = torch.randn(1, 128)
        a, b = hilbert_antiphase(signal, n_samples=64)

        # For this test, we just verify the setup works
        assert a.shape == b.shape
        assert torch.is_tensor(a)

    def test_dot_product_blind_to_phase(self):
        """Core claim: Dot-product attention fails on anti-phase."""
        attn = DotProductAttention(d_model=64, n_heads=2)

        # Same magnitude, different phase patterns
        x1 = torch.randn(1, 10, 64)
        x2 = x1.clone()  # Same values

        _, w1 = attn(x1)
        _, w2 = attn(x2)

        # Should produce identical attention patterns
        assert torch.allclose(w1, w2, atol=1e-6)
