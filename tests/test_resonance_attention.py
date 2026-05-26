"""Tests for ResonanceAttention and S2 Spectral Binding."""

import pytest
import torch

from fbc.resonance_attention.attention import ResonanceAttention
from fbc.resonance_attention.binding import SpectralBinding
from fbc.spectral_tensor import SpectralTensor


@pytest.fixture
def attn():
    return ResonanceAttention(d_model=64, n_heads=4, n_bands=8, dropout=0.0)


@pytest.fixture
def binding():
    return SpectralBinding(d_model=64, n_heads=4, n_bands=8, dropout=0.0)


class TestResonanceAttention:
    def test_output_shape(self, attn):
        x = torch.randn(2, 8, 64)  # (batch, seq, d_model)
        out, coh = attn(x)
        assert out.shape == (2, 8, 64)
        assert coh.shape == (2, 4, 8, 8)  # (batch, heads, seq, seq)

    def test_coherence_softmax(self, attn):
        x = torch.randn(1, 4, 64)
        _, coh = attn(x)
        # Each row should sum to ~1 (softmax)
        row_sums = coh.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_with_explicit_phase(self, attn):
        x = torch.randn(2, 4, 64)
        phase = torch.randn(2, 4, 64)
        out, coh = attn(x, phase=phase)
        assert out.shape == (2, 4, 64)

    def test_with_mask(self, attn):
        x = torch.randn(1, 4, 64)
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[0, :, 3] = True  # mask out position 3
        out, coh = attn(x, mask=mask)
        # Weights to masked position should be ~0
        assert coh[0, :, :, 3].max().item() < 0.01

    def test_gradient_flow(self, attn):
        x = torch.randn(1, 4, 64, requires_grad=True)
        out, _ = attn(x)
        out.sum().backward()
        # Verify gradients flow to learnable parameters
        grads = [p.grad for p in attn.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_learnable_tau(self, attn):
        assert attn.tau.requires_grad
        assert attn.tau.shape == (4,)

    def test_learnable_band_weights(self, attn):
        assert attn.band_weights.requires_grad
        assert attn.band_weights.shape == (8,)

    def test_identical_phase_high_coherence(self, attn):
        """Tokens with identical phase should have high mutual coherence."""
        x = torch.randn(1, 1, 64).expand(1, 4, 64).clone()
        _, coh = attn(x)
        # Off-diagonal coherence weights should be relatively uniform
        off_diag = coh[0, 0, 0, 1:]
        assert off_diag.std().item() < 0.2


class TestSpectralBinding:
    def test_with_spectral_tensor(self, binding):
        n_freq = 64
        st = SpectralTensor(
            amplitude=torch.rand(2, 4, n_freq),
            phase=torch.rand(2, 4, n_freq) * 6.28 - 3.14,
            scale=torch.linspace(0, 8000, n_freq).unsqueeze(0).unsqueeze(0).expand(2, 4, -1),
            uncertainty=torch.ones(2, 4, n_freq),
        )
        bound_st, coh = binding(st)
        assert isinstance(bound_st, SpectralTensor)
        assert bound_st.metadata["stage"] == "bind"
        assert coh.shape[1] == 4  # n_heads

    def test_2d_input(self, binding):
        """Test with (channels, n_freq) — no batch dim."""
        n_freq = 64
        st = SpectralTensor(
            amplitude=torch.rand(3, n_freq),
            phase=torch.rand(3, n_freq),
            scale=torch.linspace(0, 8000, n_freq).unsqueeze(0).expand(3, -1),
            uncertainty=torch.ones(3, n_freq),
        )
        bound_st, coh = binding(st)
        assert isinstance(bound_st, SpectralTensor)
