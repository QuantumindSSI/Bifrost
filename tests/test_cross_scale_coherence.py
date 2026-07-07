"""Tests for the CrossScaleCoherence module."""

import pytest
import torch

from bifrost.cross_scale_coherence import CrossScaleCoherence


@pytest.fixture
def coherence():
    """Small cross-scale coherence module with 3 dyadic scales."""
    return CrossScaleCoherence(n_scales=3, dyadic=True)


@pytest.fixture
def coherence_nondyadic():
    """Cross-scale coherence module with dyadic=False."""
    return CrossScaleCoherence(n_scales=3, dyadic=False)


@pytest.fixture
def random_multi_scale_phases():
    """List of 3 random phase tensors (B=4, T=16)."""
    torch.manual_seed(42)
    return [torch.randn(4, 16) * 3.14 for _ in range(3)]


@pytest.fixture
def random_multi_scale_amplitudes():
    """List of 3 random amplitude tensors (B=4, T=16)."""
    torch.manual_seed(42)
    return [torch.rand(4, 16) for _ in range(3)]


class TestCrossScaleCoherence:
    def test_forward_shape_dyadic(self, coherence, random_multi_scale_phases):
        """Dyadic forward produces (B, 2*n_pairs + 2)."""
        out = coherence(random_multi_scale_phases)
        assert out.shape[0] == 4
        assert out.shape[1] == coherence.feature_dim
        # n_pairs for 3 scales = 3; dyadic => 2*3 + 2 = 8
        assert coherence.n_pairs == 3
        assert coherence.feature_dim == 2 * 3 + 2

    def test_forward_shape_nondyadic(self, coherence_nondyadic, random_multi_scale_phases):
        """Non-dyadic forward produces (B, n_pairs + 2)."""
        out = coherence_nondyadic(random_multi_scale_phases)
        assert out.shape[0] == 4
        assert out.shape[1] == coherence_nondyadic.feature_dim
        assert coherence_nondyadic.feature_dim == 3 + 2

    def test_output_is_finite(self, coherence, random_multi_scale_phases):
        """Output must not contain NaN or Inf."""
        out = coherence(random_multi_scale_phases)
        assert torch.isfinite(out).all()

    def test_scale_count_assertion(self, coherence):
        """Wrong number of scales raises AssertionError."""
        bad = [torch.randn(4, 16) for _ in range(2)]
        with pytest.raises(AssertionError):
            coherence(bad)

    def test_plv_in_unit_range(self, coherence, random_multi_scale_phases):
        """PLV components (first n_pairs columns) lie in [0, 1]."""
        out = coherence(random_multi_scale_phases)
        plv_part = out[:, : coherence.n_pairs]
        assert (plv_part >= 0).all()
        assert (plv_part <= 1).all()

    def test_different_inputs_different_outputs(self, coherence):
        """Different phase tensors produce different coherence features."""
        torch.manual_seed(0)
        phases_a = [torch.randn(2, 16) for _ in range(3)]
        torch.manual_seed(1)
        phases_b = [torch.randn(2, 16) for _ in range(3)]
        out_a = coherence(phases_a)
        out_b = coherence(phases_b)
        assert not torch.allclose(out_a, out_b)
