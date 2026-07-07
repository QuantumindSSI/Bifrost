"""Tests for the PhaseCoherenceSignalMetrics module."""

import pytest
import torch

from bifrost.validation.phase_metrics import PhaseCoherenceSignalMetrics


@pytest.fixture
def metrics():
    """Pure signal-processing phase metrics (no learned params)."""
    return PhaseCoherenceSignalMetrics()


@pytest.fixture
def locked_phases():
    """Two phase tensors that are perfectly locked (identical)."""
    torch.manual_seed(0)
    p = torch.rand(4, 32) * 2 * torch.pi - torch.pi
    return p, p.clone()


@pytest.fixture
def unlocked_phases():
    """Two phase tensors that are independent (low PLV)."""
    torch.manual_seed(0)
    a = torch.rand(4, 32) * 2 * torch.pi - torch.pi
    torch.manual_seed(1)
    b = torch.rand(4, 32) * 2 * torch.pi - torch.pi
    return a, b


class TestPhaseCoherenceSignalMetrics:
    def test_plv_perfect_locking(self, metrics, locked_phases):
        """PLV of identical phases is ~1."""
        a, b = locked_phases
        plv = metrics.phase_locking_value(a, b, dim=-1)
        assert plv.shape == (4,)
        assert torch.allclose(plv, torch.ones(4), atol=1e-5)

    def test_plv_range(self, metrics, unlocked_phases):
        """PLV is in [0, 1]."""
        a, b = unlocked_phases
        plv = metrics.phase_locking_value(a, b, dim=-1)
        assert (plv >= 0).all()
        assert (plv <= 1).all()

    def test_plv_unlocked_below_locked(self, metrics, locked_phases, unlocked_phases):
        """Mean PLV of unlocked phases is lower than locked phases."""
        locked = metrics.phase_locking_value(*locked_phases, dim=-1).mean()
        unlocked = metrics.phase_locking_value(*unlocked_phases, dim=-1).mean()
        assert unlocked < locked

    def test_weighted_plv(self, metrics, locked_phases):
        """Weighted PLV of identical phases is ~1 regardless of weights."""
        a, b = locked_phases
        w = torch.rand(4, 32) + 0.1
        plv = metrics.weighted_plv(a, b, w, dim=-1)
        assert plv.shape == (4,)
        assert torch.allclose(plv, torch.ones(4), atol=1e-4)

    def test_phase_entropy_range(self, metrics):
        """Phase entropy normalized to [0, 1]."""
        torch.manual_seed(0)
        phases = torch.rand(4, 128) * 2 * torch.pi - torch.pi
        ent = metrics.phase_entropy(phases, n_bins=16, dim=-1)
        assert (ent >= 0).all()
        assert (ent <= 1.0 + 1e-5).all()

    def test_phase_entropy_concentrated_lower(self, metrics):
        """Concentrated phases have lower entropy than dispersed phases."""
        # Concentrated around 0
        concentrated = torch.randn(4, 256) * 0.1
        # Dispersed uniformly
        torch.manual_seed(1)
        dispersed = torch.rand(4, 256) * 2 * torch.pi - torch.pi
        ent_conc = metrics.phase_entropy(concentrated, n_bins=32, dim=-1).mean()
        ent_disp = metrics.phase_entropy(dispersed, n_bins=32, dim=-1).mean()
        assert ent_conc < ent_disp

    def test_phase_stability_range(self, metrics):
        """Phase stability is in [0, 1]."""
        torch.manual_seed(0)
        phases = torch.rand(4, 16, 32) * 2 * torch.pi - torch.pi
        stab = metrics.phase_stability(phases, time_dim=-2)
        assert stab.shape == (4, 32)
        assert (stab >= 0).all()
        assert (stab <= 1.0 + 1e-5).all()

    def test_phase_stability_constant_high(self, metrics):
        """Constant phase over time has stability ~1."""
        # Same phase repeated across time dim
        base = torch.rand(4, 32) * 2 * torch.pi - torch.pi
        phases = base.unsqueeze(1).repeat(1, 16, 1)  # (4, 16, 32)
        stab = metrics.phase_stability(phases, time_dim=-2)
        assert torch.allclose(stab, torch.ones(4, 32), atol=1e-4)

    def test_phase_congruency_shape(self, metrics):
        """Phase congruency reduces the scale dimension."""
        torch.manual_seed(0)
        multi_phase = torch.rand(4, 5, 16) * 2 * torch.pi - torch.pi  # (B, scales, T)
        amp = torch.rand(4, 5, 16) + 0.1
        pc = metrics.phase_congruency(multi_phase, amp, scale_dim=1)
        assert pc.shape == (4, 16)
        assert (pc >= 0).all()
        assert (pc <= 1.0 + 1e-5).all()

    def test_cross_frequency_coupling(self, metrics):
        """Cross-frequency coupling returns a scalar in [0, 1]."""
        torch.manual_seed(0)
        phases = torch.rand(4, 16, 32) * 2 * torch.pi - torch.pi  # (B, freq, T)
        low_idx = [0, 1, 2]
        high_idx = [10, 11, 12]
        cfc = metrics.cross_frequency_coupling(phases, low_idx, high_idx, freq_dim=1)
        assert cfc.dim() == 0
        assert 0 <= cfc.item() <= 1.0 + 1e-5
