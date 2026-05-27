"""Tests for Phase-Lock Bridge and FrequencyAttractor."""

import pytest
import torch

from bifrost.phase_lock_bridge import PhaseLockBridge, FrequencyAttractor
from bifrost.phase_lock_bridge.bridge import BridgeCandidate
from bifrost.spectral_tensor import SpectralTensor


def _make_attractor(d=64, n_bands=8, domain="test", aid="a0", phase_offset=0.0):
    return FrequencyAttractor(
        centroid=torch.randn(d),
        phase_signature=torch.randn(n_bands) + phase_offset,
        amplitude_profile=torch.rand(d),
        stability=0.8,
        domain=domain,
        attractor_id=aid,
    )


class TestFrequencyAttractor:
    def test_properties(self):
        a = _make_attractor(d=128, n_bands=8)
        assert a.d_model == 128
        assert a.n_bands == 8

    def test_to_device(self):
        a = _make_attractor()
        a2 = a.to(torch.device("cpu"))
        assert a2.centroid.device.type == "cpu"

    def test_spectral_energy(self):
        a = _make_attractor()
        assert a.spectral_energy() > 0

    def test_repr(self):
        a = _make_attractor(aid="test_01")
        assert "test_01" in repr(a)


class TestPhaseLockBridge:
    def test_identical_phase_activates(self):
        """Attractors with identical phase should activate the bridge."""
        plb = PhaseLockBridge(n_bands=8, min_locked_bands=3,
                              band_threshold=0.4, activation_threshold=0.5)
        phase = torch.zeros(8)
        a = FrequencyAttractor(
            centroid=torch.randn(64), phase_signature=phase,
            amplitude_profile=torch.rand(64), domain="audio", attractor_id="a0",
        )
        b = FrequencyAttractor(
            centroid=torch.randn(64), phase_signature=phase,
            amplitude_profile=torch.rand(64), domain="vision", attractor_id="b0",
        )
        cand = plb.evaluate(a, b)
        assert cand.is_activated
        assert cand.activation_score > 0.9
        assert cand.n_locked_bands == 8

    def test_opposite_phase_rejected(self):
        """Attractors with anti-phase should not activate."""
        plb = PhaseLockBridge(n_bands=8, min_locked_bands=3,
                              band_threshold=0.5, activation_threshold=0.6)
        a = FrequencyAttractor(
            centroid=torch.randn(64), phase_signature=torch.zeros(8),
            amplitude_profile=torch.rand(64), domain="audio", attractor_id="a0",
        )
        b = FrequencyAttractor(
            centroid=torch.randn(64),
            phase_signature=torch.full((8,), 3.14159),  # π offset
            amplitude_profile=torch.rand(64), domain="vision", attractor_id="b0",
        )
        cand = plb.evaluate(a, b)
        assert not cand.is_activated
        assert cand.activation_score < 0.2

    def test_band_coherences_shape(self):
        plb = PhaseLockBridge(n_bands=8)
        a = _make_attractor(n_bands=8)
        b = _make_attractor(n_bands=8)
        coh = plb.compute_band_coherences(a, b)
        assert coh.shape == (8,)
        assert coh.min() >= -1.0
        assert coh.max() <= 1.0

    def test_min_locked_bands_gate(self):
        """Even high overall score should fail if too few bands lock."""
        plb = PhaseLockBridge(n_bands=8, min_locked_bands=6,
                              band_threshold=0.9, activation_threshold=0.3)
        # Only 2 bands aligned, rest random
        phase_a = torch.zeros(8)
        phase_b = torch.randn(8) * 3.0
        phase_b[:2] = 0.0  # lock first 2 bands

        a = FrequencyAttractor(
            centroid=torch.randn(64), phase_signature=phase_a,
            amplitude_profile=torch.rand(64), domain="d1", attractor_id="a",
        )
        b = FrequencyAttractor(
            centroid=torch.randn(64), phase_signature=phase_b,
            amplitude_profile=torch.rand(64), domain="d2", attractor_id="b",
        )
        cand = plb.evaluate(a, b)
        assert cand.n_locked_bands < 6
        assert not cand.is_activated

    def test_find_bridges(self):
        plb = PhaseLockBridge(n_bands=4, min_locked_bands=2,
                              band_threshold=0.4, activation_threshold=0.5)
        phase = torch.zeros(4)
        audio = [FrequencyAttractor(
            centroid=torch.randn(32), phase_signature=phase,
            amplitude_profile=torch.rand(32), domain="audio",
            attractor_id=f"aud_{i}",
        ) for i in range(3)]
        vision = [FrequencyAttractor(
            centroid=torch.randn(32), phase_signature=phase,
            amplitude_profile=torch.rand(32), domain="vision",
            attractor_id=f"vis_{i}",
        ) for i in range(3)]

        bridges = plb.find_bridges(audio, vision)
        assert len(bridges) == 9  # all pairs lock (identical phase)
        assert all(b.is_activated for b in bridges)
        # Sorted by score descending
        scores = [b.activation_score for b in bridges]
        assert scores == sorted(scores, reverse=True)

    def test_bridge_candidate_repr(self):
        plb = PhaseLockBridge(n_bands=4)
        a = _make_attractor(n_bands=4, aid="src")
        b = _make_attractor(n_bands=4, aid="tgt")
        cand = plb.evaluate(a, b)
        assert "src" in repr(cand)
        assert "tgt" in repr(cand)


class TestExtractAttractorsFromS2:
    def test_basic_extraction(self):
        st = SpectralTensor(
            amplitude=torch.rand(4, 64),
            phase=torch.rand(4, 64) * 6.28 - 3.14,
            scale=torch.linspace(0, 8000, 64).unsqueeze(0).expand(4, -1),
            uncertainty=torch.ones(4, 64),
            metadata={"stage": "bind"},
        )
        attractors = PhaseLockBridge.extract_attractors_from_s2(
            st, n_bands=8, domain="audio", prefix="aud"
        )
        assert len(attractors) == 4
        assert attractors[0].domain == "audio"
        assert attractors[0].n_bands == 8
        assert attractors[0].d_model == 64
        assert attractors[0].attractor_id == "aud_0000"

    def test_batched_extraction(self):
        st = SpectralTensor(
            amplitude=torch.rand(2, 3, 32),
            phase=torch.rand(2, 3, 32),
            scale=torch.rand(2, 3, 32),
            uncertainty=torch.ones(2, 3, 32),
            metadata={"stage": "bind"},
        )
        attractors = PhaseLockBridge.extract_attractors_from_s2(st, n_bands=4)
        assert len(attractors) == 6  # 2 * 3

    def test_1d_extraction(self):
        st = SpectralTensor(
            amplitude=torch.rand(64),
            phase=torch.rand(64),
            scale=torch.rand(64),
            uncertainty=torch.ones(64),
            metadata={"stage": "bind"},
        )
        attractors = PhaseLockBridge.extract_attractors_from_s2(st, n_bands=8)
        assert len(attractors) == 1
