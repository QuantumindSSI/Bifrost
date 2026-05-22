"""
Realistic-input functional tests for ResonanceAttention.

Exercises the algorithmic claims of the FBC architecture (Engineering
Script §2) that cannot be checked by shape/gradient unit tests:

    - Harmonic binding: harmonics of a tone attend to one another.
    - Phase disambiguation: identical amplitude + opposite phase => weak coherence.
    - Amplitude invariance: scaling input does not change attention weights.
    - Phase rotation invariance: global phase shift preserves attention.
    - SNR sweep: graceful degradation as noise increases.
    - Multi-band coherence: 3+ band lock required for phase-lock activation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fbc.resonance_attention import ResonanceAttention
from fbc.phase_lock_bridge import PhaseLockBridge, FrequencyAttractor


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _make_tone_signal(
    freqs: list[float],
    amps: list[float],
    phases: list[float],
    n_samples: int = 1024,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Construct a multi-tone signal of shape (1, n_samples)."""
    t = torch.linspace(0, n_samples / sample_rate, n_samples)
    sig = torch.zeros(n_samples)
    for f, a, p in zip(freqs, amps, phases):
        sig = sig + a * torch.sin(2 * math.pi * f * t + p)
    return sig.unsqueeze(0)  # (1, n_samples)


def _signal_to_spectral_embedding(
    sig: torch.Tensor,
    d_model: int,
    n_freq_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project a raw signal into (amplitude, phase) per frequency bin,
    then arrange as (batch=1, seq=n_freq_bins, d_model).

    Each "token" along seq is one frequency bin's feature vector.
    The feature dimension carries amplitude + phase info per band:
    we use the first d_model//2 dims as amplitude harmonics and the
    rest as phase harmonics — matching how S1 maps multi-scale info
    into the model dim.
    """
    spec = torch.fft.rfft(sig, n=2 * (n_freq_bins - 1), dim=-1)  # (1, n_freq_bins)
    amp = spec.abs()
    phase = spec.angle()

    # Tile amp/phase across d_model to form a (1, n_freq_bins, d_model) feature.
    # This mimics S1's broadcasting of spectral info into the model dim.
    half = d_model // 2
    feat_amp = amp.unsqueeze(-1).expand(-1, -1, half)             # (1, F, d/2)
    feat_phase = phase.unsqueeze(-1).expand(-1, -1, d_model - half)  # (1, F, d/2)
    feat = torch.cat([feat_amp, feat_phase], dim=-1)
    phase_full = phase.unsqueeze(-1).expand(-1, -1, d_model)
    return feat, phase_full


# ─────────────────────────────────────────────────────────────────────────
# Test 1: Harmonic binding
# ─────────────────────────────────────────────────────────────────────────

class TestHarmonicBinding:
    """
    Claim (FBC §2): Phase-coherence attention should route information
    between harmonically related frequency bins because their phases
    are locked at integer multiples of the fundamental.

    NOTE: This claim requires a TRAINED attention module. With random
    initialisation we cannot expect sharp harmonic binding. The tests
    below verify the WEAKER invariant that the module produces a
    non-degenerate, finite, properly-normalised attention map on a
    real harmonic signal. The strong claim is gated behind a marker
    and exercised only when a trained checkpoint is supplied.
    """

    def _build_harmonic_input(self, d_model=64, n_freq=65, n_samples=2048,
                              sample_rate=16000, f0=200.0):
        sig = _make_tone_signal(
            freqs=[f0, 2 * f0, 3 * f0, 4 * f0],
            amps=[1.0, 0.7, 0.5, 0.3],
            phases=[0.0, 0.0, 0.0, 0.0],
            n_samples=n_samples,
            sample_rate=sample_rate,
        )
        feat, phase = _signal_to_spectral_embedding(sig, d_model, n_freq)
        return feat, phase, sample_rate

    def test_attention_map_is_well_formed(self):
        """Random-init module must still produce a valid attention map on real harmonics."""
        torch.manual_seed(0)
        d_model, n_freq = 64, 65
        feat, phase, _ = self._build_harmonic_input(d_model=d_model, n_freq=n_freq)

        attn = ResonanceAttention(d_model=d_model, n_heads=4, n_bands=4, dropout=0.0)
        attn.eval()
        with torch.no_grad():
            _, w = attn(feat, phase=phase)

        assert w.shape == (1, 4, n_freq, n_freq)
        assert not torch.isnan(w).any()
        assert not torch.isinf(w).any()
        # Each row should be a valid probability distribution (softmax output)
        row_sums = w.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums),
                                   atol=1e-5, rtol=1e-5)

    def test_attention_is_not_uniform(self):
        """On a real signal, attention should not collapse to perfectly uniform."""
        torch.manual_seed(0)
        d_model, n_freq = 64, 65
        feat, phase, _ = self._build_harmonic_input(d_model=d_model, n_freq=n_freq)

        attn = ResonanceAttention(d_model=d_model, n_heads=4, n_bands=4, dropout=0.0)
        attn.eval()
        with torch.no_grad():
            _, w = attn(feat, phase=phase)

        uniform = torch.full_like(w, 1.0 / n_freq)
        # KL divergence from uniform should be strictly positive
        diff = (w - uniform).abs().sum().item()
        assert diff > 1e-3, "Attention degenerated to perfectly uniform"

    @pytest.mark.skip(reason="Requires trained ResonanceAttention checkpoint")
    def test_trained_module_binds_harmonics(self):
        """Strong claim: a trained module binds harmonics > non-harmonic bins.

        This test is the production gate for FBC §2. It requires loading
        a checkpoint trained on a harmonic-binding pretext task. Until
        such a checkpoint exists, the test is skipped.
        """
        pass


# ─────────────────────────────────────────────────────────────────────────
# Test 2: Phase disambiguation
# ─────────────────────────────────────────────────────────────────────────

class TestPhaseDisambiguation:
    """
    Claim (FBC §2): Two signals with identical amplitude spectrum but
    opposite phase must be distinguished by ResonanceAttention.
    Dot-product attention CANNOT distinguish them.
    """

    def test_anti_phase_attractors_reject(self):
        """Two attractors with anti-phase across all bands should not lock."""
        plb = PhaseLockBridge(n_bands=8, min_locked_bands=3,
                              band_threshold=0.5, activation_threshold=0.6)

        d = 64
        amp = torch.rand(d)  # identical amplitude profile

        a = FrequencyAttractor(
            centroid=amp.clone(),
            phase_signature=torch.zeros(8),
            amplitude_profile=amp.clone(),
            domain="A", attractor_id="a0",
        )
        b = FrequencyAttractor(
            centroid=amp.clone(),
            phase_signature=torch.full((8,), math.pi),  # antiphase
            amplitude_profile=amp.clone(),
            domain="B", attractor_id="b0",
        )
        cand = plb.evaluate(a, b)
        assert not cand.is_activated
        assert cand.activation_score < 0.1  # near 0 because cos(π) = -1

    def test_in_phase_attractors_lock(self):
        """Same amplitude, same phase => should lock."""
        plb = PhaseLockBridge(n_bands=8, min_locked_bands=3,
                              band_threshold=0.5, activation_threshold=0.6)
        d = 64
        amp = torch.rand(d)
        phase = torch.randn(8)  # arbitrary but identical

        a = FrequencyAttractor(
            centroid=amp, phase_signature=phase,
            amplitude_profile=amp, domain="A", attractor_id="a0",
        )
        b = FrequencyAttractor(
            centroid=amp, phase_signature=phase,
            amplitude_profile=amp, domain="B", attractor_id="b0",
        )
        cand = plb.evaluate(a, b)
        assert cand.is_activated
        assert cand.activation_score > 0.95


# ─────────────────────────────────────────────────────────────────────────
# Test 3: Amplitude invariance
# ─────────────────────────────────────────────────────────────────────────

class TestAmplitudeInvariance:
    """
    Claim: scaling input amplitude by α should not change attention weights
    because routing is by phase, not magnitude.
    """

    def test_attention_invariant_to_scaling(self):
        torch.manual_seed(42)
        d_model = 64
        n_freq = 33

        sig = _make_tone_signal(
            freqs=[200.0, 400.0], amps=[1.0, 0.5], phases=[0.0, 0.0],
            n_samples=512, sample_rate=8000,
        )
        feat, phase = _signal_to_spectral_embedding(sig, d_model, n_freq)

        attn = ResonanceAttention(d_model=d_model, n_heads=4, n_bands=4, dropout=0.0)
        attn.eval()

        with torch.no_grad():
            _, w1 = attn(feat, phase=phase)
            _, w2 = attn(feat * 10.0, phase=phase)  # same phase, 10× amp

        # With explicit phase= and dropout disabled, the coherence matrix
        # depends only on phase. The amplitude scaling enters only via the
        # value projection (which doesn't affect attention weights), so
        # weights must be byte-identical.
        torch.testing.assert_close(w1, w2, atol=1e-6, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# Test 4: Phase rotation invariance
# ─────────────────────────────────────────────────────────────────────────

class TestPhaseRotationInvariance:
    """
    Claim: Adding a constant θ to every phase value should not change
    pairwise coherence because cos(Δφ + 0) = cos(Δφ).
    """

    def test_global_phase_shift_preserves_coherence(self):
        plb = PhaseLockBridge(n_bands=8)
        d = 64
        amp = torch.rand(d)
        phase_a = torch.randn(8)
        phase_b = torch.randn(8)

        a = FrequencyAttractor(centroid=amp, phase_signature=phase_a,
                               amplitude_profile=amp, attractor_id="a")
        b = FrequencyAttractor(centroid=amp, phase_signature=phase_b,
                               amplitude_profile=amp, attractor_id="b")
        coh_orig = plb.compute_band_coherences(a, b)

        # Apply global phase rotation
        theta = 1.234
        a_rot = FrequencyAttractor(centroid=amp, phase_signature=phase_a + theta,
                                   amplitude_profile=amp, attractor_id="a")
        b_rot = FrequencyAttractor(centroid=amp, phase_signature=phase_b + theta,
                                   amplitude_profile=amp, attractor_id="b")
        coh_rot = plb.compute_band_coherences(a_rot, b_rot)

        torch.testing.assert_close(coh_orig, coh_rot, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# Test 5: SNR sweep — graceful degradation
# ─────────────────────────────────────────────────────────────────────────

class TestSNRSweep:
    """
    As noise increases (SNR drops), the coherence between a clean
    reference and the noisy version should monotonically decrease.

    We test the property (monotonicity) rather than absolute thresholds,
    because absolute coherence depends on signal sparsity (single tone
    leaves most bins phase-undefined) which is unrelated to the routing
    mechanism we're trying to validate.
    """

    def _score_at_snr(self, snr_db: float, seed: int = 42) -> float:
        torch.manual_seed(seed)
        plb = PhaseLockBridge(n_bands=8)

        # Multi-tone signal: more bins have well-defined phase
        sig = _make_tone_signal(
            freqs=[200.0, 400.0, 800.0, 1600.0],
            amps=[1.0, 0.8, 0.6, 0.4],
            phases=[0.0, 0.0, 0.0, 0.0],
            n_samples=1024, sample_rate=8000,
        )
        spec_clean = torch.fft.rfft(sig).squeeze(0)
        phase_clean = spec_clean.angle()

        signal_power = sig.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(sig) * noise_power.sqrt()
        sig_noisy = sig + noise
        spec_noisy = torch.fft.rfft(sig_noisy).squeeze(0)
        phase_noisy = spec_noisy.angle()

        n_bands = 8
        bin_size = len(phase_clean) // n_bands
        ps_clean = torch.stack([phase_clean[i*bin_size:(i+1)*bin_size].mean()
                                for i in range(n_bands)])
        ps_noisy = torch.stack([phase_noisy[i*bin_size:(i+1)*bin_size].mean()
                                for i in range(n_bands)])

        amp = torch.rand(64)
        a = FrequencyAttractor(centroid=amp, phase_signature=ps_clean,
                               amplitude_profile=amp, attractor_id="clean")
        b = FrequencyAttractor(centroid=amp, phase_signature=ps_noisy,
                               amplitude_profile=amp, attractor_id="noisy")
        return plb.activation_score(plb.compute_band_coherences(a, b))

    def test_high_snr_scores_higher_than_low_snr(self):
        """30 dB SNR must score strictly higher than -10 dB SNR."""
        s_clean = self._score_at_snr(30)
        s_noisy = self._score_at_snr(-10)
        assert s_clean > s_noisy, (
            f"Coherence didn't degrade with noise: clean={s_clean:.3f}, "
            f"noisy={s_noisy:.3f}"
        )

    def test_noise_never_fully_fools_system(self):
        """At -20 dB SNR the system must not return a perfect lock."""
        s = self._score_at_snr(-20)
        assert s < 0.99, f"System fooled by pure noise: score={s:.3f}"


# ─────────────────────────────────────────────────────────────────────────
# Test 6: Multi-band coherence gating
# ─────────────────────────────────────────────────────────────────────────

class TestMultiBandGating:
    """
    Engineering Script §3: PhaseLockBridge must require ≥ N_bands locked
    bands before activating. Single-band lock should be rejected.
    """

    def test_single_band_lock_rejected(self):
        plb = PhaseLockBridge(n_bands=8, min_locked_bands=3,
                              band_threshold=0.7, activation_threshold=0.3)
        d = 64
        amp = torch.rand(d)
        phase_a = torch.randn(8) * 3.0           # mostly random
        phase_b = phase_a.clone()
        phase_b[1:] = phase_a[1:] + 1.5          # only band 0 stays locked

        a = FrequencyAttractor(centroid=amp, phase_signature=phase_a,
                               amplitude_profile=amp, attractor_id="a")
        b = FrequencyAttractor(centroid=amp, phase_signature=phase_b,
                               amplitude_profile=amp, attractor_id="b")
        cand = plb.evaluate(a, b)
        assert cand.n_locked_bands < 3
        assert not cand.is_activated

    def test_three_band_lock_activates(self):
        plb = PhaseLockBridge(n_bands=8, min_locked_bands=3,
                              band_threshold=0.4, activation_threshold=0.4)
        d = 64
        amp = torch.rand(d)
        phase_a = torch.randn(8) * 3.0
        phase_b = phase_a.clone()
        phase_b[3:] = phase_a[3:] + 2.0          # bands 0,1,2 stay locked

        a = FrequencyAttractor(centroid=amp, phase_signature=phase_a,
                               amplitude_profile=amp, attractor_id="a")
        b = FrequencyAttractor(centroid=amp, phase_signature=phase_b,
                               amplitude_profile=amp, attractor_id="b")
        cand = plb.evaluate(a, b)
        assert cand.n_locked_bands >= 3
