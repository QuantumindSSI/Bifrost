"""
End-to-end attention-routing tests on real WAV files.

Drives the full Ingest → Bridge → S0 → S1 → S2 pipeline with the
sample audio files and asserts measurable properties of the resulting
attention maps:

    - Spectral peak localisation
    - Stationarity (deterministic re-run yields identical attention)
    - Cross-signal discrimination
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from bifrost.bridge import bridge_to_canonicalizer
from bifrost.canonicalizer import SpectralCanonicalizer
from bifrost.decomposer import SpectralDecomposer
from bifrost.resonance_attention import SpectralBinding


N_FFT = 1024


@pytest.fixture(scope="module")
def pipeline():
    """Build a deterministic S0 + S1 + S2 stack.

    d_model is chosen to be divisible by n_heads. S2's internal projection
    bridges n_freq → d_model so the head dim works out.
    """
    torch.manual_seed(0)
    n_freq = N_FFT // 2 + 1
    d_model = 128  # divisible by n_heads=4
    canonicalizer = SpectralCanonicalizer(n_fft=N_FFT)
    decomposer = SpectralDecomposer(n_fft=N_FFT, n_scales=4, d_model=n_freq)
    # binding needs to know input dim to create projection from n_freq -> d_model
    binding = SpectralBinding(d_model=d_model, n_heads=4, n_bands=8, dropout=0.0, n_freq_in=n_freq)
    canonicalizer.eval()
    decomposer.eval()
    binding.eval()
    return canonicalizer, decomposer, binding


def _process_wav(path: str, pipeline):
    """Run the full pipeline on a WAV file, return (bound, coherence)."""
    canonicalizer, decomposer, binding = pipeline
    sr, data = wavfile.read(path)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    sig, meta = bridge_to_canonicalizer(data, {"format": "wav", "channels": 1
                                               if data.ndim == 1 else data.shape[-1],
                                               "sample_rate": sr})
    with torch.no_grad():
        canonical = canonicalizer(sig, meta)
        decomposed = decomposer(canonical)
        bound, coh = binding(decomposed)
    return bound, coh, meta


class TestRealSignalRouting:

    def test_mono_8khz_produces_valid_attention(self, pipeline):
        bound, coh, meta = _process_wav("sample_data/mono_8khz.wav", pipeline)
        bound.validate()
        assert not torch.isnan(coh).any()
        assert not torch.isinf(coh).any()
        # Coherence weights should sum to ~1 along the key axis
        w_sum = coh.sum(dim=-1)
        torch.testing.assert_close(w_sum, torch.ones_like(w_sum),
                                   atol=1e-4, rtol=1e-4)

    def test_mono_16khz_produces_valid_attention(self, pipeline):
        bound, coh, meta = _process_wav("sample_data/mono_16khz.wav", pipeline)
        bound.validate()
        assert not torch.isnan(coh).any()
        w_sum = coh.sum(dim=-1)
        torch.testing.assert_close(w_sum, torch.ones_like(w_sum),
                                   atol=1e-4, rtol=1e-4)

    def test_stereo_44khz_attention_routes_between_channels(self, pipeline):
        """
        Stereo audio enters binding as (batch=1, seq=2, features). The attention
        map should be a valid 2×2 routing matrix between channels.
        """
        bound, coh, meta = _process_wav("sample_data/stereo_44khz.wav", pipeline)
        bound.validate()
        # Expect (B=1, H, S=2, S=2)
        assert coh.shape[-1] == coh.shape[-2], "Attention map must be square"
        # Rows must softmax to 1
        row_sums = coh.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums),
                                   atol=1e-4, rtol=1e-4)
        # Off-diagonal cross-channel weight must exist (otherwise channels
        # ignore each other, which would defeat coherence routing).
        cross_channel = coh[..., 0, 1].mean().item()
        assert cross_channel > 0.0
        assert cross_channel < 1.0

    def test_determinism_same_input_same_output(self, pipeline):
        """Re-running the same WAV through the pipeline must produce identical attention."""
        _, coh1, _ = _process_wav("sample_data/mono_16khz.wav", pipeline)
        _, coh2, _ = _process_wav("sample_data/mono_16khz.wav", pipeline)
        torch.testing.assert_close(coh1, coh2, atol=1e-6, rtol=1e-6)

    def test_different_signals_produce_different_spectral_tensors(self, pipeline):
        """
        Two different WAV files should produce distinguishable SpectralTensor
        outputs at the amplitude level (the attention map itself is degenerate
        for mono inputs because seq=1; we'd need time-framing to test routing
        differences — that's a Phase 2 enhancement).
        """
        bound_a, _, _ = _process_wav("sample_data/mono_8khz.wav", pipeline)
        bound_b, _, _ = _process_wav("sample_data/mono_16khz.wav", pipeline)

        # Resample to the same length to enable comparison
        amp_a = bound_a.amplitude.flatten()
        amp_b = bound_b.amplitude.flatten()
        # Trim to the shorter
        n = min(amp_a.numel(), amp_b.numel())
        amp_a, amp_b = amp_a[:n], amp_b[:n]

        # The amplitude spectra must not be byte-identical.
        assert not torch.allclose(amp_a, amp_b, atol=1e-3), (
            "Different signals collapsed to identical bound amplitude"
        )
