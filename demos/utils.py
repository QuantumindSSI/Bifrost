"""
Shared utilities for atomic demos.

    - Hilbert-transform anti-phase signal generation
    - Synthetic harmonic-tone corpus
    - Signal → (feat, phase) embedding for attention input
    - L1 attention-map distance
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from scipy.signal import hilbert


# ─── Signal generation ────────────────────────────────────────────────────

def make_tone(
    freqs: list[float],
    amps: list[float],
    phases: list[float],
    n_samples: int = 4096,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Multi-tone signal of shape (n_samples,)."""
    t = torch.linspace(0, n_samples / sample_rate, n_samples, dtype=torch.float32)
    sig = torch.zeros(n_samples, dtype=torch.float32)
    for f, a, p in zip(freqs, amps, phases):
        sig = sig + a * torch.sin(2 * math.pi * f * t + p)
    return sig


def make_harmonic_signal(
    f0: float,
    n_harmonics: int = 5,
    amp_decay: float = 0.7,
    n_samples: int = 4096,
    sample_rate: int = 16000,
    phase_jitter: float = 0.0,
    seed: int = 0,
) -> Tuple[torch.Tensor, float]:
    """
    A harmonic stack at fundamental f0 with n_harmonics components.

    Returns (signal, f0). Each harmonic k has amplitude amp_decay^(k-1).
    phase_jitter > 0 adds random phase per harmonic to make the test
    less degenerate.
    """
    rng = np.random.default_rng(seed)
    freqs = [f0 * (k + 1) for k in range(n_harmonics)]
    amps = [amp_decay ** k for k in range(n_harmonics)]
    phases = [float(rng.uniform(-phase_jitter, phase_jitter))
              for _ in range(n_harmonics)]
    sig = make_tone(freqs, amps, phases, n_samples, sample_rate)
    return sig, f0


def hilbert_antiphase(sig: torch.Tensor) -> torch.Tensor:
    """
    Produce x'(t) with same amplitude spectrum but global phase
    shifted by π. Mathematically: x'(t) = -x(t) preserves |X(f)| and
    rotates ∠X by π for every bin. That's the cleanest anti-phase pair.

    We also offer a Hilbert-based version below that swaps phase
    asymmetrically — useful for harder tests.
    """
    return -sig


def hilbert_phase_shifted(sig: torch.Tensor, shift: float = math.pi) -> torch.Tensor:
    """
    Construct a signal with the same amplitude spectrum but rotated
    phase by ``shift`` radians on the analytic signal. Uses scipy.hilbert.

    For shift = π this is equivalent to negation up to numerical error,
    but the analytic-signal path lets us test arbitrary shifts.
    """
    arr = sig.detach().cpu().numpy().astype(np.float64)
    analytic = hilbert(arr)                       # x(t) + i·H{x}(t)
    rotated = analytic * np.exp(1j * shift)       # rotate phase
    out = rotated.real
    return torch.from_numpy(out.astype(np.float32))


# ─── Spectral embedding for attention input ───────────────────────────────

def signal_to_spectral_input(
    sig: torch.Tensor,
    d_model: int,
    n_fft: int = 256,
    n_frames: int = 16,
    hop: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a 1-D signal to (feat, phase) of shape (1, n_frames, d_model)
    via STFT framing — exactly the layout ResonanceAttention expects.

    Returns
    -------
    feat  : (1, n_frames, d_model) amplitude features
    phase : (1, n_frames, d_model) phase features
    """
    if hop is None:
        hop = max(1, (len(sig) - n_fft) // max(1, n_frames - 1))

    # Window each frame and FFT
    frames = []
    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        if end > len(sig):
            # Pad with zeros at the end
            frame = torch.zeros(n_fft, dtype=sig.dtype)
            available = len(sig) - start
            if available > 0:
                frame[:available] = sig[start:start + available]
        else:
            frame = sig[start:end]
        frames.append(frame)

    frames = torch.stack(frames)                          # (n_frames, n_fft)
    spec = torch.fft.rfft(frames, dim=-1)                 # (n_frames, n_fft//2+1)
    amp = spec.abs()
    phase = spec.angle()

    # Resize feature dim to d_model
    n_freq = amp.shape[-1]
    if n_freq != d_model:
        amp = torch.nn.functional.interpolate(
            amp.unsqueeze(0), size=d_model, mode="linear", align_corners=False
        ).squeeze(0)
        phase = torch.nn.functional.interpolate(
            phase.unsqueeze(0), size=d_model, mode="linear", align_corners=False
        ).squeeze(0)

    return amp.unsqueeze(0), phase.unsqueeze(0)   # add batch dim


# ─── Distance helpers ─────────────────────────────────────────────────────

def attention_l1_distance(w1: torch.Tensor, w2: torch.Tensor) -> float:
    """Mean L1 distance between two attention maps (averaged over heads)."""
    if w1.dim() == 4:
        w1 = w1.mean(dim=1)
    if w2.dim() == 4:
        w2 = w2.mean(dim=1)
    return (w1 - w2).abs().mean().item()


def attention_kl_divergence(w1: torch.Tensor, w2: torch.Tensor,
                            eps: float = 1e-8) -> float:
    """Mean KL(w1 || w2) — sharper distance for distributions."""
    if w1.dim() == 4:
        w1 = w1.mean(dim=1)
    if w2.dim() == 4:
        w2 = w2.mean(dim=1)
    return ((w1 + eps) * ((w1 + eps).log() - (w2 + eps).log())).sum(-1).mean().item()
