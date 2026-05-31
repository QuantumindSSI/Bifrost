"""
Harmonic Coherence Detector — energy-based harmonic structure detection.

Computes coherence based on energy concentration at harmonic frequency bins,
not generic phase coherence. This distinguishes harmonic signals (energy at
440Hz, 880Hz, 1320Hz, etc.) from inharmonic signals (energy spread across
non-harmonic frequencies).

Key difference from ResonanceAttention:
- ResonanceAttention: Phase coherence (phase similarity across time/frequency)
- HarmonicCoherence: Energy concentration at harmonic frequency bins
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HarmonicCoherenceDetector(nn.Module):
    """
    Detects harmonic structure by measuring energy at harmonic frequency bins.

    Parameters
    ----------
    n_freq : int
        Number of frequency bins (e.g., 257 for n_fft=512).
    n_harmonics : int
        Number of harmonics to consider (e.g., 5 for f, 2f, 3f, 4f, 5f).
    base_freq : float, optional
        Base frequency in Hz. If None, auto-detected from amplitude spectrum.
    sample_rate : float
        Sample rate in Hz (default: 16000).
    """

    def __init__(
        self,
        n_freq: int = 257,
        n_harmonics: int = 5,
        base_freq: Optional[float] = None,
        sample_rate: float = 16000.0,
    ) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.n_harmonics = n_harmonics
        self.base_freq = base_freq
        self.sample_rate = sample_rate

        # Learnable base frequency if not provided
        if base_freq is None:
            self.learned_base_freq = nn.Parameter(torch.tensor(440.0))
        else:
            self.register_buffer('base_freq_buffer', torch.tensor(base_freq))
            self.learned_base_freq = None

    def _get_base_freq(self) -> float:
        """Get base frequency (learned or fixed)."""
        if self.learned_base_freq is not None:
            return self.learned_base_freq.item()
        return self.base_freq_buffer.item()

    def _get_harmonic_bins(self, n_fft: int) -> torch.Tensor:
        """
        Get frequency bin indices for harmonics of base frequency.

        Args:
            n_fft: FFT size used to compute frequency resolution.

        Returns:
            Tensor of harmonic bin indices (n_harmonics,).
        """
        base = self._get_base_freq()
        freq_resolution = self.sample_rate / n_fft

        harmonic_bins = []
        for h in range(1, self.n_harmonics + 1):
            harmonic_freq = h * base
            bin_idx = int(round(harmonic_freq / freq_resolution))
            if bin_idx < self.n_freq:
                harmonic_bins.append(bin_idx)

        return torch.tensor(harmonic_bins, dtype=torch.long)

    def _compute_harmonic_energy(
        self,
        amplitude: torch.Tensor,
        harmonic_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy at harmonic frequency bins.

        Args:
            amplitude: (B, T, n_freq) amplitude spectrum.
            harmonic_bins: (n_harmonics,) bin indices.

        Returns:
            (B, T, n_harmonics) energy at harmonic bins.
        """
        # Extract energy at harmonic bins
        harmonic_energy = amplitude[..., harmonic_bins]  # (B, T, n_harmonics)
        return harmonic_energy

    def _compute_harmonic_coherence_matrix(
        self,
        harmonic_energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coherence matrix based on harmonic energy similarity.

        For each time step, compute similarity of harmonic energy profiles.
        High similarity = both have energy at same harmonics = harmonic signal.

        Args:
            harmonic_energy: (B, T, n_harmonics) energy at harmonic bins.

        Returns:
            (B, 1, T, T) coherence matrix.
        """
        B, T, H = harmonic_energy.shape

        # Normalize energy profiles per time step
        energy_norm = F.normalize(harmonic_energy, p=2, dim=-1)  # (B, T, H)

        # Compute cosine similarity between all time pairs using matmul
        # energy_norm: (B, T, H) -> (B, T, H) @ (B, H, T) -> (B, T, T)
        similarity = torch.matmul(energy_norm, energy_norm.transpose(1, 2))  # (B, T, T)
        similarity = similarity.unsqueeze(1)  # (B, 1, T, T)

        return similarity

    def forward(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        n_fft: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute harmonic coherence matrix.

        Args:
            amplitude: (B, T, n_freq) amplitude spectrum.
            phase: (B, T, n_freq) phase spectrum (unused, kept for interface).
            n_fft: FFT size for frequency resolution.

        Returns:
            bound: Dummy output (same as amplitude, for interface compatibility).
            coherence: (B, 1, T, T) harmonic coherence matrix.
        """
        # Get harmonic frequency bins
        harmonic_bins = self._get_harmonic_bins(n_fft)

        # If no valid harmonic bins, return uniform coherence
        if len(harmonic_bins) == 0:
            B, T, _ = amplitude.shape
            uniform_coh = torch.ones(B, 1, T, T, device=amplitude.device) / T
            return amplitude, uniform_coh

        # Compute energy at harmonic bins
        harmonic_energy = self._compute_harmonic_energy(amplitude, harmonic_bins)

        # DEBUG: Print harmonic energy stats
        print(f"[DEBUG_HARM] bins={harmonic_bins.tolist()}, energy_mean={harmonic_energy.mean():.4f}, "
              f"energy_std={harmonic_energy.std():.4f}, energy_max={harmonic_energy.max():.4f}")

        # Compute harmonic coherence matrix
        coherence = self._compute_harmonic_coherence_matrix(harmonic_energy)

        # Return dummy bound (same as amplitude) for interface compatibility
        # The actual binding output is computed in SpectralBinding
        return amplitude, coherence


def demo_harmonic_coherence():
    """Demonstrate harmonic coherence detection."""
    print("=" * 60)
    print("HARMONIC COHERENCE DETECTOR DEMO")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_freq = 257
    n_fft = 512
    sample_rate = 16000.0
    base_freq = 440.0

    detector = HarmonicCoherenceDetector(
        n_freq=n_freq,
        n_harmonics=5,
        base_freq=base_freq,
        sample_rate=sample_rate,
    ).to(device)

    # Generate harmonic signal (energy at 440, 880, 1320, 1760, 2200 Hz)
    B, T = 1, 32
    harmonic_bins = detector._get_harmonic_bins(n_fft)
    print(f"Harmonic bins (indices): {harmonic_bins.tolist()}")

    harmonic_amp = torch.zeros(B, T, n_freq, device=device)
    harmonic_amp[:, :, harmonic_bins] = 1.0  # Energy at harmonics

    # Generate inharmonic signal (energy at random non-harmonic bins)
    inharmonic_amp = torch.zeros(B, T, n_freq, device=device)
    random_bins = torch.randint(0, n_freq, (len(harmonic_bins),), device=device)
    inharmonic_amp[:, :, random_bins] = 1.0

    # Compute coherence
    _, harmonic_coh = detector(harmonic_amp, torch.zeros_like(harmonic_amp), n_fft)
    _, inharmonic_coh = detector(inharmonic_amp, torch.zeros_like(inharmonic_amp), n_fft)

    print(f"\nHarmonic signal coherence: mean={harmonic_coh.mean():.4f}, std={harmonic_coh.std():.4f}")
    print(f"Inharmonic signal coherence: mean={inharmonic_coh.mean():.4f}, std={inharmonic_coh.std():.4f}")

    # Harmonic signals should have higher coherence (energy concentrated at harmonics)
    # Inharmonic signals should have lower coherence (energy spread randomly)
    print(f"\nDiscrimination: {harmonic_coh.mean() - inharmonic_coh.mean():.4f}")

    print("=" * 60)


if __name__ == "__main__":
    demo_harmonic_coherence()
