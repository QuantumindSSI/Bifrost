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
        """
        Initialize harmonic coherence detector.

        Parameters
        ----------
        n_freq : int
            Number of frequency bins (e.g., 257 for n_fft=512). Must be > 0.
        n_harmonics : int
            Number of harmonics to consider (e.g., 5 for f, 2f, 3f, 4f, 5f). Must be > 0.
        base_freq : float, optional
            Base frequency in Hz. If None, auto-detected from amplitude spectrum.
            Must be > 0 if provided.
        sample_rate : float
            Sample rate in Hz (default: 16000). Must be > 0.

        Raises
        ------
        ValueError
            If n_freq <= 0, n_harmonics <= 0, sample_rate <= 0, or base_freq <= 0.

        Complexity
        ----------
        O(1) - initialization only.

        Side Effects
        ------------
        Registers learnable parameter or buffer for base frequency.
        """
        if n_freq <= 0:
            raise ValueError(f"n_freq must be > 0, got {n_freq}")
        if n_harmonics <= 0:
            raise ValueError(f"n_harmonics must be > 0, got {n_harmonics}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        if base_freq is not None and base_freq <= 0:
            raise ValueError(f"base_freq must be > 0 if provided, got {base_freq}")

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
        """
        Get base frequency (learned or fixed).

        Returns
        -------
        float
            Base frequency in Hz.

        Complexity
        ----------
        O(1) - single item retrieval.

        Side Effects
        ------------
        None.
        """
        if self.learned_base_freq is not None:
            return self.learned_base_freq.item()
        return self.base_freq_buffer.item()

    def _get_harmonic_bins(self, n_fft: int) -> torch.Tensor:
        """
        Get frequency bin indices for harmonics of base frequency.

        Args
        ----
        n_fft : int
            FFT size used to compute frequency resolution. Must be > 0.

        Returns
        -------
        torch.Tensor
            Tensor of harmonic bin indices (n_harmonics,).

        Raises
        ------
        ValueError
            If n_fft <= 0.

        Complexity
        ----------
        O(n_harmonics) - iterates over harmonic indices.

        Side Effects
        ------------
        None.
        """
        if n_fft <= 0:
            raise ValueError(f"n_fft must be > 0, got {n_fft}")

        base = self._get_base_freq()
        freq_resolution = self.sample_rate / n_fft

        harmonic_bins = []
        for h in range(1, self.n_harmonics + 1):
            harmonic_freq = h * base
            bin_idx = int(round(harmonic_freq / freq_resolution))
            if 0 <= bin_idx < self.n_freq:
                harmonic_bins.append(bin_idx)

        return torch.tensor(harmonic_bins, dtype=torch.long)

    def _compute_harmonic_energy(
        self,
        amplitude: torch.Tensor,
        harmonic_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy at harmonic frequency bins.

        Args
        ----
        amplitude : torch.Tensor
            (B, T, n_freq) amplitude spectrum. Must be 3D and finite.
        harmonic_bins : torch.Tensor
            (n_harmonics,) bin indices. Must be within [0, n_freq).

        Returns
        -------
        torch.Tensor
            (B, T, n_harmonics) energy at harmonic bins.

        Raises
        ------
        ValueError
            If amplitude is not 3D, if harmonic_bins indices are out of bounds,
            or if amplitude contains NaN/Inf.

        Complexity
        ----------
        O(B * T * n_harmonics) - tensor indexing operation.

        Side Effects
        ------------
        None.
        """
        if amplitude.dim() != 3:
            raise ValueError(f"amplitude must be 3D (B, T, n_freq), got shape {amplitude.shape}")
        if not torch.isfinite(amplitude).all():
            raise ValueError("amplitude contains NaN or Inf values")
        if harmonic_bins.numel() == 0:
            raise ValueError("harmonic_bins is empty")
        if (harmonic_bins < 0).any() or (harmonic_bins >= amplitude.shape[-1]).any():
            raise ValueError(
                f"harmonic_bins indices out of bounds [0, {amplitude.shape[-1]}): {harmonic_bins.tolist()}"
            )

        # Extract energy at harmonic bins
        harmonic_energy = amplitude[..., harmonic_bins]  # (B, T, n_harmonics)
        return harmonic_energy

    def _compute_harmonic_coherence_matrix(
        self,
        harmonic_energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coherence matrix based on harmonic energy concentration.

        For each time step, compute the ratio of energy at harmonic bins vs total energy.
        High harmonic energy ratio = harmonic signal.
        Low harmonic energy ratio = inharmonic signal.

        Args
        ----
        harmonic_energy : torch.Tensor
            (B, T, n_harmonics) energy at harmonic bins. Must be 3D and finite.

        Returns
        -------
        torch.Tensor
            (B, 1, T, T) coherence matrix.

        Raises
        ------
        ValueError
            If harmonic_energy is not 3D or contains NaN/Inf.

        Complexity
        ----------
        O(B * T^2) - batch matrix multiplication for outer product.

        Side Effects
        ------------
        None.
        """
        if harmonic_energy.dim() != 3:
            raise ValueError(f"harmonic_energy must be 3D (B, T, n_harmonics), got shape {harmonic_energy.shape}")
        if not torch.isfinite(harmonic_energy).all():
            raise ValueError("harmonic_energy contains NaN or Inf values")

        B, T, H = harmonic_energy.shape

        # Compute harmonic energy ratio: sum of harmonic energy / total energy
        # Since we only have harmonic energy, use the mean as the harmonic strength
        harmonic_strength = harmonic_energy.mean(dim=-1, keepdim=True)  # (B, T, 1)

        # Compute coherence as outer product of harmonic strength
        # High harmonic strength at both time indices = high coherence
        coherence = torch.bmm(
            harmonic_strength,  # (B, T, 1)
            harmonic_strength.transpose(1, 2),  # (B, 1, T)
        )  # (B, T, T)
        coherence = coherence.unsqueeze(1)  # (B, 1, T, T)

        return coherence

    def forward(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        n_fft: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute harmonic coherence matrix.

        Args
        ----
        amplitude : torch.Tensor
            (B, T, n_freq) amplitude spectrum. Must be 3D and finite.
        phase : torch.Tensor
            (B, T, n_freq) phase spectrum (unused, kept for interface compatibility).
            Must match amplitude shape.
        n_fft : int
            FFT size for frequency resolution. Must be > 0.

        Returns
        -------
        torch.Tensor
            Dummy output (same as amplitude, for interface compatibility).
        torch.Tensor
            (B, 1, T, T) harmonic coherence matrix.

        Raises
        ------
        ValueError
            If amplitude or phase have invalid shapes, contain NaN/Inf,
            or if n_fft <= 0.

        Complexity
        ----------
        O(n_harmonics + B * T * n_harmonics + B * T^2) - bin computation + energy extraction + coherence.

        Side Effects
        ------------
        None.
        """
        if amplitude.dim() != 3:
            raise ValueError(f"amplitude must be 3D (B, T, n_freq), got shape {amplitude.shape}")
        if phase.shape != amplitude.shape:
            raise ValueError(f"phase shape {phase.shape} must match amplitude shape {amplitude.shape}")
        if not torch.isfinite(amplitude).all():
            raise ValueError("amplitude contains NaN or Inf values")
        if not torch.isfinite(phase).all():
            raise ValueError("phase contains NaN or Inf values")

        # Get harmonic frequency bins
        harmonic_bins = self._get_harmonic_bins(n_fft)

        # If no valid harmonic bins, return uniform coherence
        if len(harmonic_bins) == 0:
            B, T, _ = amplitude.shape
            uniform_coh = torch.ones(B, 1, T, T, device=amplitude.device) / T
            return amplitude, uniform_coh

        # Compute energy at harmonic bins
        harmonic_energy = self._compute_harmonic_energy(amplitude, harmonic_bins)

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
