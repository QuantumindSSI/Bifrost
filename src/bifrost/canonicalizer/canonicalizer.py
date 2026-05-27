"""
SpectralCanonicalizer — convert raw ingested data into canonical SpectralTensor form.

Responsibilities:
    1. Input normalisation — map arbitrary numeric arrays to float32.
    2. Frequency-domain projection — FFT of the signal to obtain
       amplitude + phase decomposition in L² space.
    3. Confidence / uncertainty representation — initial uniform uncertainty
       that downstream stages can refine.

The module is implemented as a ``torch.nn.Module`` so that any learnable
pre-processing (e.g. a learnable low-pass filter) can be added later
while keeping the same forward interface.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..spectral_tensor import SpectralTensor


class SpectralCanonicalizer(nn.Module):
    """
    Canonicalization stage: raw signal → SpectralTensor.

    Supports both 1D (temporal) and 2D (spatial) FFT modes for audio
    and image modalities respectively.

    Parameters
    ----------
    n_fft : int
        FFT size.  Defaults to 1024.  When the input is shorter it is
        zero-padded; when longer it is chunked and averaged.
    hop_length : int | None
        Hop length for STFT-style windowed FFT.  ``None`` → ``n_fft // 4``.
    normalize_input : bool
        If True, z-score normalise the input before FFT.
    initial_uncertainty : float
        Constant value for the initial per-element uncertainty field.
    preserve_frames : bool
        If True, preserve STFT frames for sequence attention.
    use_2d_fft : bool
        If True, use 2D FFT for spatial data (images). If False, use 1D FFT.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        normalize_input: bool = True,
        initial_uncertainty: float = 1.0,
        preserve_frames: bool = False,
        use_2d_fft: bool = False,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.normalize_input = normalize_input
        self.initial_uncertainty = initial_uncertainty
        self.preserve_frames = preserve_frames
        self.use_2d_fft = use_2d_fft

        # Hann window (non-learnable but registered so it follows .to())
        self.register_buffer("window", torch.hann_window(n_fft))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpectralTensor:
        """
        Canonicalise a raw signal tensor into a SpectralTensor.

        Parameters
        ----------
        signal : torch.Tensor
            1-D ``(samples,)`` or 2-D ``(channels, samples)`` raw waveform
            (time-domain), or 2-D/3-D spatial data for 2D FFT.
            Also accepts a batch dimension.
        metadata : dict, optional
            Provenance metadata from the ingest layer.

        Returns
        -------
        SpectralTensor
            With fields ``amplitude``, ``phase``, ``scale``, ``uncertainty``
            all shaped ``(…, n_freq_bins)`` where ``n_freq_bins = n_fft // 2 + 1``.
        """
        metadata = metadata or {}

        # Auto-detect 2D spatial data from metadata
        # Only use 2D FFT if explicitly requested AND we have 2D spatial layout
        # Bridge-flattened images have "num_samples" and 2D tensor (channels, samples)
        # True 2D spatial data comes from raw images before bridge flattening
        is_bridge_flattened = "num_samples" in metadata or "original_spatial" in metadata
        is_spatial = (
            self.use_2d_fft
            and not is_bridge_flattened  # Don't use 2D FFT on already-flattened data
            and (metadata.get("is_spatial", False) or "height" in metadata)
        )

        # --- ensure torch float32 and ≥ 2-D ---------------------------------
        signal = self._prepare_input(signal)

        # --- optional z-score normalisation ----------------------------------
        if self.normalize_input:
            signal = self._z_normalize(signal)

        # --- frequency-domain projection -------------------------------------
        if is_spatial and signal.ndim >= 2:
            amplitude, phase = self._compute_fft_2d(signal)
        else:
            amplitude, phase = self._compute_fft(signal)

        # --- scale: linearly spaced frequency bin centres (Hz) ---------------
        sample_rate = metadata.get("sample_rate", 1.0)
        n_freq = amplitude.shape[-1]
        scale = torch.linspace(0.0, sample_rate / 2.0, n_freq, device=amplitude.device)
        scale = scale.expand_as(amplitude)

        # --- initial uniform uncertainty -------------------------------------
        uncertainty = torch.full_like(amplitude, self.initial_uncertainty)

        # --- package ---
        st = SpectralTensor(
            amplitude=amplitude,
            phase=phase,
            scale=scale,
            uncertainty=uncertainty,
            metadata={
                **metadata,
                "stage": "canonicalize",
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "normalize_input": self.normalize_input,
            },
        )
        st.validate()
        return st

    def from_numpy(
        self,
        array: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpectralTensor:
        """Convenience wrapper that accepts a NumPy array (from ingest)."""
        tensor = torch.from_numpy(array.astype(np.float32))
        return self.forward(tensor, metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_input(signal: torch.Tensor) -> torch.Tensor:
        """Cast to float32 and ensure at least 2-D (channels, samples)."""
        if signal.dtype != torch.float32:
            signal = signal.float()

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)  # (1, samples)
        return signal

    @staticmethod
    def _z_normalize(signal: torch.Tensor) -> torch.Tensor:
        """Per-channel zero-mean unit-variance normalisation."""
        mean = signal.mean(dim=-1, keepdim=True)
        std = signal.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (signal - mean) / std

    def _compute_fft(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Windowed FFT → (amplitude, phase).

        For signals shorter than ``n_fft`` the signal is zero-padded.
        For longer signals an STFT is computed. If ``preserve_frames=True``,
        returns frames as sequence (…, n_frames, n_freq). Otherwise averages
        across frames to yield a single spectral snapshot.
        """
        n_samples = signal.shape[-1]

        if n_samples <= self.n_fft:
            # Zero-pad and do a single FFT
            padded = torch.nn.functional.pad(signal, (0, self.n_fft - n_samples))
            windowed = padded * self.window
            spectrum = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)
        else:
            # STFT: unfold into overlapping frames → FFT
            frames = signal.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
            # frames: (…, n_frames, n_fft)
            windowed = frames * self.window
            spectrum = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)
            # (…, n_frames, n_freq)
            if not self.preserve_frames:
                # Average magnitude and circular-mean phase across frames
                spectrum = spectrum.mean(dim=-2)

        amplitude = spectrum.abs()
        phase = spectrum.angle()

        # Normalise amplitude to [0, 1] per-channel
        amp_max = amplitude.amax(dim=-1, keepdim=True).clamp(min=1e-8)
        amplitude = amplitude / amp_max

        return amplitude, phase

    def _compute_fft_2d(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        2D FFT for spatial data (images) → radially-flattened spectra.

        Input: (…, H, W) spatial data (e.g., from image channels)
        Output: (…, n_freq) where n_freq = n_fft // 2 + 1

        Strategy:
            1. Apply 2D FFT to get spatial frequency representation.
            2. Convert to polar coordinates (magnitude vs frequency radius).
            3. Radially average to get 1D spectrum per channel.
            4. Truncate/pad to n_freq bins.
        """
        # signal shape: (…, H, W)
        *batch_dims, h, w = signal.shape

        # Apply 2D FFT
        spectrum_2d = torch.fft.fft2(signal, s=(self.n_fft, self.n_fft), dim=(-2, -1))
        spectrum_2d = torch.fft.fftshift(spectrum_2d, dim=(-2, -1))

        # Compute magnitude
        amplitude_2d = spectrum_2d.abs()

        # Create radial frequency bins
        h_bins = torch.arange(self.n_fft, device=signal.device) - self.n_fft // 2
        w_bins = torch.arange(self.n_fft, device=signal.device) - self.n_fft // 2
        yy, xx = torch.meshgrid(h_bins, w_bins, indexing='ij')
        radius = torch.sqrt(xx**2 + yy**2).long()

        n_freq = self.n_fft // 2 + 1

        # Clamp radius indices to valid bin range [0, n_freq-1]
        radius_clamped = radius.clamp(0, n_freq - 1)  # (n_fft, n_fft)
        flat_radius = radius_clamped.reshape(-1)  # (n_fft*n_fft,)

        # Flatten spatial dims for scatter: amplitude_2d is (..., n_fft, n_fft)
        flat_shape = list(amplitude_2d.shape[:-2]) + [self.n_fft * self.n_fft]
        amplitude_flat = amplitude_2d.reshape(flat_shape)  # (..., n_fft*n_fft)

        # Count pixels per radial bin (same for all batch elements)
        bin_counts = torch.zeros(n_freq, device=signal.device)
        bin_counts.scatter_add_(0, flat_radius, torch.ones_like(flat_radius, dtype=torch.float32))
        bin_counts = bin_counts.clamp(min=1.0)

        # Scatter-sum amplitude per radial bin for each batch element
        radial_amplitude = torch.zeros(*batch_dims, n_freq, device=signal.device)
        idx = flat_radius.unsqueeze(0).expand(amplitude_flat.shape[:-1] + (flat_radius.shape[0],))
        radial_amplitude.scatter_add_(-1, idx, amplitude_flat)
        radial_amplitude = radial_amplitude / bin_counts  # mean per bin

        # Phase: circular mean per radial bin to preserve per-band spatial phase information
        # spectrum_2d: (..., n_fft, n_fft) complex
        phase_2d = spectrum_2d.angle()  # (..., n_fft, n_fft)
        phase_flat = phase_2d.reshape(flat_shape)  # (..., n_fft*n_fft)

        sin_sum = torch.zeros(*batch_dims, n_freq, device=signal.device)
        cos_sum = torch.zeros(*batch_dims, n_freq, device=signal.device)
        sin_sum.scatter_add_(-1, idx, torch.sin(phase_flat))
        cos_sum.scatter_add_(-1, idx, torch.cos(phase_flat))
        phase = torch.atan2(sin_sum, cos_sum)  # circular mean phase per bin (..., n_freq)

        # Normalize amplitude
        amp_max = radial_amplitude.amax(dim=-1, keepdim=True).clamp(min=1e-8)
        radial_amplitude = radial_amplitude / amp_max

        return radial_amplitude, phase
