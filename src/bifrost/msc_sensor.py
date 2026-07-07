"""
Multi-Scale Structural Coherence (MSC) — Sensor instance.

Implements wavelet coherence as the sensor instance of the MSC framework.
Wavelet coherence measures cross-channel phase relationships at multiple
time scales for activity recognition and sensor fusion.

This is the sensor analog of CBMPC (audio) and PhaseCongruency (image).
All three measure phase coherence across scales, but in different domains:
    - CBMPC:        cross-band phase coherence in modulation frequency domain
    - PhaseCongruency: cross-scale phase coherence in spatial frequency domain
    - WaveletCoherence: cross-channel phase coherence in time-scale domain

Mathematical formulation (Grinsted et al. 2004):
    Cross-wavelet transform: W_ij(a, t) = W_i(a, t) * conj(W_j(a, t))
    Wavelet coherence: R^2(a, t) = |S(s^-1 * W_ij)|^2 / (S(s^-1 * |W_i|^2) * S(s^-1 * |W_j|^2))
    Phase angle: arctan(Im(S(s^-1 * W_ij)) / Re(S(s^-1 * W_ij)))

where S is a smoothing operator and s is the scale.

References:
    - Grinsted, Moore & Jevrejeva (2004): wavelet coherence for geophysical
      time series. Nonlinear Processes in Geophysics, 11, 561-566.
    - Torrence & Webster (1999): interdecadal changes in the ENSO-monsoon
      system. Journal of Climate, 12(8), 2679-2690.
    - Maraun & Kurths (2004): cross wavelet analysis.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.spectral_utils import EPS, circular_mean


class WaveletCoherenceExtractor(nn.Module):
    """Sensor MSC instance: Cross-channel wavelet coherence.

    Computes wavelet coherence between all pairs of sensor channels at
    multiple time scales, producing a fixed-dimensional feature vector
    capturing the structural coherence of multi-channel sensor data.

    Parameters
    ----------
    n_scales : int
        Number of wavelet scales (default: 12).
    n_channels : int
        Number of sensor channels (default: 6 for UCI HAR).
    sample_rate : float
        Sensor sample rate in Hz (default: 50 for UCI HAR).
    wavelet : str
        Wavelet type: 'morlet' or 'paul' (default: 'morlet').
    smoothing_window : int
        Smoothing window size for coherence computation (default: 5).
    """

    def __init__(
        self,
        n_scales: int = 12,
        n_channels: int = 6,
        sample_rate: float = 50.0,
        wavelet: str = "morlet",
        smoothing_window: int = 5,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.wavelet_type = wavelet
        self.smoothing_window = smoothing_window

        # Pre-compute scales (dyadic spacing)
        scales = torch.tensor([2 ** (i / 2) for i in range(n_scales)],
                              dtype=torch.float32)
        self.register_buffer("scales", scales)

        # Morlet wavelet parameter
        self.omega0 = 6.0  # Center frequency for Morlet

        # Number of channel pairs
        self.n_pairs = n_channels * (n_channels - 1) // 2

        # Feature dimension:
        # - Mean coherence per scale per pair: n_scales * n_pairs
        # - Mean phase angle per scale per pair: n_scales * n_pairs
        # - Global coherence stats: 4 (mean, std, max, min)
        # - Per-scale mean coherence: n_scales
        # - Per-pair mean coherence: n_pairs
        self.feature_dim = (
            2 * n_scales * self.n_pairs  # coherence + phase per pair per scale
            + 4  # global stats
            + n_scales  # per-scale means
            + self.n_pairs  # per-pair means
        )

    def _morlet_cwt(self, signal: torch.Tensor) -> torch.Tensor:
        """Continuous wavelet transform using Morlet wavelet.

        Parameters
        ----------
        signal : torch.Tensor
            Shape (B, C, T) — batch of multi-channel signals.

        Returns
        -------
        torch.Tensor
            Complex CWT coefficients. Shape (B, C, n_scales, T).
        """
        B, C, T = signal.shape
        device = signal.device

        # Compute FFT of signal
        signal_fft = torch.fft.fft(signal, dim=-1)  # (B, C, T)

        # Compute wavelet Fourier domain representation for each scale
        # Morlet wavelet in Fourier domain:
        # psi_hat(s*omega) = pi^(-1/4) * exp(-(s*omega - omega0)^2 / 2) for omega > 0
        omega = torch.fft.fftfreq(T, d=1.0 / self.sample_rate).to(device)  # (T,)
        omega = omega * 2 * torch.pi  # angular frequency

        cwt = torch.zeros(B, C, self.n_scales, T, dtype=torch.complex64,
                          device=device)

        for s_idx, s in enumerate(self.scales):
            # Wavelet in Fourier domain
            s_val = s.item()
            psi_hat = torch.zeros(T, dtype=torch.complex64, device=device)
            pos_mask = omega > 0
            # Compute real-valued envelope, then convert to complex
            envelope = torch.exp(-((s_val * omega[pos_mask] - self.omega0) ** 2) / 2) \
                       * (s_val / torch.pi) ** 0.25
            psi_hat[pos_mask] = envelope.to(torch.complex64)

            # Convolve: multiply in Fourier domain
            # signal_fft: (B, C, T), psi_hat: (T,)
            cwt[:, :, s_idx, :] = torch.fft.ifft(
                signal_fft * psi_hat.unsqueeze(0).unsqueeze(0), dim=-1
            )

        return cwt

    def _smooth(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Smoothing operator S for coherence computation.

        Smooths along the time dimension using a moving average.
        """
        if window <= 1:
            return x
        # 1D average pooling along time
        # x shape: (..., T)
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        else:
            # Flatten leading dims
            x = x.reshape(-1, 1, orig_shape[-1])

        # Reflect padding to preserve length
        pad = window // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        x = F.avg_pool1d(x, kernel_size=window, stride=1, padding=0)

        x = x.reshape(orig_shape)
        return x

    def _wavelet_coherence(
        self,
        cwt_i: torch.Tensor,
        cwt_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute wavelet coherence and phase between two CWTs.

        Parameters
        ----------
        cwt_i, cwt_j : torch.Tensor
            CWT of channels i and j. Shape (B, n_scales, T) complex.

        Returns
        -------
        coherence : torch.Tensor
            R^2(a, t). Shape (B, n_scales, T). Range [0, 1].
        phase : torch.Tensor
            Phase angle. Shape (B, n_scales, T). Range [-pi, pi].
        """
        # Cross-wavelet transform
        w_ij = cwt_i * torch.conj(cwt_j)  # (B, n_scales, T)

        # Auto-wavelet power
        w_ii = cwt_i * torch.conj(cwt_i)  # |W_i|^2
        w_jj = cwt_j * torch.conj(cwt_j)  # |W_j|^2

        # Smooth with scale-dependent window
        # Smoothing in time and scale
        s_smooth_ij = self._smooth(w_ij.abs() ** 2, self.smoothing_window)
        s_smooth_ii = self._smooth(w_ii.abs(), self.smoothing_window)
        s_smooth_jj = self._smooth(w_jj.abs(), self.smoothing_window)

        # Smoothed cross-wavelet (complex)
        s_ij_real = self._smooth(w_ij.real, self.smoothing_window)
        s_ij_imag = self._smooth(w_ij.imag, self.smoothing_window)
        s_ij = s_ij_real + 1j * s_ij_imag

        # Coherence: R^2 = |S(W_ij)|^2 / (S(|W_i|^2) * S(|W_j|^2))
        coherence = (s_ij.real ** 2 + s_ij.imag ** 2) / \
                    (s_smooth_ii * s_smooth_jj + EPS)
        coherence = torch.clamp(coherence, 0.0, 1.0)

        # Phase angle
        phase = torch.atan2(s_ij.imag, s_ij.real)

        return coherence, phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract wavelet coherence features from multi-channel sensor data.

        Parameters
        ----------
        x : torch.Tensor
            Multi-channel sensor signal. Shape (B, C, T) where C is
            number of channels and T is signal length.

        Returns
        -------
        torch.Tensor
            Wavelet coherence feature vector. Shape (B, feature_dim).
        """
        B, C, T = x.shape
        assert C == self.n_channels, \
            f"Expected {self.n_channels} channels, got {C}"

        # Step 1: Compute CWT for all channels
        cwt = self._morlet_cwt(x)  # (B, C, n_scales, T) complex

        # Step 2: Compute pairwise wavelet coherence
        features = []
        coherence_per_scale = []
        coherence_per_pair = []
        all_coherence_values = []

        pair_idx = 0
        for i in range(C):
            for j in range(i + 1, C):
                coh, phase = self._wavelet_coherence(
                    cwt[:, i, :, :], cwt[:, j, :, :]
                )  # (B, n_scales, T)

                # Mean coherence per scale (average over time)
                mean_coh_per_scale = coh.mean(dim=-1)  # (B, n_scales)
                features.append(mean_coh_per_scale)

                # Mean phase per scale (circular mean over time)
                mean_phase = circular_mean(phase, dim=-1)  # (B, n_scales)
                features.append(mean_phase)

                # Per-pair mean coherence (average over scales and time)
                pair_mean_coh = coh.mean(dim=(1, 2))  # (B,)
                coherence_per_pair.append(pair_mean_coh)

                # Collect for global stats
                all_coherence_values.append(coh.reshape(B, -1))

                pair_idx += 1

        # Per-scale mean coherence (average over pairs)
        # Reshape features to extract per-scale coherence
        coh_features = torch.stack([
            features[2 * k] for k in range(self.n_pairs)
        ], dim=1)  # (B, n_pairs, n_scales)
        per_scale_mean = coh_features.mean(dim=1)  # (B, n_scales)

        # Per-pair mean coherence
        per_pair_mean = torch.stack(coherence_per_pair, dim=1)  # (B, n_pairs)

        # Global coherence stats
        all_coh = torch.cat(all_coherence_values, dim=1)  # (B, n_scales * T * n_pairs)
        global_stats = torch.stack([
            all_coh.mean(dim=-1),
            all_coh.std(dim=-1),
            all_coh.max(dim=-1).values,
            all_coh.min(dim=-1).values,
        ], dim=-1)  # (B, 4)

        # Concatenate all features
        # features contains: [coh_pair0_scale, phase_pair0_scale,
        #                     coh_pair1_scale, phase_pair1_scale, ...]
        # Each is (B, n_scales)
        pair_features = torch.cat(features, dim=-1)  # (B, 2*n_pairs*n_scales)

        result = torch.cat([
            pair_features,       # (B, 2*n_pairs*n_scales)
            global_stats,        # (B, 4)
            per_scale_mean,      # (B, n_scales)
            per_pair_mean,       # (B, n_pairs)
        ], dim=-1)

        return result.float()

    def extra_repr(self) -> str:
        return (
            f"n_scales={self.n_scales}, n_channels={self.n_channels}, "
            f"n_pairs={self.n_pairs}, sample_rate={self.sample_rate}, "
            f"feature_dim={self.feature_dim}"
        )
