"""
Modulation-Preserving SSM architectures.

Investigates whether an SSM can be designed that preserves the temporal
modulation structure of speech while still learning phase relationships.

Architecture C (Residual SSM): keeps the current SSM but adds a residual
skip connection that preserves the original spectrogram's modulation structure.

Architecture A (Band-wise SSM): runs a separate SSM per mel band, preserving
band-wise structure for CBMPC extraction.

See dev-docs/04_MODULATION_PRESERVING_SSM_INVESTIGATION.md for the full
investigation plan.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .spectral_tensor import SpectralTensor


class MelProjection(nn.Module):
    """Fixed mel filterbank projection from n_freq to n_mels."""

    def __init__(self, n_fft: int, n_mels: int, sample_rate: int):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        fb = self._build_mel_filterbank(n_fft, n_mels, sample_rate)
        self.register_buffer("mel_fb", fb)  # (n_mels, n_freq)

    def _build_mel_filterbank(self, n_fft: int, n_mels: int, sr: int) -> torch.Tensor:
        n_freq = n_fft // 2 + 1
        f_min, f_max = 0.0, sr / 2.0
        def hz_to_mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
        mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
        fft_freqs = torch.linspace(0, f_max, n_freq)
        fb = torch.zeros(n_mels, n_freq)
        for m in range(n_mels):
            left, center, right = hz_points[m], hz_points[m+1], hz_points[m+2]
            for k in range(n_freq):
                f = fft_freqs[k]
                if f < left or f > right: continue
                if f <= center:
                    fb[m, k] = (f - left) / (center - left + 1e-8)
                else:
                    fb[m, k] = (right - f) / (right - center + 1e-8)
        return fb

    def forward(self, stft_mag: torch.Tensor) -> torch.Tensor:
        """stft_mag: (B, n_freq, T) → (B, n_mels, T)"""
        return torch.matmul(self.mel_fb, stft_mag)


class ResidualModulationSSM(nn.Module):
    """
    Architecture C: Residual SSM with modulation-preserving skip connection.

    The SSM processes the spectrogram as before, but a residual connection
    preserves the original mel-spectrogram's modulation structure. The output
    is: SSM_output + alpha * mel_projection.

    This allows CBMPC to be extracted from the residual path, which retains
    the natural modulation structure of speech.
    """

    def __init__(
        self,
        ssm_decomposer: nn.Module,
        n_fft: int = 1024,
        n_mels: int = 64,
        sample_rate: int = 16000,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.ssm = ssm_decomposer
        self.mel_proj = MelProjection(n_fft, n_mels, sample_rate)
        self.alpha = alpha
        self.n_mels = n_mels

    def forward(
        self,
        st: SpectralTensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Process through SSM with residual modulation preservation.

        Returns a SpectralTensor where:
        - amplitude = SSM_amplitude + alpha * mel_spectrogram
        - phase = SSM_phase (preserved from the SSM)
        """
        # Run the SSM as normal
        ssm_output, h_T = self.ssm(st, h_0)

        # Compute the mel projection of the original spectrogram
        # st.amplitude shape: (B, T, n_freq) or (B, n_freq)
        amp = st.amplitude
        if amp.dim() == 3:
            # (B, T, n_freq) → transpose to (B, n_freq, T) for mel projection
            amp_t = amp.transpose(1, 2)  # (B, n_freq, T)
            mel = self.mel_proj(amp_t)  # (B, n_mels, T)
            mel = mel.transpose(1, 2)  # (B, T, n_mels)
        else:
            # (B, n_freq) → (B, n_mels)
            mel = torch.matmul(self.mel_fb, amp)

        # Residual: add the mel projection to the SSM output amplitude
        # The SSM output may have different d_model than n_mels.
        # We need to handle the dimension mismatch.
        ssm_amp = ssm_output.amplitude

        if ssm_amp.dim() == 3 and mel.dim() == 3:
            B, T_ssm, d = ssm_amp.shape
            B_m, T_mel, m = mel.shape
            if T_ssm != T_mel:
                # Interpolate mel to match SSM temporal resolution
                mel = mel.transpose(1, 2)  # (B, m, T_mel)
                mel = torch.nn.functional.interpolate(
                    mel, size=T_ssm, mode='linear', align_corners=False
                )
                mel = mel.transpose(1, 2)  # (B, T_ssm, m)

            if d != m:
                # Project mel to d_model dimensions
                # Use a simple linear projection (not learned, to keep it simple)
                if m > d:
                    # Average pool mel bands to match d
                    mel = mel.reshape(B, T_ssm, d, m // d).mean(dim=-1)
                else:
                    # Pad mel to match d
                    pad = torch.zeros(B, T_ssm, d - m, device=mel.device)
                    mel = torch.cat([mel, pad], dim=-1)

            # Residual addition
            residual_amp = ssm_amp + self.alpha * mel
        else:
            residual_amp = ssm_amp

        return SpectralTensor(
            amplitude=residual_amp,
            phase=ssm_output.phase,
            scale=ssm_output.scale,
            uncertainty=ssm_output.uncertainty,
            metadata=ssm_output.metadata,
        ), h_T


class BandWiseSSM(nn.Module):
    """
    Architecture A: Band-wise SSM (no cross-band mixing).

    Runs a separate lightweight SSM for each mel band, preserving the
    band-wise structure that CBMPC relies on. Each band learns its own
    temporal dynamics without mixing across bands.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        n_mels: int = 64,
        sample_rate: int = 16000,
        d_state: int = 8,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.mel_proj = MelProjection(n_fft, n_mels, sample_rate)

        # Per-band temporal smoothing (simple learned IIR filter)
        # For each band: a simple recurrent update
        # h[t] = alpha * h[t-1] + (1-alpha) * x[t]
        # where alpha is learned per band
        self.band_alphas = nn.Parameter(torch.zeros(n_mels))  # sigmoid → (0,1)

    def forward(
        self,
        st: SpectralTensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """Process each band independently with a learned temporal filter."""
        amp = st.amplitude
        phase = st.phase

        if amp.dim() == 3:
            # (B, T, n_freq) → mel → (B, T, n_mels)
            amp_t = amp.transpose(1, 2)  # (B, n_freq, T)
            mel = self.mel_proj(amp_t)  # (B, n_mels, T)
            mel = mel.transpose(1, 2)  # (B, T, n_mels)

            # Apply per-band temporal smoothing
            alphas = torch.sigmoid(self.band_alphas)  # (n_mels,)
            B, T, M = mel.shape
            output = torch.zeros_like(mel)
            h = torch.zeros(B, M, device=mel.device)
            for t in range(T):
                h = alphas.unsqueeze(0) * h + (1 - alphas.unsqueeze(0)) * mel[:, t, :]
                output[:, t, :] = h

            # Return as SpectralTensor with mel-band amplitude
            return SpectralTensor(
                amplitude=output,
                phase=phase if phase.dim() == 3 and phase.shape[-1] == M else phase[..., :M] if phase.dim() >= 2 else phase,
                scale=st.scale,
                uncertainty=st.uncertainty,
                metadata=st.metadata,
            ), h
        else:
            # No temporal dimension — just project
            mel = torch.matmul(self.mel_proj.mel_fb, amp)
            return SpectralTensor(
                amplitude=mel,
                phase=phase,
                scale=st.scale,
                uncertainty=st.uncertainty,
                metadata=st.metadata,
            ), torch.zeros(amp.shape[0], self.n_mels, device=amp.device)
