"""
Cross-Band Modulation Phase Coherence (CBMPC) extractor.

Implements the technique proposed in CBMPC_TECHNIQUE_PROPOSAL.md:
    1. Compute spectrogram S(t, f) via STFT or filterbank.
    2. Log-compress: L(t, f) = log(|S(t, f)| + eps).
    3. For each frequency band f, compute temporal modulation spectrum
       via FFT along the time axis: L_tilde(omega_t, f).
    4. Extract modulation amplitude A(omega_t, f) and phase phi(omega_t, f).
    5. Compute cross-band phase locking value:
       C(omega_t) = |mean_f exp(i * phi(omega_t, f))|
    6. Feature vector = [C(omega_1)...C(omega_K), mean_A(omega_1)...mean_A(omega_K)].

This is a pure signal-processing module (no learned parameters) that can be
used as a feature extractor in front of any classifier.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils.spectral_utils import EPS, hz_to_mel, mel_to_hz


class CBMPCExtractor(nn.Module):
    """
    Cross-Band Modulation Phase Coherence extractor.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    n_fft : int
        FFT size for the initial spectrogram.
    hop_length : int
        STFT hop length.
    n_mels : int
        Number of mel-spaced frequency bands to project the spectrogram into
        before computing modulation spectra. Reduces from n_fft//2+1 bands
        to n_mels bands.
    modulation_freqs : list of float
        Temporal modulation frequencies (Hz) at which to measure coherence.
    duration_seconds : float
        Expected clip duration (used to compute the number of STFT frames).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        modulation_freqs: Optional[list] = None,
        duration_seconds: float = 1.0,
        feature_mode: str = "rich",
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration_seconds = duration_seconds

        if modulation_freqs is None:
            modulation_freqs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        self.modulation_freqs = modulation_freqs
        self.n_mod_freqs = len(modulation_freqs)

        # Pre-compute mel filterbank (n_mels x n_freq)
        mel_fb = self._build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)  # (n_mels, n_freq)

        # Feature dimension depends on mode:
        #   "compact": 2 * n_mod_freqs (PLV + mean amplitude per mod freq)
        #   "rich": n_mels * n_mod_freqs + n_mod_freqs + n_mod_freqs
        #           (per-band modulation amplitudes + PLV + mean amplitudes)
        self.feature_mode = feature_mode
        if feature_mode == "compact":
            self.feature_dim = 2 * self.n_mod_freqs
        else:  # "rich"
            self.feature_dim = self.n_mels * self.n_mod_freqs + 2 * self.n_mod_freqs

    def _build_mel_filterbank(self) -> torch.Tensor:
        """Build a mel filterbank matrix (n_mels, n_fft//2+1)."""
        n_freq = self.n_fft // 2 + 1
        f_min = 0.0
        f_max = self.sample_rate / 2.0

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])

        # FFT frequency bins
        fft_freqs = torch.linspace(0, f_max, n_freq)

        # Triangular filters
        fb = torch.zeros(self.n_mels, n_freq)
        for m in range(self.n_mels):
            left = hz_points[m]
            center = hz_points[m + 1]
            right = hz_points[m + 2]
            for k in range(n_freq):
                f = fft_freqs[k]
                if f < left or f > right:
                    continue
                if f <= center:
                    fb[m, k] = (f - left) / (center - left + EPS)
                else:
                    fb[m, k] = (right - f) / (right - center + EPS)
        return fb

    def _stft_to_mel(self, stft_mag: torch.Tensor) -> torch.Tensor:
        """
        Convert STFT magnitude to mel-scaled magnitude.

        Parameters
        ----------
        stft_mag : torch.Tensor
            Shape (B, n_freq, T_frames) — magnitude spectrogram.

        Returns
        -------
        mel_mag : torch.Tensor
            Shape (B, n_mels, T_frames) — mel-scaled magnitude.
        """
        # mel_fb: (n_mels, n_freq)
        # stft_mag: (B, n_freq, T)
        mel_mag = torch.matmul(self.mel_fb, stft_mag)  # (B, n_mels, T)
        return mel_mag

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract CBMPC features from raw audio waveforms.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape (B, T_samples) — raw audio.

        Returns
        -------
        features : torch.Tensor
            Shape (B, feature_dim) — CBMPC feature vector.
        """
        B = waveform.shape[0]

        # Step 1: STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )  # (B, n_freq, T_frames)
        stft_mag = stft.abs()  # (B, n_freq, T_frames)

        # Step 2: Mel projection
        mel_mag = self._stft_to_mel(stft_mag)  # (B, n_mels, T_frames)

        # Step 3: Log compression
        log_mag = torch.log(mel_mag + EPS)  # (B, n_mels, T_frames)

        # Step 4: Temporal FFT (modulation spectrum)
        # FFT along the time axis (last dim)
        modulation_spectrum = torch.fft.rfft(log_mag, dim=-1)
        # Shape: (B, n_mels, T_frames // 2 + 1)

        # Step 5: Extract modulation amplitude and phase
        mod_amp = modulation_spectrum.abs()  # (B, n_mels, n_mod_bins)
        mod_phase = modulation_spectrum.angle()  # (B, n_mels, n_mod_bins)

        # Step 6: Map modulation frequency bins to target frequencies
        n_frames = log_mag.shape[-1]
        frame_rate = self.sample_rate / self.hop_length  # Hz
        mod_freqs_all = torch.fft.rfftfreq(n_frames, d=1.0 / frame_rate)

        # For each target modulation frequency, find the nearest bin
        target_bins = []
        for target_f in self.modulation_freqs:
            if len(mod_freqs_all) == 0:
                target_bins.append(0)
                continue
            bin_idx = torch.argmin(torch.abs(mod_freqs_all - target_f)).item()
            target_bins.append(bin_idx)

        # Step 7: Compute cross-band phase locking value for each modulation freq
        plv_values = []
        mean_amp_values = []
        per_band_amp_values = []  # For rich mode

        for bin_idx in target_bins:
            # Phase across frequency bands at this modulation frequency
            phases = mod_phase[:, :, bin_idx]  # (B, n_mels)
            # PLV = |mean_f exp(i * phi)|
            plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=1)).real
            plv_values.append(plv)  # (B,)

            # Mean modulation amplitude across bands
            amp = mod_amp[:, :, bin_idx].mean(dim=1)  # (B,)
            mean_amp_values.append(amp)

            # Per-band modulation amplitude (for rich mode)
            per_band_amp_values.append(mod_amp[:, :, bin_idx])  # (B, n_mels)

        # Step 8: Concatenate into feature vector
        plv_tensor = torch.stack(plv_values, dim=1)  # (B, n_mod_freqs)
        amp_tensor = torch.stack(mean_amp_values, dim=1)  # (B, n_mod_freqs)

        if self.feature_mode == "compact":
            features = torch.cat([plv_tensor, amp_tensor], dim=1)  # (B, 2*n_mod_freqs)
        else:  # "rich"
            # Per-band modulation amplitudes: (B, n_mels * n_mod_freqs)
            per_band = torch.stack(per_band_amp_values, dim=2)  # (B, n_mels, n_mod_freqs)
            per_band_flat = per_band.reshape(per_band.shape[0], -1)  # (B, n_mels*n_mod_freqs)
            features = torch.cat([per_band_flat, plv_tensor, amp_tensor], dim=1)

        return features.float()


class CBMPCClassifier(nn.Module):
    """
    CBMPC feature extractor + linear classifier.
    """

    def __init__(
        self,
        extractor: CBMPCExtractor,
        n_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(extractor.feature_dim, n_classes)

    def forward(self, x, *_):
        features = self.extractor(x)
        features = self.dropout(features)
        return self.classifier(features)
