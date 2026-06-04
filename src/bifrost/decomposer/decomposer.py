"""
SpectralDecomposer — multi-resolution spectral decomposition.

Responsibilities:
    1. Frame time-domain signal into T overlapping windows.
    2. Per-frame: wavelet bank + FFT → sub-band spectral features.
    3. Selective scan over T frames (pure-PyTorch on CPU/MPS,
       real mamba-ssm on CUDA when available).
    4. Return SpectralTensor with temporal shape (B, T, n_freq).

Key features:
    - Multi-frame output: (B, T, n_freq)
    - Pure-PyTorch selective scan with optional mamba-ssm acceleration on CUDA
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spectral_tensor import SpectralTensor
from .selective_scan import SelectiveScan

def _check_mamba_available() -> bool:
    """Dynamic check for mamba-ssm (allows late installation)."""
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        return True
    except ImportError:
        return False


class LearnableWaveletBank(nn.Module):
    """Learnable 1-D convolution bank approximating multi-scale wavelet decomposition.

    Each filter operates at a different dilation (scale), producing one
    sub-band output per scale.
    """

    def __init__(
        self,
        in_channels: int,
        n_scales: int = 6,
        kernel_size: int = 31,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.filters = nn.ModuleList()
        for s in range(n_scales):
            dilation = 2 ** s
            padding = (kernel_size - 1) * dilation // 2
            self.filters.append(
                nn.Conv1d(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    groups=in_channels,
                    bias=False,
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: (B, 1, samples) → list of (B, 1, samples), one per scale."""
        return [f(x) for f in self.filters]


class MambaBlock(nn.Module):
    """SSM block: real mamba-ssm on CUDA, SelectiveScan everywhere else.

    Args:
        d_model: Input/output dimension.
        d_state: SSM state size.
        d_conv:  Depthwise conv kernel size.
        expand:  Inner expansion factor.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.use_cuda_mamba = _check_mamba_available() and torch.cuda.is_available()

        if self.use_cuda_mamba:
            from mamba_ssm.modules.mamba_simple import Mamba as MambaSSM
            self.ssm = MambaSSM(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.ssm_type = "mamba-ssm (CUDA)"
        else:
            self.ssm = SelectiveScan(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.ssm_type = "SelectiveScan (PyTorch)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        return self.ssm(x)


class SpectralDecomposer(nn.Module):
    """Decomposition stage: SpectralTensor → temporal multi-resolution spectral embedding.

    Pipeline (Phase 2):
        1. iFFT SpectralTensor → time-domain signal.
        2. Frame signal into T overlapping windows (hop = frame_size // 2).
        3. Per frame: wavelet bank → multi-scale sub-bands → FFT → amplitude.
        4. Project sub-band amplitudes → d_model (input_proj).
        5. MambaBlock selective scan over T frames → (B, T, d_model).
        6. Project back → (B, T, n_freq) amplitude + phase SpectralTensor.

    Args:
        n_fft:          FFT size for sub-band analysis.
        n_scales:       Number of wavelet scales.
        d_model:        SSM hidden dimension.
        wavelet_kernel: Kernel size for wavelet convolutions.
        n_frames:       Number of time frames T (default 32).
        use_mamba:      If False, forces SelectiveScan regardless of CUDA.
        d_state:        SSM state size (default 16).
        expand:         SSM expansion factor (default 2).
        d_conv:         SSM conv kernel size (default 4).
    """

    def __init__(
        self,
        n_fft: int = 512,
        n_scales: int = 6,
        d_model: int = 128,
        wavelet_kernel: int = 31,
        n_frames: int = 32,
        use_mamba: bool = True,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.n_scales = n_scales
        self.d_model = d_model
        self.n_freq = n_fft // 2 + 1
        self.n_frames = n_frames
        self.use_mamba = use_mamba

        self.wavelet_bank = LearnableWaveletBank(
            in_channels=1,
            n_scales=n_scales,
            kernel_size=wavelet_kernel,
        )

        # n_freq → d_model per frame
        self.input_proj = nn.Linear(self.n_freq, d_model)

        # Temporal SSM over T frames
        if use_mamba:
            self.ssm = MambaBlock(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            self.ssm = nn.Sequential(
                SelectiveScan(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            )

        # d_model → n_freq per frame
        self.output_proj = nn.Linear(d_model, self.n_freq)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    @property
    def ssm_type(self) -> str:
        if isinstance(self.ssm, MambaBlock):
            return self.ssm.ssm_type
        return "SelectiveScan (PyTorch)"

    def forward(self, st: SpectralTensor) -> SpectralTensor:
        """
        Args:
            st: SpectralTensor from canonicalization. amplitude shape: (B, n_freq_in)

        Returns:
            SpectralTensor with amplitude/phase shape: (B, T, n_freq)
            where T = n_frames.
        """
        amp = st.amplitude       # (B, n_freq_in) or (n_freq_in,)
        if amp.dim() == 1:
            amp = amp.unsqueeze(0)
        B = amp.shape[0]
        n_freq_in = amp.shape[-1]

        # 1. iFFT → time-domain signal (B, signal_len)
        complex_spec = st.complex_spectrum()
        if complex_spec.dim() == 1:
            complex_spec = complex_spec.unsqueeze(0)
        signal_len = 2 * (n_freq_in - 1)
        time_signal = torch.fft.irfft(complex_spec, n=signal_len, dim=-1)  # (B, L)

        # 2. Collapse any channel dims → (B, L) then frame
        if time_signal.dim() > 2:
            time_signal = time_signal.mean(dim=tuple(range(1, time_signal.dim() - 1)))
        frames, frame_size = _frame_signal(time_signal, self.n_frames)
        # frames: (B, T, frame_size)

        # 3. Per-frame wavelet → FFT → amplitude  (B, T, n_freq)
        frame_amps, frame_phases = self._per_frame_spectral(frames, B)

        # 4. Project → (B, T, d_model) → SSM → norm
        h = self.input_proj(frame_amps)     # (B, T, d_model)
        h = self.ssm(h)                      # (B, T, d_model)
        h = self.norm(h)

        # 5. Project back to frequency domain
        out_amp = self.output_proj(h).abs()  # (B, T, n_freq)
        out_phase = frame_phases             # (B, T, n_freq)

        # 6. Scale and uncertainty
        # Flatten scale to 1-D, interpolate to n_freq, broadcast over (B, T)
        scale_flat = st.scale.reshape(-1)[:n_freq_in].float()
        if self.n_freq != n_freq_in:
            scale_1d = F.interpolate(
                scale_flat.view(1, 1, -1),
                size=self.n_freq, mode="linear", align_corners=True,
            ).squeeze()
        else:
            scale_1d = scale_flat[:self.n_freq]
        out_scale = scale_1d.view(1, 1, self.n_freq).expand(B, self.n_frames, -1).contiguous()

        # Uncertainty: std across frames (more agreement → lower uncertainty)
        out_uncertainty = out_amp.std(dim=1, keepdim=True).expand_as(out_amp) / (
            math.sqrt(self.n_frames) + 1e-8
        )

        return SpectralTensor(
            amplitude=out_amp,
            phase=out_phase,
            scale=out_scale,
            uncertainty=out_uncertainty,
            metadata={
                **st.metadata,
                "stage": "decompose",
                "n_scales": self.n_scales,
                "n_fft_decompose": self.n_fft,
                "d_model": self.d_model,
                "n_frames": self.n_frames,
                "ssm_type": self.ssm_type,
                "frame_size": 2 * (n_freq_in - 1),
            },
        )

    def _per_frame_spectral(
        self, frames: torch.Tensor, B: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply wavelet bank + FFT to every frame.

        Args:
            frames: (B, T, frame_size)
            B: batch size

        Returns:
            (amplitudes, phases) each (B, T, n_freq)
        """
        T = frames.shape[1]
        # Reshape to process all frames at once: (B*T, 1, frame_size)
        flat = frames.reshape(B * T, 1, -1)

        sub_bands = self.wavelet_bank(flat)  # list of (B*T, 1, frame_size)

        sub_amps, sub_phases = [], []
        for sb in sub_bands:
            spec = torch.fft.rfft(sb.squeeze(1), n=self.n_fft, dim=-1)  # (B*T, n_freq)
            sub_amps.append(spec.abs())
            sub_phases.append(spec.angle())

        # Average amplitudes across scales: (B*T, n_freq)
        amp = torch.stack(sub_amps, dim=0).mean(dim=0)

        # Circular mean of phases
        sin_m = torch.stack([p.sin() for p in sub_phases], dim=0).mean(dim=0)
        cos_m = torch.stack([p.cos() for p in sub_phases], dim=0).mean(dim=0)
        phase = torch.atan2(sin_m, cos_m)

        # Reshape back to (B, T, n_freq)
        amp = amp.view(B, T, self.n_freq)
        phase = phase.view(B, T, self.n_freq)
        return amp, phase


# ---------------------------------------------------------------------------
# Signal framing utility
# ---------------------------------------------------------------------------

def _frame_signal(
    signal: torch.Tensor,
    n_frames: int,
) -> Tuple[torch.Tensor, int]:
    """Split signal into n_frames overlapping windows.

    Args:
        signal:   (B, L)
        n_frames: number of output frames T

    Returns:
        frames:     (B, T, frame_size)
        frame_size: int
    """
    B, L = signal.shape
    # Pad if needed so we get exactly n_frames
    frame_size = max(64, math.ceil(L / n_frames) * 2)
    hop = frame_size // 2
    needed = hop * (n_frames - 1) + frame_size
    if needed > L:
        signal = F.pad(signal, (0, needed - L))

    frames = signal.unfold(dimension=1, size=frame_size, step=hop)  # (B, T', frame_size)

    # Take exactly n_frames
    if frames.shape[1] >= n_frames:
        frames = frames[:, :n_frames, :]
    else:
        # Pad time axis with zeros
        pad_frames = torch.zeros(
            B, n_frames - frames.shape[1], frame_size,
            device=signal.device, dtype=signal.dtype,
        )
        frames = torch.cat([frames, pad_frames], dim=1)

    return frames, frame_size
