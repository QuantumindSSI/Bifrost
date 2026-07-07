"""
Multi-Scale Structural Coherence (MSC) — Image instance.

Implements phase congruency as the image instance of the MSC framework.
Phase congruency detects image features (edges, corners, lines) by measuring
the alignment of phases across multiple spatial frequency scales and orientations.

This is the image analog of CBMPC (audio instance of MSC). Both measure
cross-band phase coherence at multiple scales.

Mathematical formulation:
    PC(x,y) = |Σ_{s,b} W_{s,b} A_{s,b}(x,y) cos(φ_{s,b}(x,y) - φ̄(x,y))|
              / Σ_{s,b} A_{s,b}(x,y)

where:
    s indexes spatial frequency scales
    b indexes orientations
    A_{s,b} is the filter response amplitude
    φ_{s,b} is the filter response phase
    φ̄ is the weighted mean phase
    W_{s,b} are weighting factors

References:
    - Oppenheim & Lim (1981): phase carries structural information
    - Morrone & Burr (1988): phase-dependent energy model for feature detection
    - Kovesi (1999): phase congruency as image feature detector
    - Freeman & Adelson (1991): steerable filters for multi-scale decomposition
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_gabor_filter(
    size: int,
    scale: float,
    orientation: float,
    wavelength: float,
    sigma_f: float = 0.55,
    sigma_theta: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a log-Gabor filter pair (even and odd) in the frequency domain.

    Returns the complex filter response for even (cosine) and odd (sine) components.
    """
    # Create frequency grid centered at 0
    freq = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq, freq)
    radius = np.sqrt(fx**2 + fy**2)
    theta = np.arctan2(fy, fx)

    # Radial component (log-Gabor)
    fc = 1.0 / (wavelength * scale)
    radial = np.exp(-(np.log(radius / fc + 1e-12))**2 / (2 * sigma_f**2))
    radial[0, 0] = 0  # Remove DC

    # Angular component (Gaussian)
    dtheta = theta - orientation
    # Wrap to [-pi, pi]
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
    angular = np.exp(-(dtheta**2) / (2 * sigma_theta**2))

    # Combined filter
    filter_freq = radial * angular
    return filter_freq


class PhaseCongruencyExtractor(nn.Module):
    """
    Image MSC instance: Phase Congruency feature extractor.

    Computes phase congruency across multiple spatial frequency scales and
    orientations using log-Gabor filters. The output is a fixed-dimensional
    feature vector capturing the structural coherence of the image.

    Parameters
    ----------
    n_scales : int
        Number of spatial frequency scales (default: 5).
    n_orientations : int
        Number of orientations (default: 6).
    base_wavelength : float
        Wavelength of the smallest scale filter (default: 3.0 pixels).
    scale_factor : float
        Geometric scaling between scales (default: 2.0).
    image_size : int
        Expected image dimension (default: 32 for CIFAR-10).
    n_pc_bins : int
        Number of histogram bins for the phase congruency distribution (default: 16).
    """

    def __init__(
        self,
        n_scales: int = 5,
        n_orientations: int = 6,
        base_wavelength: float = 3.0,
        scale_factor: float = 2.0,
        image_size: int = 32,
        n_pc_bins: int = 16,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.base_wavelength = base_wavelength
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.n_pc_bins = n_pc_bins

        # Pre-compute log-Gabor filters
        filters = self._build_filters(image_size)
        # filters shape: (n_scales, n_orientations, image_size, image_size) complex
        self.register_buffer("filters", torch.from_numpy(filters))

        # Feature dimension:
        # - PC histogram: n_pc_bins
        # - Per-scale mean PC: n_scales
        # - Per-orientation mean PC: n_orientations
        # - Per-scale per-orientation mean amplitude: n_scales * n_orientations
        # - Global PC stats: 4 (mean, std, max, min)
        # - Downsampled PC map: 8*8 = 64
        self.feature_dim = (
            n_pc_bins
            + n_scales
            + n_orientations
            + n_scales * n_orientations
            + 4
            + 64
        )

    def _build_filters(self, size: int) -> np.ndarray:
        """Build the bank of log-Gabor filters in the frequency domain."""
        filters = np.zeros(
            (self.n_scales, self.n_orientations, size, size),
            dtype=np.complex64,
        )
        for s_idx in range(self.n_scales):
            scale = self.scale_factor ** s_idx
            for o_idx in range(self.n_orientations):
                orientation = o_idx * np.pi / self.n_orientations
                filt = _log_gabor_filter(
                    size=size,
                    scale=scale,
                    orientation=orientation,
                    wavelength=self.base_wavelength,
                )
                filters[s_idx, o_idx] = filt.astype(np.complex64)
        return filters

    def _compute_phase_congruency(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute phase congruency for a batch of images.

        Parameters
        ----------
        image : torch.Tensor
            (B, 1, H, W) grayscale image.

        Returns
        -------
        pc_map : torch.Tensor
            (B, H, W) phase congruency at each pixel.
        per_scale_pc : torch.Tensor
            (B, n_scales) mean phase congruency per scale.
        per_orient_pc : torch.Tensor
            (B, n_orientations) mean phase congruency per orientation.
        """
        B, C, H, W = image.shape
        assert C == 1, f"Expected single-channel image, got {C} channels"
        assert H == self.image_size and W == self.image_size, \
            f"Expected {self.image_size}x{self.image_size} image, got {H}x{W}"

        # Compute 2D FFT of the image
        img_fft = torch.fft.fft2(image.squeeze(1))  # (B, H, W)

        # Apply each filter and compute inverse FFT
        # filters: (S, O, H, W) complex
        # img_fft: (B, H, W)
        # response: (B, S, O, H, W) complex
        responses = []
        for s in range(self.n_scales):
            for o in range(self.n_orientations):
                filt = self.filters[s, o]  # (H, W)
                resp = img_fft * filt.unsqueeze(0)  # (B, H, W)
                resp_spatial = torch.fft.ifft2(resp)  # (B, H, W) complex
                responses.append(resp_spatial)
        responses = torch.stack(responses, dim=1)  # (B, S*O, H, W)
        responses = responses.view(B, self.n_scales, self.n_orientations, H, W)

        # Amplitude and phase
        amplitude = responses.abs()  # (B, S, O, H, W)
        phase = responses.angle()  # (B, S, O, H, W)

        # Weighted mean phase (Kovesi's approach)
        # φ̄ = atan2(Σ A sin(φ), Σ A cos(φ))
        sum_sin = (amplitude * torch.sin(phase)).sum(dim=(1, 2))  # (B, H, W)
        sum_cos = (amplitude * torch.cos(phase)).sum(dim=(1, 2))  # (B, H, W)
        mean_phase = torch.atan2(sum_sin, sum_cos)  # (B, H, W)

        # Phase deviation: |cos(φ - φ̄)| weighted by amplitude
        phase_diff = phase - mean_phase.unsqueeze(1).unsqueeze(1)  # (B, S, O, H, W)
        energy = (amplitude * torch.cos(phase_diff)).sum(dim=(1, 2))  # (B, H, W)
        total_amplitude = amplitude.sum(dim=(1, 2)) + 1e-8  # (B, H, W)

        # Phase congruency
        pc_map = energy.abs() / total_amplitude  # (B, H, W)

        # Per-scale and per-orientation PC
        per_scale_pc = (amplitude * torch.cos(phase_diff)).abs().sum(dim=2) / \
            (amplitude.sum(dim=2) + 1e-8)  # (B, S, H, W)
        per_scale_pc = per_scale_pc.mean(dim=(2, 3))  # (B, S)

        per_orient_pc = (amplitude * torch.cos(phase_diff)).abs().sum(dim=1) / \
            (amplitude.sum(dim=1) + 1e-8)  # (B, O, H, W)
        per_orient_pc = per_orient_pc.mean(dim=(2, 3))  # (B, O)

        return pc_map, per_scale_pc, per_orient_pc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract phase congruency features from a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            (B, C, H, W) image. If C > 1, converted to grayscale.

        Returns
        -------
        features : torch.Tensor
            (B, feature_dim) phase congruency feature vector.
        """
        B, C, H, W = x.shape

        # Convert to grayscale if needed
        if C > 1:
            x = x.mean(dim=1, keepdim=True)
        elif C == 1:
            pass
        else:
            x = x.unsqueeze(1)

        # Normalize to [0, 1]
        x = x.float()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        # Resize if needed
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        # Compute phase congruency
        pc_map, per_scale_pc, per_orient_pc = self._compute_phase_congruency(x)

        # Extract features
        features = []

        # 1. PC histogram (16 bins)
        pc_flat = pc_map.view(B, -1)
        pc_hist = torch.zeros(B, self.n_pc_bins, device=x.device)
        for b in range(B):
            hist = torch.histc(pc_flat[b], bins=self.n_pc_bins, min=0, max=1)
            pc_hist[b] = hist / (hist.sum() + 1e-8)
        features.append(pc_hist)

        # 2. Per-scale mean PC (n_scales)
        features.append(per_scale_pc)

        # 3. Per-orientation mean PC (n_orientations)
        features.append(per_orient_pc)

        # 4. Per-scale per-orientation mean amplitude (n_scales * n_orientations)
        # Recompute amplitude means
        img_fft = torch.fft.fft2(x.squeeze(1))
        amp_means = torch.zeros(B, self.n_scales * self.n_orientations, device=x.device)
        idx = 0
        for s in range(self.n_scales):
            for o in range(self.n_orientations):
                filt = self.filters[s, o]
                resp = (img_fft * filt.unsqueeze(0)).abs().mean(dim=(1, 2))
                amp_means[:, idx] = resp
                idx += 1
        features.append(amp_means)

        # 5. Global PC stats (4: mean, std, max, min)
        pc_stats = torch.stack([
            pc_flat.mean(dim=1),
            pc_flat.std(dim=1),
            pc_flat.max(dim=1).values,
            pc_flat.min(dim=1).values,
        ], dim=1)
        features.append(pc_stats)

        # 6. Downsampled PC map (8x8 = 64)
        pc_map_4d = pc_map.unsqueeze(1)  # (B, 1, H, W)
        pc_downsampled = F.adaptive_avg_pool2d(pc_map_4d, (8, 8))  # (B, 1, 8, 8)
        features.append(pc_downsampled.view(B, -1))

        return torch.cat(features, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"n_scales={self.n_scales}, n_orientations={self.n_orientations}, "
            f"base_wavelength={self.base_wavelength}, scale_factor={self.scale_factor}, "
            f"image_size={self.image_size}, feature_dim={self.feature_dim}"
        )
