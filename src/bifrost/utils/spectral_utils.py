"""
Shared spectral utilities for Bifrost.

This module centralises the small, frequently-duplicated signal-processing
helpers used across the MSC (Multi-Scale Structural Coherence) modules,
validation harnesses, and decomposers:

    - ``wrap_phase``      : wrap phase angles to [-pi, pi]
    - ``circular_mean``   : circular mean of phase angles
    - ``compute_plv``     : phase locking value |mean(exp(i*phase))|
    - ``hz_to_mel``       : convert Hz to the mel scale
    - ``mel_to_hz``       : convert mel scale back to Hz
    - ``EPS``             : small constant for numerical stability

Keeping these in one place avoids drift between the audio (CBMPC), image
(PhaseCongruency), sensor (WaveletCoherence), and validation code paths
that all rely on the same underlying phase statistics.
"""

from __future__ import annotations

import math
from typing import Union

import torch

# Numerical-stability constant used by division-heavy coherence metrics.
EPS: float = 1e-8

# Type alias for tensors or python floats accepted by the mel helpers.
_TensorLike = Union[float, torch.Tensor]


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Wrap phase angles to the principal interval [-pi, pi].

    Uses atan2(sin, cos) so the result is always numerically in range
    regardless of how many multiples of 2*pi the input contains.

    Parameters
    ----------
    phase : torch.Tensor
        Phase angles in radians (any shape).

    Returns
    -------
    torch.Tensor
        Wrapped phase with the same shape and dtype as ``phase``.
    """
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def circular_mean(phases: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Circular mean of phase angles along ``dim``.

    The circular mean is the angle of the mean resultant vector:

        mean = atan2(mean(sin(phi)), mean(cos(phi)))

    Unlike an arithmetic mean it correctly handles the wrap-around at
    +/- pi (e.g. the circular mean of -pi + epsilon and +pi - epsilon is
    ~0, not ~0).

    Parameters
    ----------
    phases : torch.Tensor
        Phase angles in radians.
    dim : int
        Dimension to reduce (default: last).

    Returns
    -------
    torch.Tensor
        Circular mean phase with ``dim`` removed.
    """
    mean_sin = torch.sin(phases).mean(dim=dim)
    mean_cos = torch.cos(phases).mean(dim=dim)
    return torch.atan2(mean_sin, mean_cos)


def compute_plv(phases: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Phase Locking Value (PLV) = |mean(exp(i * phase))|.

    PLV measures phase synchrony across the reduced dimension. It is 1
    when all phases are identical and 0 when they are uniformly spread.

    Parameters
    ----------
    phases : torch.Tensor
        Phase angles in radians.
    dim : int
        Dimension to reduce (default: last).

    Returns
    -------
    torch.Tensor
        Real-valued PLV in [0, 1] with ``dim`` removed.
    """
    return torch.abs(torch.mean(torch.exp(1j * phases), dim=dim)).real


def hz_to_mel(hz: _TensorLike) -> _TensorLike:
    """Convert frequency in Hz to the mel scale.

    Uses the standard Slaney/HTK formula::

        mel = 2595 * log10(1 + hz / 700)

    Accepts either a python float or a torch tensor and returns the same
    type so it can be used both in filterbank construction loops and in
    vectorised code.
    """
    if isinstance(hz, torch.Tensor):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: _TensorLike) -> _TensorLike:
    """Convert mel scale value back to frequency in Hz.

    Inverse of :func:`hz_to_mel`::

        hz = 700 * (10^(mel / 2595) - 1)

    Accepts either a python float or a torch tensor and returns the same
    type.
    """
    if isinstance(mel, torch.Tensor):
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)
