"""
SpectralTensor — canonical data container for the Bifröst pipeline.

Every pipeline stage operates on SpectralTensor instances.
Fields follow the Engineering Script specification:
    amplitude : torch.Tensor   — magnitude spectrum
    phase     : torch.Tensor   — phase spectrum (radians, [-π, π])
    scale     : torch.Tensor   — per-band scale / resolution metadata
    uncertainty : torch.Tensor — confidence / noise estimate per element
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

import torch


@dataclasses.dataclass
class SpectralTensor:
    """Immutable-ish canonical spectral representation used across all Bifröst stages."""

    amplitude: torch.Tensor
    phase: torch.Tensor
    scale: torch.Tensor
    uncertainty: torch.Tensor

    # Provenance metadata carried through the pipeline
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # ----- convenience helpers -----

    @property
    def device(self) -> torch.device:
        return self.amplitude.device

    @property
    def dtype(self) -> torch.dtype:
        return self.amplitude.dtype

    @property
    def shape(self) -> torch.Size:
        """Return shape of the amplitude tensor (representative)."""
        return self.amplitude.shape

    @property
    def num_bands(self) -> int:
        """Number of frequency bands (last dim by convention)."""
        return self.amplitude.shape[-1]

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> SpectralTensor:
        """Move all tensors to *device* (and optionally cast *dtype*)."""
        kwargs: Dict[str, Any] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return SpectralTensor(
            amplitude=self.amplitude.to(**kwargs),
            phase=self.phase.to(**kwargs),
            scale=self.scale.to(**kwargs),
            uncertainty=self.uncertainty.to(**kwargs),
            metadata=dict(self.metadata),
        )

    def detach(self) -> SpectralTensor:
        """Detach all tensors from the computation graph."""
        return SpectralTensor(
            amplitude=self.amplitude.detach(),
            phase=self.phase.detach(),
            scale=self.scale.detach(),
            uncertainty=self.uncertainty.detach(),
            metadata=dict(self.metadata),
        )

    def complex_spectrum(self) -> torch.Tensor:
        """Reconstruct the complex spectrum: amplitude * exp(j * phase)."""
        return self.amplitude * torch.exp(1j * self.phase)

    def energy(self) -> torch.Tensor:
        """Total spectral energy (sum of squared amplitudes)."""
        return (self.amplitude ** 2).sum()

    # ----- validation -----

    def validate(self) -> None:
        """Raise ``ValueError`` if tensor shapes are inconsistent."""
        ref = self.amplitude.shape
        for name in ("phase", "scale", "uncertainty"):
            t = getattr(self, name)
            if t.shape != ref:
                raise ValueError(
                    f"SpectralTensor shape mismatch: amplitude {ref} vs {name} {t.shape}"
                )

    def __repr__(self) -> str:
        return (
            f"SpectralTensor(shape={list(self.shape)}, "
            f"bands={self.num_bands}, device={self.device})"
        )
