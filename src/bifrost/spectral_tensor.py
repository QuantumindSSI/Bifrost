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

    def validate(self) -> None:
        """
        Validate tensor invariants. Call explicitly when strict checks needed.

        Raises
        ------
        ValueError
            If tensor shapes are inconsistent, if amplitude/phase/scale/uncertainty
            have invalid values, or if tensors are on different devices.

        Complexity
        ----------
        O(N) - where N is total number of elements in tensors.

        Side Effects
        ------------
        None.
        """
        # === VALIDATION CHECKS ===
        if self.amplitude.shape != self.phase.shape:
            raise ValueError(
                f"SpectralTensor shape mismatch: amplitude {self.amplitude.shape} vs phase {self.phase.shape}"
            )
        if self.amplitude.shape != self.scale.shape:
            raise ValueError(
                f"SpectralTensor shape mismatch: amplitude {self.amplitude.shape} vs scale {self.scale.shape}"
            )
        if self.amplitude.shape != self.uncertainty.shape:
            raise ValueError(
                f"SpectralTensor shape mismatch: amplitude {self.amplitude.shape} vs uncertainty {self.uncertainty.shape}"
            )
        if not torch.all(self.amplitude >= 0):
            raise ValueError("SpectralTensor amplitude must be non-negative")
        if not (torch.all(self.phase >= -torch.pi) and torch.all(self.phase <= torch.pi)):
            raise ValueError(
                f"SpectralTensor phase must be in [-π, π], got range [{self.phase.min()}, {self.phase.max()}]"
            )
        if not torch.all(self.scale > 0):
            raise ValueError("SpectralTensor scale must be positive")
        if not torch.all(self.uncertainty >= 0):
            raise ValueError("SpectralTensor uncertainty must be non-negative")
        if not (self.amplitude.device == self.phase.device == self.scale.device == self.uncertainty.device):
            raise ValueError("SpectralTensor tensors must be on the same device")

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
        """
        Move all tensors to device (and optionally cast dtype).

        Args
        ----
        device : torch.device or str
            Target device for all tensors.
        dtype : torch.dtype, optional
            Optional dtype for casting all tensors.

        Returns
        -------
        SpectralTensor
            New SpectralTensor with tensors moved to device.

        Complexity
        ----------
        O(N) - where N is total number of elements in tensors.

        Side Effects
        ------------
        None (returns new instance).
        """
        if not isinstance(device, (torch.device, str)):
            raise TypeError(
                f"device must be torch.device or str, got {type(device).__name__}"
            )
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError(
                f"dtype must be torch.dtype, got {type(dtype).__name__}"
            )

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
        """
        Detach all tensors from the computation graph.

        Returns
        -------
        SpectralTensor
            New SpectralTensor with detached tensors.

        Complexity
        ----------
        O(1) - shallow copy with detached tensors.

        Side Effects
        ------------
        None (returns new instance).
        """
        return SpectralTensor(
            amplitude=self.amplitude.detach(),
            phase=self.phase.detach(),
            scale=self.scale.detach(),
            uncertainty=self.uncertainty.detach(),
            metadata=dict(self.metadata),
        )

    def complex_spectrum(self) -> torch.Tensor:
        """
        Reconstruct the complex spectrum: amplitude * exp(j * phase).

        Returns
        -------
        torch.Tensor
            Complex spectrum tensor.

        Raises
        ------
        ValueError
            If amplitude or phase contain NaN/Inf values.

        Complexity
        ----------
        O(N) - where N is total number of elements in tensors.

        Side Effects
        ------------
        None.
        """
        if not torch.isfinite(self.amplitude).all():
            raise ValueError("amplitude contains NaN or Inf values")
        if not torch.isfinite(self.phase).all():
            raise ValueError("phase contains NaN or Inf values")
        return self.amplitude * torch.exp(1j * self.phase)

    def energy(self) -> torch.Tensor:
        """
        Total spectral energy (sum of squared amplitudes).

        Returns
        -------
        torch.Tensor
            Scalar energy value.

        Raises
        ------
        ValueError
            If amplitude contains NaN/Inf values.

        Complexity
        ----------
        O(N) - where N is total number of elements in amplitude.

        Side Effects
        ------------
        None.
        """
        if not torch.isfinite(self.amplitude).all():
            raise ValueError("amplitude contains NaN or Inf values")
        return (self.amplitude ** 2).sum()

    def __repr__(self) -> str:
        return (
            f"SpectralTensor(shape={list(self.shape)}, "
            f"bands={self.num_bands}, device={self.device})"
        )
