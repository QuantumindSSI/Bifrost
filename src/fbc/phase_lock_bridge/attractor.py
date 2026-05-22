"""
FrequencyAttractor — stable spectral pattern container.

An attractor is a persistent spectral embedding discovered by S3.
It represents a stable pattern in spectral space that the system
has identified as a meaningful unit of knowledge.

This module provides the data structure that the Phase-Lock Bridge
operates on when comparing attractors across domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class FrequencyAttractor:
    """
    A stable spectral pattern identified by the attractor identification stage.

    Attributes
    ----------
    centroid : torch.Tensor
        Mean spectral embedding of the attractor, shape ``(d_model,)``.
    phase_signature : torch.Tensor
        Characteristic phase profile, shape ``(n_bands,)``.
    amplitude_profile : torch.Tensor
        Characteristic amplitude profile, shape ``(d_model,)``.
    stability : float
        Temporal stability score in [0, 1].  Higher = more stable.
    domain : str
        Domain label (e.g. "audio", "vision", "language").
    attractor_id : str
        Unique identifier for this attractor.
    metadata : dict
        Arbitrary provenance metadata.
    """

    centroid: torch.Tensor
    phase_signature: torch.Tensor
    amplitude_profile: torch.Tensor
    stability: float = 0.0
    domain: str = "unknown"
    attractor_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Derived properties ─────────────────────────────────────────────

    @property
    def d_model(self) -> int:
        return self.centroid.shape[-1]

    @property
    def n_bands(self) -> int:
        return self.phase_signature.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.centroid.device

    # ── Utilities ──────────────────────────────────────────────────────

    def to(self, device: torch.device) -> FrequencyAttractor:
        """Move all tensors to *device*."""
        return FrequencyAttractor(
            centroid=self.centroid.to(device),
            phase_signature=self.phase_signature.to(device),
            amplitude_profile=self.amplitude_profile.to(device),
            stability=self.stability,
            domain=self.domain,
            attractor_id=self.attractor_id,
            metadata=self.metadata,
        )

    def spectral_energy(self) -> float:
        """Total spectral energy (L2 norm of amplitude profile)."""
        return self.amplitude_profile.norm().item()

    def __repr__(self) -> str:
        return (
            f"FrequencyAttractor(id={self.attractor_id!r}, domain={self.domain!r}, "
            f"d_model={self.d_model}, n_bands={self.n_bands}, "
            f"stability={self.stability:.3f})"
        )
