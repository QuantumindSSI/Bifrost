"""
PhaseLockBridge — cross-domain knowledge transfer via phase-locked attractors.

Per Engineering Script §3:
    - Compute phase-lock activation score between attractors.
    - Enforce multi-band coherence confirmation (≥ 3 bands).
    - Prevent false positives via cross-band agreement.
    - When activated, create bridge edges between attractor nodes.
    - Transfer knowledge from source attractor to analogous target.

This is the Phase 1 initial implementation: scoring, gating, and
candidate identification.  SKG edge creation is deferred to Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attractor import FrequencyAttractor
from ..spectral_tensor import SpectralTensor


@dataclass
class BridgeCandidate:
    """Result of a phase-lock evaluation between two attractors."""

    source: FrequencyAttractor
    target: FrequencyAttractor
    activation_score: float
    band_coherences: torch.Tensor  # per-band coherence scores
    n_locked_bands: int
    is_activated: bool
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BridgeCandidate(src={self.source.attractor_id!r}→"
            f"tgt={self.target.attractor_id!r}, "
            f"score={self.activation_score:.4f}, "
            f"locked={self.n_locked_bands}, active={self.is_activated})"
        )


class PhaseLockBridge(nn.Module):
    """
    Detects phase-locked relationships between FrequencyAttractors
    and gates cross-domain knowledge transfer.

    Parameters
    ----------
    n_bands : int
        Number of spectral bands to evaluate coherence over.
    min_locked_bands : int
        Minimum number of bands that must show coherence above
        ``band_threshold`` for activation (Engineering Script: ≥ 3).
    band_threshold : float
        Per-band coherence threshold for counting a band as "locked".
    activation_threshold : float
        Overall activation score threshold for bridge creation.
    """

    def __init__(
        self,
        n_bands: int = 8,
        min_locked_bands: int = 3,
        band_threshold: float = 0.5,
        activation_threshold: float = 0.6,
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.min_locked_bands = min_locked_bands
        self.band_threshold = band_threshold
        self.activation_threshold = activation_threshold

        # Learnable band importance weights
        self.band_weights = nn.Parameter(torch.ones(n_bands) / n_bands)

    # ── Core scoring ───────────────────────────────────────────────────

    def compute_band_coherences(
        self,
        source: FrequencyAttractor,
        target: FrequencyAttractor,
    ) -> torch.Tensor:
        """
        Per-band phase coherence between two attractors.

        Returns tensor of shape ``(n_bands,)`` with values in [-1, 1].
        """
        n = min(self.n_bands, source.n_bands, target.n_bands)
        src_phase = source.phase_signature[:n]
        tgt_phase = target.phase_signature[:n]

        # cos(Δphase) — 1.0 = perfectly locked, -1.0 = anti-phase
        band_coh = torch.cos(src_phase - tgt_phase)

        # Pad if fewer bands available
        if n < self.n_bands:
            pad = torch.zeros(self.n_bands - n, device=band_coh.device)
            band_coh = torch.cat([band_coh, pad])

        return band_coh

    def activation_score(
        self,
        band_coherences: torch.Tensor,
    ) -> float:
        """
        Weighted activation score from per-band coherences.

        Returns a scalar in [0, 1].
        """
        w = F.softmax(self.band_weights, dim=0)
        # Map coherence from [-1, 1] to [0, 1] for scoring
        coh_01 = (band_coherences + 1.0) / 2.0
        score = (w * coh_01).sum()
        return score.clamp(0, 1).item()

    def count_locked_bands(
        self,
        band_coherences: torch.Tensor,
    ) -> int:
        """Count bands exceeding the per-band threshold."""
        coh_01 = (band_coherences + 1.0) / 2.0
        return int((coh_01 >= self.band_threshold).sum().item())

    # ── High-level API ─────────────────────────────────────────────────

    def evaluate(
        self,
        source: FrequencyAttractor,
        target: FrequencyAttractor,
    ) -> BridgeCandidate:
        """
        Evaluate phase-lock between *source* and *target* attractors.

        Returns a ``BridgeCandidate`` with activation decision.
        """
        band_coh = self.compute_band_coherences(source, target)
        score = self.activation_score(band_coh)
        n_locked = self.count_locked_bands(band_coh)

        is_activated = (
            score >= self.activation_threshold
            and n_locked >= self.min_locked_bands
        )

        return BridgeCandidate(
            source=source,
            target=target,
            activation_score=score,
            band_coherences=band_coh,
            n_locked_bands=n_locked,
            is_activated=is_activated,
            metadata={
                "band_threshold": self.band_threshold,
                "activation_threshold": self.activation_threshold,
                "min_locked_bands": self.min_locked_bands,
            },
        )

    def find_bridges(
        self,
        attractors_a: List[FrequencyAttractor],
        attractors_b: List[FrequencyAttractor],
    ) -> List[BridgeCandidate]:
        """
        Find all activated bridge candidates between two sets of
        attractors (e.g. from different domains).
        """
        candidates: List[BridgeCandidate] = []
        for src in attractors_a:
            for tgt in attractors_b:
                cand = self.evaluate(src, tgt)
                if cand.is_activated:
                    candidates.append(cand)

        # Sort by activation score descending
        candidates.sort(key=lambda c: c.activation_score, reverse=True)
        return candidates

    # ── S2 output → attractor extraction (bridge to S3) ───────────────

    @staticmethod
    def extract_attractors_from_s2(
        st: SpectralTensor,
        n_bands: int = 8,
        domain: str = "unknown",
        prefix: str = "att",
    ) -> List[FrequencyAttractor]:
        """
        Convert an S2 SpectralTensor into a list of FrequencyAttractors
        (one per channel / sequence position).

        This is the bridge interface between S2 output and S3 attractor
        identification.  In Phase 2, this will be replaced by a learned
        attractor discovery module.
        """
        amp = st.amplitude
        phase = st.phase

        # Ensure at least 2D
        if amp.dim() == 1:
            amp = amp.unsqueeze(0)
            phase = phase.unsqueeze(0)

        # Flatten any batch dims: work with (N, d_model)
        if amp.dim() > 2:
            amp = amp.reshape(-1, amp.shape[-1])
            phase = phase.reshape(-1, phase.shape[-1])

        attractors: List[FrequencyAttractor] = []
        d = amp.shape[-1]
        band_size = max(1, d // n_bands)

        for i in range(amp.shape[0]):
            # Phase signature: mean phase per band
            ps_bands = []
            for b in range(n_bands):
                start = b * band_size
                end = start + band_size if b < n_bands - 1 else d
                ps_bands.append(phase[i, start:end].mean())
            phase_sig = torch.stack(ps_bands)

            attractors.append(
                FrequencyAttractor(
                    centroid=amp[i],
                    phase_signature=phase_sig,
                    amplitude_profile=amp[i],
                    stability=0.5,  # placeholder until S3 refines
                    domain=domain,
                    attractor_id=f"{prefix}_{i:04d}",
                    metadata={**st.metadata, "position": i},
                )
            )

        return attractors
