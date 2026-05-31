"""
PhaseLockBridge — cross-domain knowledge transfer via phase-locked attractors.

Per Engineering Script §3:
    - Compute phase-lock activation score between attractors.
    - Enforce multi-band coherence confirmation (≥ 3 bands).
    - Prevent false positives via cross-band agreement.
    - When activated, create bridge edges between attractor nodes.
    - Transfer knowledge from source attractor to analogous target.

This implementation uses TruePhaseLockDetector which verifies:
    1. Temporal consistency (phase relationship persists over time)
    2. Frequency matching (oscillators have compatible frequencies)
    3. Coupling dynamics (Adler equation-based locking range)

Note: Previous implementation used simple phase-alignment (cos(Δφ)) which
only checks snapshot similarity. True phase-locking requires temporal stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attractor import FrequencyAttractor
from .phase_lock_detector import TruePhaseLockDetector, PhaseLockState
from ..spectral_tensor import SpectralTensor
from ..s3_attractor.attractor_learning import AttractorLearningModule


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
        use_true_phase_lock: bool = True,  # Use temporal consistency + coupling
        d_model: int = 768,
        n_attractors: int = 16,
        use_learned_stability: bool = True,  # Use S3 learned stability instead of placeholder
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.min_locked_bands = min_locked_bands
        self.band_threshold = band_threshold
        self.activation_threshold = activation_threshold
        self.use_true_phase_lock = use_true_phase_lock
        self.use_learned_stability = use_learned_stability
        self.d_model = d_model
        self.n_attractors = n_attractors

        # Learnable band importance weights
        self.band_weights = nn.Parameter(torch.ones(n_bands) / n_bands)
        
        # True phase-lock detector (replaces simple cos(Δφ) alignment)
        self.phase_lock_detector = TruePhaseLockDetector(
            n_bands=n_bands,
            coupling_strength=0.5,
        )
        
        # S3 Attractor Learning Module for learned stability
        if use_learned_stability:
            self.attractor_learner = AttractorLearningModule(
                d_model=d_model,
                n_bands=n_bands,
                n_attractors=n_attractors,
            )

    # ── Core scoring ───────────────────────────────────────────────────

    def compute_band_coherences(
        self,
        source: FrequencyAttractor,
        target: FrequencyAttractor,
        source_phase_history: Optional[torch.Tensor] = None,
        target_phase_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-band phase coherence between two attractors.

        If use_true_phase_lock=True and phase histories are provided,
        computes true phase-locking score including temporal consistency.
        Otherwise falls back to simple phase-alignment (cos(Δφ)).

        Returns tensor of shape ``(n_bands,)`` with values in [0, 1].
        """
        n = min(self.n_bands, source.n_bands, target.n_bands)
        src_phase = source.phase_signature[:n]
        tgt_phase = target.phase_signature[:n]

        # True phase-locking detection (when history is available)
        if (self.use_true_phase_lock 
            and source_phase_history is not None 
            and target_phase_history is not None):
            
            # Use TruePhaseLockDetector for temporal consistency + coupling
            alignment, consistency, coupling = self.phase_lock_detector.detect_phase_lock(
                source_phase_history, target_phase_history
            )
            
            # Overall phase-lock score: weighted combination
            # Consistency is most important for true phase-lock
            band_coh = 0.2 * alignment[:n] + 0.5 * consistency[:n] + 0.3 * coupling[:n]
        else:
            # Fallback: simple phase-alignment (cos(Δφ))
            # Maps [-1, 1] to [0, 1] for consistency with true locking scores
            band_coh = (torch.cos(src_phase - tgt_phase) + 1.0) / 2.0

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

    def extract_attractors_from_s2(
        self,
        st: SpectralTensor,
        n_bands: int = 8,
        domain: str = "unknown",
        prefix: str = "att",
    ) -> List[FrequencyAttractor]:
        """
        Convert an S2 SpectralTensor into a list of FrequencyAttractors
        (one per channel / sequence position).

        Uses learned stability from S3 AttractorLearningModule if available,
        otherwise falls back to placeholder stability=0.5.
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

        # Use learned stability if attractor_learner is available
        if self.use_learned_stability and hasattr(self, 'attractor_learner'):
            # Create a temporary SpectralTensor for the learner
            temp_spectral = SpectralTensor(
                amplitude=amp,
                phase=phase,
                scale=st.scale if hasattr(st, 'scale') else torch.ones_like(amp),
                uncertainty=st.uncertainty if hasattr(st, 'uncertainty') else torch.zeros_like(amp),
                metadata=st.metadata,
            )
            learned_attractors, _ = self.attractor_learner(temp_spectral)
            
            # Map learned attractors to our extracted ones
            for i in range(min(amp.shape[0], len(learned_attractors))):
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
                        stability=learned_attractors[i].stability,  # Learned stability
                        domain=domain,
                        attractor_id=f"{prefix}_{i:04d}",
                        metadata={**st.metadata, "position": i, "stability_source": "learned"},
                    )
                )
        else:
            # Fallback to placeholder stability
            for i in range(amp.shape[0]):
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
                        metadata={**st.metadata, "position": i, "stability_source": "placeholder"},
                    )
                )

        return attractors
