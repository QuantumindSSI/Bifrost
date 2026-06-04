"""
Riemannian Semantic Coherence Module

Implements learned Riemannian manifold structure on frequency attractor space
to enable semantic coherence measurement.

Exports:
    - RiemannianSemanticCoherence: Main coherence class
    - RiemannianMetricLearner: Learn metric tensor g_ij
    - GeodesicComputer: Compute shortest paths on manifold
    - CoherenceScorer: Map geodesic distances to coherence
    - ManifoldProjector: Project to 2D/3D for visualization
    - TripletSemanticLoss: Training loss for semantic similarity
    - SemanticCoherenceOutput: Dataclass for coherence outputs
    - create_triplets_from_labels: Helper for training data preparation

Example:
    >>> from bifrost.riemannian_coherence import RiemannianSemanticCoherence
    >>> coherence = RiemannianSemanticCoherence(d_model=768, metric_dim=64)
    >>> output = coherence(attractors_from_phase_lock)
    >>> print(output.coherence_scores)
"""

from .riemannian_coherence import (
    RiemannianSemanticCoherence,
    RiemannianMetricLearner,
    GeodesicComputer,
    CoherenceScorer,
    ManifoldProjector,
    TripletSemanticLoss,
    SemanticCoherenceOutput,
    create_triplets_from_labels,
)

__all__ = [
    "RiemannianSemanticCoherence",
    "RiemannianMetricLearner",
    "GeodesicComputer",
    "CoherenceScorer",
    "ManifoldProjector",
    "TripletSemanticLoss",
    "SemanticCoherenceOutput",
    "create_triplets_from_labels",
]
