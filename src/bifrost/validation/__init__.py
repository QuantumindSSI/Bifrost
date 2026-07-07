"""
Bifrost Validation Module

Provides empirical validation for phase coherence claims.
"""

from .empirical_validation import (
    PhaseCoherenceValidator,
    ValidationReport,
    run_empirical_validation,
)
from .phase_ablation import PhaseAblationHarness
from .phase_metrics import PhaseCoherenceSignalMetrics
from .scale_ablation import ScaleAblationHarness

__all__ = [
    "PhaseCoherenceValidator",
    "ValidationReport",
    "run_empirical_validation",
    "PhaseAblationHarness",
    "PhaseCoherenceSignalMetrics",
    "ScaleAblationHarness",
]
