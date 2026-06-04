"""
Bifrost Validation Module

Provides empirical validation for phase coherence claims.
"""

from .empirical_validation import (
    PhaseCoherenceValidator,
    ValidationReport,
    run_empirical_validation,
)

__all__ = [
    "PhaseCoherenceValidator",
    "ValidationReport", 
    "run_empirical_validation",
]
