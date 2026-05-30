"""
Bifrost Validation Module

Provides empirical validation for phase coherence claims.
Per Agentic CTO-Persona policy: all scientific claims must be traceable to evidence.
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
