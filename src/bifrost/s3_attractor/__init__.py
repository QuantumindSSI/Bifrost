"""Learned attractor dynamics module.

Provides AttractorLearningModule for stability prediction and temporal tracking
of frequency attractors in the phase-lock bridge.
"""

from .attractor_learning import AttractorLearningModule, FrequencyAttractor

__all__ = ["AttractorLearningModule", "FrequencyAttractor"]
