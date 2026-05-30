"""
Semantic Coherence Training for Bifrost

Connects phase coherence to semantic meaning through supervised learning.

This module provides:
    - PhaseCoherenceExtractor: Convert spectral phase to semantic features
    - SupervisedSemanticCoherenceLoss: Contrastive loss for semantic alignment
    - SemanticCoherenceHead: Classification on phase features
    - SemanticCoherenceTrainer: End-to-end training pipeline
    - train_semantic_coherence: High-level training function

Example:
    >>> from bifrost.semantic_coherence import train_semantic_coherence
    >>> trainer = train_semantic_coherence(
    ...     pipeline=bifrost_pipeline,
    ...     train_signals=audio_clips,
    ...     train_labels=emotion_labels,
    ...     num_classes=8,
    ... )
    >>> metrics = trainer.evaluate_semantic_coherence(test_signals, test_labels)
    >>> print(f"Semantic correlation: {metrics.coherence_semantic_correlation:.3f}")
"""

from .core import (
    SemanticCoherenceMetrics,
    PhaseCoherenceExtractor,
    SupervisedSemanticCoherenceLoss,
    SemanticCoherenceHead,
    SemanticCoherenceTrainer,
    train_semantic_coherence,
)

__all__ = [
    "SemanticCoherenceMetrics",
    "PhaseCoherenceExtractor",
    "SupervisedSemanticCoherenceLoss",
    "SemanticCoherenceHead",
    "SemanticCoherenceTrainer",
    "train_semantic_coherence",
]
