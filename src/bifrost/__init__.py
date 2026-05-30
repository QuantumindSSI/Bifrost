"""
Bifröst — The Spectral Rainbow Bridge

Frequency-Based Cognition core modules by Quantumind.
Implements the spectral pipeline: canonicalize → decompose → bind.
"""

__version__ = "0.1.1"

from .spectral_tensor import SpectralTensor
from .canonicalizer import SpectralCanonicalizer
from .decomposer import SpectralDecomposer
from .resonance_attention import ResonanceAttention, SpectralBinding
from .resonance_attention.harmonic_binding import (
    HarmonicBinding,
    HarmonicAttention,
    HarmonicFrequencyGrid,
)
from .phase_lock_bridge import PhaseLockBridge, FrequencyAttractor
from .pipeline import BifrostPipeline
from .bridge import bridge_to_canonicalizer
from .ingest import IngestPipeline, Modality as IngestModality
from .training import BifrostTrainer, NextFramePredictionLoss, train_fbc_simple
from .multimodal_pipeline import (
    MultiModalSpectralPipeline,
    Modality,
    TextSpectralEncoder,
    ImageSpectralDecomposer,
    TensorSpectralAdapter,
    create_multimodal_pipeline,
)
from .s1_decomposer.complex_decomposer import (
    ComplexSpectralDecomposer,
    ComplexSelectiveScan,
    ComplexLinear,
)
from .complex_training import (
    ComplexBifrostTrainer,
    ComplexNextStepLoss,
    PhaseCoherenceMetrics,
    train_complex_bifrost_simple,
    train_complex_fbc_simple,
)

# Backward compatibility aliases (FBC naming deprecated)
import warnings
from typing import Any


class _FBCPipeline(BifrostPipeline):
    """Deprecated alias for BifrostPipeline."""
    def __new__(cls, *args: Any, **kwargs: Any) -> "BifrostPipeline":
        warnings.warn(
            "FBCPipeline is deprecated, use BifrostPipeline instead. "
            "FBC naming will be removed in v0.2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return super().__new__(cls)


class _FBCTrainer(BifrostTrainer):
    """Deprecated alias for BifrostTrainer."""
    def __new__(cls, *args: Any, **kwargs: Any) -> "BifrostTrainer":
        warnings.warn(
            "FBCTrainer is deprecated, use BifrostTrainer instead. "
            "FBC naming will be removed in v0.2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return super().__new__(cls)


class _ComplexFBCTrainer(ComplexBifrostTrainer):
    """Deprecated alias for ComplexBifrostTrainer."""
    def __new__(cls, *args: Any, **kwargs: Any) -> "ComplexBifrostTrainer":
        warnings.warn(
            "ComplexFBCTrainer is deprecated, use ComplexBifrostTrainer instead. "
            "FBC naming will be removed in v0.2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return super().__new__(cls)


FBCPipeline = _FBCPipeline
FBCTrainer = _FBCTrainer
ComplexFBCTrainer = _ComplexFBCTrainer

__all__ = [
    "SpectralTensor",
    "SpectralCanonicalizer",
    "SpectralDecomposer",
    "ResonanceAttention",
    "SpectralBinding",
    "HarmonicBinding",
    "HarmonicAttention",
    "HarmonicFrequencyGrid",
    "PhaseLockBridge",
    "FrequencyAttractor",
    "BifrostPipeline",
    "FBCPipeline",
    "bridge_to_canonicalizer",
    "IngestPipeline",
    "IngestModality",
    "BifrostTrainer",
    "FBCTrainer",
    "NextFramePredictionLoss",
    "train_fbc_simple",
    "MultiModalSpectralPipeline",
    "Modality",
    "TextSpectralEncoder",
    "ImageSpectralDecomposer",
    "TensorSpectralAdapter",
    "create_multimodal_pipeline",
    "ComplexSpectralDecomposer",
    "ComplexSelectiveScan",
    "ComplexLinear",
    "ComplexBifrostTrainer",
    "ComplexFBCTrainer",
    "ComplexNextStepLoss",
    "PhaseCoherenceMetrics",
    "train_complex_fbc_simple",
    "train_complex_bifrost_simple",
]
