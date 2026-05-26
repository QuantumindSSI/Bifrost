"""
FBC — Frequency-Based Cognition core modules.

Implements the FBC pipeline: canonicalize → decompose → bind.
"""

__version__ = "0.1.1"

from .spectral_tensor import SpectralTensor
from .canonicalizer import SpectralCanonicalizer
from .decomposer import SpectralDecomposer
from .resonance_attention import ResonanceAttention, SpectralBinding
from .phase_lock_bridge import PhaseLockBridge, FrequencyAttractor
from .pipeline import FBCPipeline
from .bridge import bridge_to_canonicalizer
from .ingest import IngestPipeline, Modality

__all__ = [
    "SpectralTensor",
    "SpectralCanonicalizer",
    "SpectralDecomposer",
    "ResonanceAttention",
    "SpectralBinding",
    "PhaseLockBridge",
    "FrequencyAttractor",
    "FBCPipeline",
    "bridge_to_canonicalizer",
    "IngestPipeline",
    "Modality",
]
