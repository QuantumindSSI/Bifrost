"""
FBC — Frequency-Based Cognition core modules.

Implements the S0–S4 pipeline stages from the FBC Engineering Script.
"""

__version__ = "0.1.1"

from .spectral_tensor import SpectralTensor
from .s0_canonicalizer import S0Canonicalizer
from .s1_decomposer import S1SpectralDecomposer
from .resonance_attention import ResonanceAttention, S2SpectralBinding
from .phase_lock_bridge import PhaseLockBridge, FrequencyAttractor
from .pipeline import FBCPipeline
from .bridge import bridge_to_s0
from .ingest import IngestPipeline, Modality

__all__ = [
    "SpectralTensor",
    "S0Canonicalizer",
    "S1SpectralDecomposer",
    "ResonanceAttention",
    "S2SpectralBinding",
    "PhaseLockBridge",
    "FrequencyAttractor",
    "FBCPipeline",
    "bridge_to_s0",
    "IngestPipeline",
    "Modality",
]
