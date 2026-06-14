"""Spectral Decomposition stage."""

from .decomposer import SpectralDecomposer
from .complex_decomposer import (
    ComplexSpectralDecomposer,
    ComplexSelectiveScan,
    ComplexLinear,
)
from .spectral_normalization import (
    ComplexSpectralNorm,
    SpectralNormalizedComplexLinear,
    apply_spectral_norm_to_module,
)
from .associative_scan import associative_scan, blelloch_scan
from .complex_ssm_triton import (
    complex_selective_scan_cuda,
    complex_selective_scan_triton,
    select_complex_scan_backend,
)

__all__ = [
    "SpectralDecomposer",
    "ComplexSpectralDecomposer",
    "ComplexSelectiveScan",
    "ComplexLinear",
    "ComplexSpectralNorm",
    "SpectralNormalizedComplexLinear",
    "apply_spectral_norm_to_module",
    "associative_scan",
    "blelloch_scan",
    "complex_selective_scan_cuda",
    "complex_selective_scan_triton",
    "select_complex_scan_backend",
]
