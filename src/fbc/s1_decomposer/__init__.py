"""Backward-compatibility shim — import from fbc.decomposer instead."""

from fbc.decomposer import SpectralDecomposer

S1SpectralDecomposer = SpectralDecomposer

__all__ = ["SpectralDecomposer", "S1SpectralDecomposer"]
