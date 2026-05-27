"""Backward-compatibility shim — import from bifrost.decomposer instead."""

from bifrost.decomposer import SpectralDecomposer

S1SpectralDecomposer = SpectralDecomposer

__all__ = ["SpectralDecomposer", "S1SpectralDecomposer"]
