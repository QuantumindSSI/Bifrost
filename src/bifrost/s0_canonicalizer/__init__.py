"""Backward-compatibility shim — import from bifrost.canonicalizer instead."""

from bifrost.canonicalizer import SpectralCanonicalizer

S0Canonicalizer = SpectralCanonicalizer

__all__ = ["SpectralCanonicalizer", "S0Canonicalizer"]
