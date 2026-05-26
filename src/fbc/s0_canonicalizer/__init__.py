"""Backward-compatibility shim — import from fbc.canonicalizer instead."""

from fbc.canonicalizer import SpectralCanonicalizer

S0Canonicalizer = SpectralCanonicalizer

__all__ = ["SpectralCanonicalizer", "S0Canonicalizer"]
