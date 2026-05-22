"""Sample data utilities for FBC users.

Quick-start example:
    >>> from fbc.data import quick_start_pipeline, list_samples
    >>> s0, s1, s2, attn = quick_start_pipeline("mono_16khz")
"""

from fbc.data.loader import (
    list_samples,
    load_sample_audio,
    load_sample_image,
    get_sample_path,
    quick_start_pipeline,
)

__all__ = [
    "list_samples",
    "load_sample_audio",
    "load_sample_image",
    "get_sample_path",
    "quick_start_pipeline",
]
