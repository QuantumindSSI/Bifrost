"""
Ingest → Canonicalizer Bridge Adapter.

Converts the output of ``fbc.ingest.pipeline.IngestPipeline``
into the canonical format expected by ``fbc.canonicalizer.SpectralCanonicalizer``:
    - np.ndarray → torch.Tensor (float32)
    - Channel axis normalised to channels-first: (channels, samples)
    - Metadata enriched with ``channel_axis`` field.
    - Text modalities (raw strings) converted via TextTokenizer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from bifrost.ingest.decoders.text import TextTokenizer


def bridge_to_canonicalizer(
    data: Any,
    metadata: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Adapt ingest output for canonicalization.

    Parameters
    ----------
    data : np.ndarray | list | dict
        Raw output from ``IngestPipeline.ingest()``.
    metadata : dict
        Metadata dict from the ingest layer.

    Returns
    -------
    signal : torch.Tensor
        Float32 tensor shaped ``(channels, samples)`` for audio / 1-D,
        or ``(channels, H*W)`` for images (rows flattened into a signal).
    metadata : dict
        Enriched copy of the input metadata.

    Raises
    ------
    TypeError
        If *data* is not a numeric array (e.g. text/JSON).
    ValueError
        If the array shape cannot be resolved.
    """
    modality = metadata.get("format", "").lower()
    content_type = metadata.get("content_type", "").lower()

    meta = dict(metadata)  # work on a copy

    # ---- text path --------------------------------------------------------
    if _is_text(meta) or isinstance(data, (str, list)):
        signal, meta = _canonicalize_text(data, meta)

    # ---- audio path -------------------------------------------------------
    elif _is_audio(meta):
        signal, meta = _canonicalize_audio(data, meta)

    # ---- image path -------------------------------------------------------
    elif _is_image(meta):
        signal, meta = _canonicalize_image(data, meta)

    # ---- generic tensor path ----------------------------------------------
    elif isinstance(data, np.ndarray):
        signal, meta = _canonicalize_tensor(data, meta)

    # ---- reject truly unsupported data ------------------------------------
    else:
        raise TypeError(
            f"bridge_to_canonicalizer requires NumPy array or text input, got {type(data).__name__}. "
            f"Text / structured data must be embedded into a numeric array "
            f"before entering the FBC pipeline."
        )

    return signal, meta


bridge_to_s0 = bridge_to_canonicalizer


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def _is_audio(meta: Dict[str, Any]) -> bool:
    return meta.get("format") in {"wav", "mp3", "flac", "ogg"} or "sample_rate" in meta


def _canonicalize_audio(
    data: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Ensure audio is ``(channels, samples)`` float32.

    scipy WAV returns ``(samples,)`` for mono and ``(samples, channels)``
    for stereo.  librosa returns ``(channels, samples)``.
    """
    arr = data.astype(np.float32) if data.dtype != np.float32 else data
    channels = meta.get("channels", 1)

    if arr.ndim == 1:
        # mono → (1, samples)
        arr = arr[np.newaxis, :]

    elif arr.ndim == 2:
        # Determine axis order:
        # scipy: (samples, channels) — samples >> channels
        # librosa: (channels, samples) — channels << samples
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 8:
            # (samples, channels) → transpose to (channels, samples)
            arr = arr.T
        # else: already (channels, samples)

    meta["channel_axis"] = 0
    meta["sample_axis"] = 1
    meta["channels"] = arr.shape[0]
    meta["num_samples"] = arr.shape[1]

    return torch.from_numpy(arr), meta


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def _is_image(meta: Dict[str, Any]) -> bool:
    return meta.get("format") in {"png", "jpg", "jpeg", "tiff", "tif", "bmp"} or "color_space" in meta


def _canonicalize_image(
    data: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Flatten image spatial dims into a 1-D signal per channel.

    Input layouts:
        - Grayscale: ``(H, W)`` → ``(1, H*W)``
        - Color:     ``(H, W, C)`` → ``(C, H*W)``
    """
    arr = data.astype(np.float32) if data.dtype != np.float32 else data

    if arr.ndim == 2:
        # (H, W) grayscale → (1, H*W)
        h, w = arr.shape
        arr = arr.reshape(1, h * w)
    elif arr.ndim == 3:
        # (H, W, C) → (C, H*W)
        h, w, c = arr.shape
        arr = arr.transpose(2, 0, 1).reshape(c, h * w)
    else:
        raise ValueError(f"Unexpected image ndim={arr.ndim}, shape={arr.shape}")

    meta["channel_axis"] = 0
    meta["sample_axis"] = 1
    meta["channels"] = arr.shape[0]
    meta["num_samples"] = arr.shape[1]
    meta["original_spatial"] = (meta.get("height", "?"), meta.get("width", "?"))

    return torch.from_numpy(arr), meta


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _is_text(meta: Dict[str, Any]) -> bool:
    """Check if metadata indicates text modality."""
    return meta.get("format") in {"csv", "json", "parquet", "txt"} or meta.get("embedding") is not None


def _canonicalize_text(
    data: Union[np.ndarray, str, list],
    meta: Dict[str, Any],
    embedding_dim: int = 64,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Convert text data to canonical (features, samples) float32 tensor.

    Handles:
        - Already-processed numeric arrays from TextDecoder
        - Raw string documents (via TextTokenizer)
        - List of strings
    """
    # Case 1: Already numeric from TextDecoder
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float32) if data.dtype != np.float32 else data

        # Ensure shape is (features, samples)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim > 2:
            # Flatten all but last dim
            arr = arr.reshape(-1, arr.shape[-1])

        meta["channel_axis"] = 0
        meta["sample_axis"] = 1
        meta["channels"] = arr.shape[0]
        meta["num_samples"] = arr.shape[1]
        meta["modality"] = "text"
        return torch.from_numpy(arr), meta

    # Case 2: Raw text strings - use tokenizer
    tokenizer = TextTokenizer(vocab_size=256, embedding_dim=embedding_dim)

    if isinstance(data, str):
        texts = [data]
    elif isinstance(data, list):
        texts = data
    else:
        raise TypeError(f"Text data must be str or list, got {type(data).__name__}")

    # Tokenize each text to (embedding_dim,) then stack to (embedding_dim, n_texts)
    embeddings = tokenizer.tokenize_batch(texts)
    arr = embeddings.T  # (features, samples) for FBC format

    meta["channel_axis"] = 0
    meta["sample_axis"] = 1
    meta["channels"] = arr.shape[0]
    meta["num_samples"] = arr.shape[1]
    meta["modality"] = "text"
    meta["text_tokenizer"] = "char_level"

    return torch.from_numpy(arr), meta


# ---------------------------------------------------------------------------
# Generic tensor
# ---------------------------------------------------------------------------

def _canonicalize_tensor(
    data: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Best-effort canonicalization for arbitrary tensors.

    If ``metadata["channel_axis"]`` is provided, use it.
    Otherwise assume the last axis is the sample/time axis and the
    preceding axes are batch/channel.
    """
    arr = data.astype(np.float32) if data.dtype != np.float32 else data

    channel_axis = meta.get("channel_axis")

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim >= 2 and channel_axis is not None:
        # Move channel_axis to position 0
        if channel_axis != 0:
            arr = np.moveaxis(arr, channel_axis, 0)
    elif arr.ndim >= 2:
        # Heuristic: if last dim is small (≤ 8), treat it as channels
        if arr.shape[-1] <= 8 and arr.shape[-1] < arr.shape[-2]:
            arr = np.moveaxis(arr, -1, 0)

    meta["channel_axis"] = 0
    meta["sample_axis"] = arr.ndim - 1
    meta["channels"] = arr.shape[0]
    meta["num_samples"] = arr.shape[-1]

    return torch.from_numpy(np.ascontiguousarray(arr)), meta
