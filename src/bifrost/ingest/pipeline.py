"""IngestPipeline — unified decode + validate + normalize entry point."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np

from bifrost.ingest.decoders.audio import AudioDecoder
from bifrost.ingest.decoders.image import ImageDecoder
from bifrost.ingest.decoders.text import TextDecoder, TensorDecoder
from bifrost.ingest.validation.audio import AudioValidator
from bifrost.ingest.validation.image import ImageValidator
from bifrost.ingest.validation.exceptions import ValidationError, IngestException


class Modality(str, Enum):
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    TENSOR = "tensor"


class IngestPipeline:
    """Unified pipeline: raw bytes → validated float32 array.

    Handles:
        - Audio: WAV (scipy), MP3/FLAC/OGG (soundfile/librosa)
        - Images: PNG, JPEG, TIFF, BMP (Pillow)
        - Text: CSV, JSON, Parquet → numeric embeddings
        - Tensor: NPZ, HDF5 → direct float32 arrays

    Args:
        strict_validation: If True, raises ValidationError on invalid data.
                           If False, logs a warning and continues.

    Example:
        >>> pipeline = IngestPipeline()
        >>> with open("audio.wav", "rb") as f:
        ...     audio, meta = pipeline.ingest(f.read(), Modality.AUDIO, "wav")
        >>> print(f"{meta['duration_sec']:.1f}s @ {meta['sample_rate']} Hz")

        >>> with open("image.png", "rb") as f:
        ...     img, meta = pipeline.ingest(f.read(), Modality.IMAGE, "png")
        >>> print(f"{meta['width']}×{meta['height']} {meta['color_space']}")
    """

    def __init__(self, strict_validation: bool = True):
        self.strict_validation = strict_validation
        self._audio_decoder = AudioDecoder()
        self._image_decoder = ImageDecoder()
        self._text_decoder = TextDecoder()
        self._tensor_decoder = TensorDecoder()

    def ingest(
        self,
        data: bytes,
        modality: Modality,
        fmt: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode, validate, and return canonical float32 array.

        Args:
            data: Raw bytes of the media file.
            modality: Modality.AUDIO or Modality.IMAGE.
            fmt: Format string ('wav', 'mp3', 'flac', 'ogg', 'png', 'jpg', etc.)

        Returns:
            (array, metadata):
                - Audio: float32 (channels, samples) in [-1, 1]
                - Image: float32 (H, W, C) or (H, W) in [0, 1]

        Raises:
            IngestException: On decode failure.
            ValidationError: On validation failure (strict mode only).
        """
        modality = Modality(modality)

        if modality == Modality.AUDIO:
            array, meta = self._audio_decoder.decode(data, fmt)
            is_valid, message = AudioValidator.validate(array, meta)
        elif modality == Modality.IMAGE:
            array, meta = self._image_decoder.decode(data, fmt)
            is_valid, message = ImageValidator.validate(array, meta)
        elif modality == Modality.TEXT:
            array, meta = self._text_decoder.decode(data, fmt)
            is_valid, message = True, "Text validation passed"
        elif modality == Modality.TENSOR:
            array, meta = self._tensor_decoder.decode(data, fmt)
            is_valid, message = True, "Tensor validation passed"
        else:
            raise IngestException(f"Unsupported modality: {modality}")

        if not is_valid:
            if self.strict_validation:
                raise ValidationError(message)
            else:
                import warnings
                warnings.warn(f"Validation warning: {message}", stacklevel=2)

        return array, meta

    def ingest_batch(
        self,
        items: List[Tuple[bytes, Modality, str]],
    ) -> List[Tuple[np.ndarray | None, Dict[str, Any], str | None]]:
        """Ingest a batch of files, collecting errors per item.

        Args:
            items: List of (data, modality, fmt) tuples.

        Returns:
            List of (array_or_None, metadata, error_or_None) tuples.
            On success: (array, meta, None)
            On failure: (None, {}, error_message)
        """
        results = []
        for data, modality, fmt in items:
            try:
                array, meta = self.ingest(data, modality, fmt)
                results.append((array, meta, None))
            except Exception as e:
                results.append((None, {}, str(e)))
        return results
