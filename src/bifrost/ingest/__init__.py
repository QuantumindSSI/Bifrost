"""Bifröst Ingest layer — decode, validate, and normalize raw audio/image/text data.

Supports:
    - Audio: WAV, MP3, FLAC, OGG
    - Images: PNG, JPEG, TIFF, BMP
    - Text: CSV, JSON, Parquet
    - Tensors: NPZ, HDF5

Example:
    >>> from bifrost.ingest import IngestPipeline, Modality
    >>> pipeline = IngestPipeline()
    >>> with open("song.wav", "rb") as f:
    ...     audio, meta = pipeline.ingest(f.read(), Modality.AUDIO, "wav")
    >>> print(f"{meta['duration_sec']:.1f}s @ {meta['sample_rate']} Hz")
"""

from bifrost.ingest.pipeline import IngestPipeline, Modality
from bifrost.ingest.decoders.audio import AudioDecoder
from bifrost.ingest.decoders.image import ImageDecoder
from bifrost.ingest.decoders.text import TextDecoder, TextTokenizer, TensorDecoder
from bifrost.ingest.validation.exceptions import IngestException, DecodingError, ValidationError
from bifrost.ingest.validation.audio import AudioValidator
from bifrost.ingest.validation.image import ImageValidator

__all__ = [
    "IngestPipeline",
    "Modality",
    "AudioDecoder",
    "ImageDecoder",
    "TextDecoder",
    "TextTokenizer",
    "TensorDecoder",
    "AudioValidator",
    "ImageValidator",
    "IngestException",
    "DecodingError",
    "ValidationError",
]
