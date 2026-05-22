"""FBC Ingest layer — decode, validate, and normalize raw audio/image/text data.

Supports:
    - Audio: WAV, MP3, FLAC, OGG
    - Images: PNG, JPEG, TIFF, BMP
    - Text: CSV, JSON, Parquet
    - Tensors: NPZ, HDF5

Example:
    >>> from fbc.ingest import IngestPipeline, Modality
    >>> pipeline = IngestPipeline()
    >>> with open("song.wav", "rb") as f:
    ...     audio, meta = pipeline.ingest(f.read(), Modality.AUDIO, "wav")
    >>> print(f"{meta['duration_sec']:.1f}s @ {meta['sample_rate']} Hz")
"""

from fbc.ingest.pipeline import IngestPipeline, Modality
from fbc.ingest.decoders.audio import AudioDecoder
from fbc.ingest.decoders.image import ImageDecoder
from fbc.ingest.decoders.text import TextDecoder, TextTokenizer, TensorDecoder
from fbc.ingest.validation.exceptions import IngestException, DecodingError, ValidationError
from fbc.ingest.validation.audio import AudioValidator
from fbc.ingest.validation.image import ImageValidator

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
