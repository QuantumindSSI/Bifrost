"""Audio format decoders — WAV, MP3, FLAC, OGG.

Decoding strategy:
    - WAV:        scipy.io.wavfile (fast, no dependencies beyond scipy)
    - MP3/FLAC/OGG: soundfile (preferred) → librosa (fallback)
"""

from __future__ import annotations

import io
from typing import Any, Dict, Tuple

import numpy as np

from bifrost.ingest.validation.exceptions import DecodingError


class AudioDecoder:
    """Decode audio files to float32 numpy arrays.

    Supported formats: WAV, MP3, FLAC, OGG

    Example:
        >>> decoder = AudioDecoder()
        >>> with open("audio.wav", "rb") as f:
        ...     audio, meta = decoder.decode(f.read(), "wav")
        >>> print(f"{meta['duration_sec']:.2f}s @ {meta['sample_rate']} Hz")
    """

    SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}

    def supports_format(self, fmt: str) -> bool:
        return fmt.lower() in self.SUPPORTED_FORMATS

    def decode(self, data: bytes, fmt: str = "wav") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode raw audio bytes to float32 array.

        Args:
            data: Raw audio bytes.
            fmt: Format string — 'wav', 'mp3', 'flac', or 'ogg'.

        Returns:
            (audio_array, metadata):
                audio_array — float32, shape (channels, samples), range [-1, 1]
                metadata — dict with sample_rate, channels, duration_sec, etc.

        Raises:
            DecodingError: If format is unsupported or decoding fails.
        """
        fmt = fmt.lower()
        if not self.supports_format(fmt):
            raise DecodingError(f"Unsupported audio format: '{fmt}'. Supported: {self.SUPPORTED_FORMATS}")

        try:
            if fmt == "wav":
                return self._decode_wav(data)
            else:
                return self._decode_compressed(data, fmt)
        except DecodingError:
            raise
        except Exception as e:
            raise DecodingError(f"Failed to decode {fmt}: {e}") from e

    def _decode_wav(self, data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode WAV using scipy (fast path)."""
        try:
            from scipy.io import wavfile
        except ImportError:
            raise DecodingError("scipy is required for WAV decoding: pip install scipy")

        bio = io.BytesIO(data)
        try:
            sample_rate, audio = wavfile.read(bio)
        except Exception as e:
            raise DecodingError(f"scipy WAV decode failed: {e}") from e

        audio_f32 = _to_float32(audio)

        # Ensure (channels, samples) shape
        if audio_f32.ndim == 1:
            audio_f32 = audio_f32[np.newaxis, :]
        elif audio_f32.ndim == 2 and audio_f32.shape[0] > audio_f32.shape[1]:
            audio_f32 = audio_f32.T  # (samples, channels) → (channels, samples)

        meta = {
            "format": "wav",
            "sample_rate": int(sample_rate),
            "channels": audio_f32.shape[0],
            "num_samples": audio_f32.shape[1],
            "bit_depth": audio.dtype.itemsize * 8,
            "duration_sec": audio_f32.shape[1] / sample_rate,
        }
        return audio_f32, meta

    def _decode_compressed(self, data: bytes, fmt: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode MP3/FLAC/OGG — soundfile preferred, librosa fallback."""
        bio = io.BytesIO(data)

        # --- Try soundfile first (supports FLAC/OGG natively) -----------------
        try:
            import soundfile as sf
            audio_f32, sample_rate = sf.read(bio, dtype="float32", always_2d=True)
            audio_f32 = audio_f32.T  # soundfile returns (samples, channels)
            meta = {
                "format": fmt,
                "sample_rate": int(sample_rate),
                "channels": audio_f32.shape[0],
                "num_samples": audio_f32.shape[1],
                "bit_depth": 16,
                "duration_sec": audio_f32.shape[1] / sample_rate,
            }
            return audio_f32, meta
        except Exception:
            pass  # fall through to librosa

        # --- Librosa fallback (handles MP3 via audioread) ----------------------
        try:
            import librosa
            bio.seek(0)
            audio_f32, sample_rate = librosa.load(bio, sr=None, mono=False)
            if audio_f32.ndim == 1:
                audio_f32 = audio_f32[np.newaxis, :]
            meta = {
                "format": fmt,
                "sample_rate": int(sample_rate),
                "channels": audio_f32.shape[0],
                "num_samples": audio_f32.shape[1],
                "bit_depth": 16,
                "duration_sec": audio_f32.shape[1] / sample_rate,
            }
            return audio_f32.astype(np.float32), meta
        except ImportError:
            raise DecodingError(
                f"Cannot decode {fmt}: install soundfile or librosa.\n"
                f"  pip install soundfile   (FLAC, OGG)\n"
                f"  pip install librosa     (MP3, FLAC, OGG)"
            )
        except Exception as e:
            raise DecodingError(f"librosa decode failed for {fmt}: {e}") from e


def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Normalize integer audio to float32 in [-1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        return audio.astype(np.float32) / 2_147_483_648.0
    elif audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    elif audio.dtype in (np.float32, np.float64):
        return audio.astype(np.float32)
    else:
        raise DecodingError(f"Unsupported audio dtype: {audio.dtype}")
