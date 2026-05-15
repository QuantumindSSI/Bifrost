"""Audio format decoders (WAV, MP3, FLAC, OGG)."""

import io
from typing import Tuple, Dict, Any
import numpy as np

from .base import Decoder
from ..validation.exceptions import DecodingError

try:
    import soundfile
except ImportError:
    soundfile = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None


class AudioDecoder(Decoder):
    """Decode audio files (WAV, MP3, FLAC, OGG) to float32 arrays."""

    SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}

    def supports_format(self, format_str: str) -> bool:
        return format_str.lower() in self.SUPPORTED_FORMATS

    def decode(self, data: bytes, format_str: str = "wav") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode audio bytes to float32 array.
        
        Args:
            data: Raw audio bytes
            format_str: Audio format ("wav", "mp3", "flac", "ogg")
            
        Returns:
            (audio_array, metadata): Audio as float32 array and metadata dict
            
        Raises:
            DecodingError: If decoding fails
        """
        if not self.supports_format(format_str):
            raise DecodingError(f"Unsupported format: {format_str}")

        try:
            format_lower = format_str.lower()
            
            if format_lower == "wav":
                return self._decode_wav(data)
            elif format_lower in {"mp3", "flac", "ogg"}:
                return self._decode_with_librosa(data, format_lower)
            else:
                raise DecodingError(f"Unknown format: {format_str}")
                
        except Exception as e:
            raise DecodingError(f"Failed to decode {format_str}: {str(e)}") from e

    def _decode_wav(self, data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode WAV using scipy.io.wavfile."""
        if wavfile is None:
            raise DecodingError("scipy not installed; cannot decode WAV")

        try:
            bio = io.BytesIO(data)
            sample_rate, audio_data = wavfile.read(bio)
        except Exception as e:
            raise DecodingError(f"scipy WAV decode failed: {e}") from e

        metadata = {
            "format": "wav",
            "sample_rate": int(sample_rate),
            "bit_depth": audio_data.dtype.itemsize * 8,
            "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            "num_samples": audio_data.shape[0],
        }

        audio_float32 = self._convert_to_float32(audio_data, audio_data.dtype)
        metadata["duration_sec"] = metadata["num_samples"] / sample_rate

        return audio_float32, metadata

    def _decode_with_librosa(self, data: bytes, format_str: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode MP3/FLAC/OGG using librosa."""
        if librosa is None:
            raise DecodingError("librosa not installed; cannot decode compressed audio")

        try:
            bio = io.BytesIO(data)
            audio_data, sample_rate = librosa.load(bio, sr=None, mono=False)
        except Exception as e:
            raise DecodingError(f"librosa {format_str} decode failed: {e}") from e

        if len(audio_data.shape) == 1:
            audio_data = audio_data[np.newaxis, :]

        metadata = {
            "format": format_str,
            "sample_rate": int(sample_rate),
            "bit_depth": 16,
            "channels": audio_data.shape[0],
            "num_samples": audio_data.shape[1],
            "duration_sec": audio_data.shape[1] / sample_rate,
        }

        return audio_data.astype(np.float32), metadata

    @staticmethod
    def _convert_to_float32(audio_data: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Convert audio to float32 in range [-1.0, 1.0]."""
        if dtype == np.int16:
            return audio_data.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            return audio_data.astype(np.float32) / 2147483648.0
        elif dtype == np.uint8:
            return (audio_data.astype(np.float32) - 128.0) / 128.0
        elif dtype == np.float32:
            return audio_data.astype(np.float32)
        elif dtype == np.float64:
            return audio_data.astype(np.float32)
        else:
            raise DecodingError(f"Unsupported audio dtype: {dtype}")
