"""Audio data validation — schema and constraint checks."""

import numpy as np
from typing import Tuple

from bifrost.ingest.validation.exceptions import ValidationError


class AudioValidator:
    """Validate audio array and metadata against production constraints.

    Constraints:
        - Sample rate: 8–48 kHz
        - Duration: 10 ms – 3600 s
        - Channels: 1–8
        - Bit depth: 8, 16, 24, or 32
        - No NaN or Inf values
    """

    MIN_SAMPLE_RATE = 8_000
    MAX_SAMPLE_RATE = 48_000
    MIN_DURATION_SEC = 0.01
    MAX_DURATION_SEC = 3_600.0
    VALID_BIT_DEPTHS = {8, 16, 24, 32}
    MAX_CHANNELS = 8

    @classmethod
    def validate(cls, audio_array: np.ndarray, metadata: dict) -> Tuple[bool, str]:
        """Validate audio array and metadata.

        Args:
            audio_array: Float32 numpy array of audio samples.
            metadata: Dict with keys: sample_rate, bit_depth, channels,
                      num_samples, duration_sec.

        Returns:
            (is_valid, message): True + "Audio is valid" or False + reason.
        """
        sample_rate = metadata.get("sample_rate")
        bit_depth = metadata.get("bit_depth")
        channels = metadata.get("channels")
        duration_sec = metadata.get("duration_sec", 0.0)

        if sample_rate is None or not (cls.MIN_SAMPLE_RATE <= sample_rate <= cls.MAX_SAMPLE_RATE):
            return False, f"Invalid sample rate: {sample_rate} Hz (must be {cls.MIN_SAMPLE_RATE}–{cls.MAX_SAMPLE_RATE})"

        if bit_depth not in cls.VALID_BIT_DEPTHS:
            return False, f"Invalid bit depth: {bit_depth} (must be one of {cls.VALID_BIT_DEPTHS})"

        if channels is None or not (1 <= channels <= cls.MAX_CHANNELS):
            return False, f"Invalid channels: {channels} (must be 1–{cls.MAX_CHANNELS})"

        if duration_sec < cls.MIN_DURATION_SEC:
            return False, f"Duration too short: {duration_sec:.3f}s (min {cls.MIN_DURATION_SEC}s)"

        if duration_sec > cls.MAX_DURATION_SEC:
            return False, f"Duration too long: {duration_sec:.1f}s (max {cls.MAX_DURATION_SEC}s)"

        if np.isnan(audio_array).any():
            return False, "Audio contains NaN values"

        if np.isinf(audio_array).any():
            return False, "Audio contains Inf values"

        return True, "Audio is valid"

    @classmethod
    def validate_strict(cls, audio_array: np.ndarray, metadata: dict) -> None:
        """Validate and raise ValidationError on failure.

        Args:
            audio_array: Float32 numpy array.
            metadata: Audio metadata dict.

        Raises:
            ValidationError: If validation fails.
        """
        is_valid, message = cls.validate(audio_array, metadata)
        if not is_valid:
            raise ValidationError(message)
