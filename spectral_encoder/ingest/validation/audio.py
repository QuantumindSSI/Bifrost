"""Audio data validation."""

import numpy as np
from typing import Tuple
from .exceptions import ValidationError


class AudioValidator:
    """Validate audio data constraints."""

    MIN_SAMPLE_RATE = 8000
    MAX_SAMPLE_RATE = 48000
    MIN_DURATION_SEC = 0.01
    MAX_DURATION_SEC = 3600
    VALID_BIT_DEPTHS = {8, 16, 24, 32}
    MAX_CHANNELS = 8

    @classmethod
    def validate(cls, audio_array: np.ndarray, metadata: dict) -> Tuple[bool, str]:
        """
        Validate audio array and metadata.
        
        Args:
            audio_array: Audio numpy array
            metadata: Audio metadata dict
            
        Returns:
            (is_valid, message): Validation result and message
        """
        sample_rate = metadata.get("sample_rate")
        bit_depth = metadata.get("bit_depth")
        channels = metadata.get("channels")
        num_samples = metadata.get("num_samples")
        duration_sec = metadata.get("duration_sec", 0)

        if sample_rate is None or not cls.MIN_SAMPLE_RATE <= sample_rate <= cls.MAX_SAMPLE_RATE:
            return False, f"Invalid sample rate: {sample_rate} Hz"

        if bit_depth not in cls.VALID_BIT_DEPTHS:
            return False, f"Invalid bit depth: {bit_depth}"

        if channels is None or channels < 1 or channels > cls.MAX_CHANNELS:
            return False, f"Invalid channels: {channels}"

        if duration_sec < cls.MIN_DURATION_SEC:
            return False, f"Duration too short: {duration_sec:.3f}s (min {cls.MIN_DURATION_SEC}s)"

        if duration_sec > cls.MAX_DURATION_SEC:
            return False, f"Duration too long: {duration_sec:.1f}s (max {cls.MAX_DURATION_SEC}s)"

        if np.isnan(audio_array).any():
            return False, "Audio contains NaN values"

        if np.isinf(audio_array).any():
            return False, "Audio contains Inf values"

        return True, "Audio is valid"
