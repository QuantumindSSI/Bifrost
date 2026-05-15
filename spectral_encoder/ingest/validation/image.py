"""Image data validation."""

import numpy as np
from typing import Tuple
from .exceptions import ValidationError


class ImageValidator:
    """Validate image data constraints."""

    MIN_WIDTH = 16
    MIN_HEIGHT = 16
    MAX_WIDTH = 8192
    MAX_HEIGHT = 8192
    VALID_CHANNELS = {1, 3, 4}
    VALID_BIT_DEPTHS = {8, 16}

    @classmethod
    def validate(cls, image_array: np.ndarray, metadata: dict) -> Tuple[bool, str]:
        """
        Validate image array and metadata.
        
        Args:
            image_array: Image numpy array
            metadata: Image metadata dict
            
        Returns:
            (is_valid, message): Validation result and message
        """
        width = metadata.get("width")
        height = metadata.get("height")
        channels = metadata.get("channels")
        bit_depth = metadata.get("bit_depth")

        if width is None or width < cls.MIN_WIDTH or width > cls.MAX_WIDTH:
            return False, f"Invalid width: {width} (min {cls.MIN_WIDTH}, max {cls.MAX_WIDTH})"

        if height is None or height < cls.MIN_HEIGHT or height > cls.MAX_HEIGHT:
            return False, f"Invalid height: {height} (min {cls.MIN_HEIGHT}, max {cls.MAX_HEIGHT})"

        if channels not in cls.VALID_CHANNELS:
            return False, f"Invalid channels: {channels} (must be 1, 3, or 4)"

        if bit_depth not in cls.VALID_BIT_DEPTHS:
            return False, f"Invalid bit depth: {bit_depth} (must be 8 or 16)"

        if image_array.size == 0:
            return False, "Image array is empty"

        if np.isnan(image_array).any():
            return False, "Image contains NaN values"

        if np.isinf(image_array).any():
            return False, "Image contains Inf values"

        return True, "Image is valid"
