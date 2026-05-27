"""Image data validation — schema and constraint checks."""

import numpy as np
from typing import Tuple

from bifrost.ingest.validation.exceptions import ValidationError


class ImageValidator:
    """Validate image array and metadata against production constraints.

    Constraints:
        - Width: 16–8192 px
        - Height: 16–8192 px
        - Channels: 1 (gray), 3 (RGB), or 4 (RGBA)
        - Bit depth: 8 or 16
        - No NaN or Inf values
        - Non-empty array
    """

    MIN_WIDTH = 16
    MIN_HEIGHT = 16
    MAX_WIDTH = 8_192
    MAX_HEIGHT = 8_192
    VALID_CHANNELS = {1, 3, 4}
    VALID_BIT_DEPTHS = {8, 16}

    @classmethod
    def validate(cls, image_array: np.ndarray, metadata: dict) -> Tuple[bool, str]:
        """Validate image array and metadata.

        Args:
            image_array: Float32 numpy array of image pixels.
            metadata: Dict with keys: width, height, channels, bit_depth.

        Returns:
            (is_valid, message): True + "Image is valid" or False + reason.
        """
        width = metadata.get("width")
        height = metadata.get("height")
        channels = metadata.get("channels")
        bit_depth = metadata.get("bit_depth")

        if width is None or not (cls.MIN_WIDTH <= width <= cls.MAX_WIDTH):
            return False, f"Invalid width: {width} (must be {cls.MIN_WIDTH}–{cls.MAX_WIDTH})"

        if height is None or not (cls.MIN_HEIGHT <= height <= cls.MAX_HEIGHT):
            return False, f"Invalid height: {height} (must be {cls.MIN_HEIGHT}–{cls.MAX_HEIGHT})"

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

    @classmethod
    def validate_strict(cls, image_array: np.ndarray, metadata: dict) -> None:
        """Validate and raise ValidationError on failure.

        Args:
            image_array: Float32 numpy array.
            metadata: Image metadata dict.

        Raises:
            ValidationError: If validation fails.
        """
        is_valid, message = cls.validate(image_array, metadata)
        if not is_valid:
            raise ValidationError(message)
