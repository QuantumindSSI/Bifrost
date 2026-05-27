"""Image format decoders — PNG, JPEG, TIFF, BMP."""

from __future__ import annotations

import io
from typing import Any, Dict, Tuple

import numpy as np

from bifrost.ingest.validation.exceptions import DecodingError


class ImageDecoder:
    """Decode image files to float32 numpy arrays in [0, 1] range.

    Supported formats: PNG, JPEG/JPG, TIFF/TIF, BMP

    Example:
        >>> decoder = ImageDecoder()
        >>> with open("photo.png", "rb") as f:
        ...     img, meta = decoder.decode(f.read(), "png")
        >>> print(f"{meta['width']}×{meta['height']} {meta['color_space']}")
    """

    SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp"}

    def supports_format(self, fmt: str) -> bool:
        return fmt.lower() in self.SUPPORTED_FORMATS

    def decode(self, data: bytes, fmt: str = "png") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode raw image bytes to float32 array.

        Args:
            data: Raw image bytes.
            fmt: Format string — 'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'.

        Returns:
            (image_array, metadata):
                image_array — float32, shape (H, W, C) or (H, W), range [0, 1]
                metadata — dict with width, height, channels, color_space, etc.

        Raises:
            DecodingError: If format is unsupported or decoding fails.
        """
        fmt = fmt.lower()
        if not self.supports_format(fmt):
            raise DecodingError(f"Unsupported image format: '{fmt}'. Supported: {self.SUPPORTED_FORMATS}")

        try:
            from PIL import Image
        except ImportError:
            raise DecodingError("Pillow is required for image decoding: pip install Pillow")

        try:
            bio = io.BytesIO(data)
            img = Image.open(bio)
            img.load()

            # Normalize mode
            mode_map = {
                "RGBA": ("rgba", 4),
                "RGB": ("rgb", 3),
                "L": ("grayscale", 1),
                "LA": ("grayscale_alpha", 2),
            }
            if img.mode not in mode_map:
                img = img.convert("RGB")
                img.mode = "RGB"

            color_space, channels = mode_map.get(img.mode, ("rgb", 3))
            width, height = img.size

            img_array = np.array(img, dtype=np.uint8)
            img_f32 = _to_float32(img_array)

            meta = {
                "format": fmt,
                "width": width,
                "height": height,
                "channels": channels,
                "bit_depth": 8,
                "color_space": color_space,
                "size_bytes": len(data),
            }
            return img_f32, meta

        except DecodingError:
            raise
        except Exception as e:
            raise DecodingError(f"Failed to decode {fmt}: {e}") from e


def _to_float32(image: np.ndarray) -> np.ndarray:
    """Convert image array to float32 in [0, 1]."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    elif image.dtype in (np.float32, np.float64):
        return np.clip(image.astype(np.float32), 0.0, 1.0)
    else:
        raise DecodingError(f"Unsupported image dtype: {image.dtype}")
