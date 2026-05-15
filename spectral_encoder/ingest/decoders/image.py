"""Image format decoders (PNG, JPEG, TIFF, BMP)."""

import io
from typing import Tuple, Dict, Any
import numpy as np

from .base import Decoder
from ..validation.exceptions import DecodingError

try:
    from PIL import Image
except ImportError:
    Image = None


class ImageDecoder(Decoder):
    """Decode image files (PNG, JPEG, TIFF, BMP) to float32 arrays."""

    SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp"}

    def supports_format(self, format_str: str) -> bool:
        return format_str.lower() in self.SUPPORTED_FORMATS

    def decode(self, data: bytes, format_str: str = "png") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode image bytes to float32 array in [0, 1] range.
        
        Args:
            data: Raw image bytes
            format_str: Image format ("png", "jpg", "tiff", "bmp")
            
        Returns:
            (image_array, metadata): Image as float32 array and metadata dict
            
        Raises:
            DecodingError: If decoding fails
        """
        if not self.supports_format(format_str):
            raise DecodingError(f"Unsupported format: {format_str}")

        if Image is None:
            raise DecodingError("Pillow not installed; cannot decode images")

        try:
            bio = io.BytesIO(data)
            img = Image.open(bio)
            img.load()
            
            width, height = img.size
            
            if img.mode == "RGBA":
                channels = 4
                color_space = "rgba"
            elif img.mode == "RGB":
                channels = 3
                color_space = "rgb"
            elif img.mode == "L":
                channels = 1
                color_space = "grayscale"
            elif img.mode == "LA":
                channels = 2
                color_space = "grayscale_alpha"
            else:
                img = img.convert("RGB")
                channels = 3
                color_space = "rgb"
            
            img_array = np.array(img, dtype=np.uint8)
            
            metadata = {
                "format": format_str.lower(),
                "width": width,
                "height": height,
                "channels": channels,
                "bit_depth": 8,
                "color_space": color_space,
                "size_bytes": len(data),
            }
            
            img_float32 = self._convert_to_float32(img_array, img_array.dtype)
            
            return img_float32, metadata
            
        except Exception as e:
            raise DecodingError(f"Failed to decode {format_str}: {str(e)}") from e

    @staticmethod
    def _convert_to_float32(image_data: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Convert image to float32 in range [0.0, 1.0]."""
        if dtype == np.uint8:
            return image_data.astype(np.float32) / 255.0
        elif dtype == np.uint16:
            return image_data.astype(np.float32) / 65535.0
        elif dtype == np.float32:
            return np.clip(image_data, 0.0, 1.0)
        elif dtype == np.float64:
            return np.clip(image_data.astype(np.float32), 0.0, 1.0)
        else:
            raise DecodingError(f"Unsupported image dtype: {dtype}")
