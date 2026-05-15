"""Data normalization and type conversion."""

import numpy as np
from typing import Tuple
from .validation.exceptions import NormalizationError


class Normalizer:
    """Normalize data to canonical float32 ranges."""

    @staticmethod
    def normalize_audio(audio_array: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Ensure audio is float32 in [-1.0, 1.0] range.
        
        Args:
            audio_array: Audio array (may be any dtype)
            metadata: Audio metadata
            
        Returns:
            Normalized float32 array in [-1.0, 1.0]
        """
        audio = audio_array.astype(np.float32)
        
        max_val = np.abs(audio).max()
        
        if max_val > 0:
            if max_val < 1e-3:
                audio = audio * 10
        
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio

    @staticmethod
    def normalize_image(image_array: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Ensure image is float32 in [0.0, 1.0] range.
        
        Args:
            image_array: Image array (may be any dtype)
            metadata: Image metadata
            
        Returns:
            Normalized float32 array in [0.0, 1.0]
        """
        image = image_array.astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return image

    @staticmethod
    def normalize_tensor(tensor_array: np.ndarray, metadata: dict = None) -> np.ndarray:
        """
        Normalize generic tensor data.
        
        Args:
            tensor_array: Tensor array
            metadata: Tensor metadata (optional)
            
        Returns:
            Normalized float32 array
        """
        tensor = tensor_array.astype(np.float32)
        
        max_val = np.abs(tensor).max()
        if max_val > 0 and max_val > 1000:
            tensor = tensor / max_val
        elif max_val > 0 and max_val < 1e-3:
            tensor = tensor * 10
        
        return tensor
