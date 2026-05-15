"""Base decoder interface."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class Decoder(ABC):
    """Abstract base class for all decoders."""

    @abstractmethod
    def decode(self, data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode raw bytes into array and metadata.
        
        Args:
            data: Raw bytes to decode
            
        Returns:
            (array, metadata): Decoded array and metadata dict
            
        Raises:
            DecodingError: If decoding fails
        """
        pass

    @abstractmethod
    def supports_format(self, format_str: str) -> bool:
        """Check if decoder supports given format."""
        pass
