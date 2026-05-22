"""Base decoder interface for FBC ingest layer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseDecoder(ABC):
    """
    Abstract base class for all FBC decoders.

    Decoders convert raw bytes from various formats into
    canonical numpy arrays suitable for FBC pipeline ingestion.
    """

    @abstractmethod
    def decode(self, data: bytes, fmt: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode raw bytes into numpy array.

        Args:
            data: Raw bytes of the file.
            fmt: Format identifier (e.g., 'wav', 'png', 'csv').

        Returns:
            (array, metadata): Canonical float32 array and metadata dict.
        """
        ...

    @property
    @abstractmethod
    def supported_formats(self) -> set[str]:
        """Return set of supported format strings."""
        ...
