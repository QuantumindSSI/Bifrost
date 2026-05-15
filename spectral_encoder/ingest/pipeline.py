"""Main ingestion pipeline orchestrator."""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from enum import Enum

from .decoders.audio import AudioDecoder
from .decoders.image import ImageDecoder
from .validation.audio import AudioValidator
from .validation.image import ImageValidator
from .normalize import Normalizer
from .validation.exceptions import DecodingError, ValidationError


class Modality(Enum):
    """Data modality types."""
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    TENSOR = "tensor"


class IngestPipeline:
    """
    Unified data ingestion pipeline for all modalities.
    
    Flow:
        Raw bytes → Decoder → Validator → Normalizer → Output
    """

    def __init__(
        self,
        strict_validation: bool = False,
        repair_on_error: bool = True,
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            strict_validation: If True, reject invalid data; if False, attempt repair
            repair_on_error: Attempt to fix data issues (resample, impute, etc.)
        """
        self.strict_validation = strict_validation
        self.repair_on_error = repair_on_error
        
        self.audio_decoder = AudioDecoder()
        self.image_decoder = ImageDecoder()
        
    def ingest(
        self,
        data: bytes,
        modality: Modality,
        format_str: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ingest raw bytes through complete pipeline.
        
        Args:
            data: Raw bytes to ingest
            modality: Data modality (audio, image, text, tensor)
            format_str: Format identifier ("wav", "png", etc.)
            
        Returns:
            (array, metadata): Processed array and metadata dict
            
        Raises:
            DecodingError: If decoding fails
            ValidationError: If validation fails (strict mode)
        """
        if modality == Modality.AUDIO:
            return self._ingest_audio(data, format_str)
        elif modality == Modality.IMAGE:
            return self._ingest_image(data, format_str)
        else:
            raise NotImplementedError(f"Modality {modality} not yet implemented")

    def _ingest_audio(self, data: bytes, format_str: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ingest audio through pipeline."""
        audio, metadata = self.audio_decoder.decode(data, format_str)
        
        is_valid, msg = AudioValidator.validate(audio, metadata)
        if not is_valid:
            if self.strict_validation:
                raise ValidationError(f"Audio validation failed: {msg}")
            else:
                print(f"⚠️  Audio validation warning: {msg}")
        
        audio = Normalizer.normalize_audio(audio, metadata)
        return audio, metadata

    def _ingest_image(self, data: bytes, format_str: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ingest image through pipeline."""
        image, metadata = self.image_decoder.decode(data, format_str)
        
        is_valid, msg = ImageValidator.validate(image, metadata)
        if not is_valid:
            if self.strict_validation:
                raise ValidationError(f"Image validation failed: {msg}")
            else:
                print(f"⚠️  Image validation warning: {msg}")
        
        image = Normalizer.normalize_image(image, metadata)
        return image, metadata

    def ingest_from_file(
        self,
        file_path: str,
        modality: Modality,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ingest from file path.
        
        Args:
            file_path: Path to file
            modality: Data modality
            
        Returns:
            (array, metadata): Processed array and metadata
        """
        with open(file_path, "rb") as f:
            data = f.read()
        
        format_str = file_path.split(".")[-1].lower()
        return self.ingest(data, modality, format_str)

    def batch_ingest(
        self,
        file_paths: List[str],
        modality: Modality,
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Ingest multiple files.
        
        Args:
            file_paths: List of file paths
            modality: Data modality
            
        Returns:
            List of (array, metadata) tuples
        """
        results = []
        for path in file_paths:
            try:
                result = self.ingest_from_file(path, modality)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to ingest {path}: {str(e)}")
                continue
        return results
