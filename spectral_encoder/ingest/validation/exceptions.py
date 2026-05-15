"""Custom exceptions for data validation and decoding."""


class IngestException(Exception):
    """Base exception for ingestion pipeline."""
    pass


class DecodingError(IngestException):
    """Raised when decoding a data format fails."""
    pass


class ValidationError(IngestException):
    """Raised when data validation fails."""
    pass


class NormalizationError(IngestException):
    """Raised when data normalization fails."""
    pass
