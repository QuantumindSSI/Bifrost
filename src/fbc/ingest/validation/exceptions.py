"""Custom exceptions for the FBC ingest pipeline."""


class IngestException(Exception):
    """Base exception for the ingestion pipeline."""
    pass


class DecodingError(IngestException):
    """Raised when decoding a data format fails."""
    pass


class ValidationError(IngestException):
    """Raised when data fails schema or constraint validation."""
    pass


class NormalizationError(IngestException):
    """Raised when data normalization fails."""
    pass
