"""TextDecoder — decode CSV, JSON, Parquet into numeric embeddings."""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .base import BaseDecoder


class TextDecoder(BaseDecoder):
    """
    Decode structured text formats (CSV, JSON, Parquet) into float32 arrays.

    Pipeline:
        1. Parse raw bytes → DataFrame or dict
        2. Convert to numeric representation
        3. Return (array, metadata) for Bifröst pipeline

    Supported formats:
        - csv: Comma-separated values
        - json: JSON records (array of objects)
        - parquet: Binary Parquet format

    Text embedding strategy:
        - Numeric columns: direct use
        - Text columns: character-level encoding + learned embedding (stub)
        - Categorical columns: one-hot encoding
    """

    @property
    def supported_formats(self) -> set[str]:
        """Return set of supported format strings."""
        return {"csv", "json", "parquet"}

    def decode(
        self,
        data: bytes,
        fmt: str,
        embedding_dim: int = 64,
        max_rows: int = 1000,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode text data into float32 array.

        Args:
            data: Raw bytes of text file.
            fmt: Format string ('csv', 'json', 'parquet').
            embedding_dim: Target dimension for text embeddings.
            max_rows: Maximum rows to process (for large files).

        Returns:
            (array, metadata):
                - array: float32 (features, samples) where samples = rows
                - metadata: dict with columns, dtypes, embedding info.

        Raises:
            ValueError: On parse failure or unsupported format.
        """
        fmt = fmt.lower()
        if fmt not in self.supported_formats:
            raise ValueError(f"Unsupported text format: {fmt}")

        # Parse to DataFrame
        if fmt == "csv":
            df = self._parse_csv(data, max_rows)
        elif fmt == "json":
            df = self._parse_json(data, max_rows)
        elif fmt == "parquet":
            df = self._parse_parquet(data, max_rows)

        # Convert to numeric embedding
        array, embedding_meta = self._dataframe_to_embedding(df, embedding_dim)

        metadata = {
            "format": fmt,
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "num_rows": len(df),
            "num_features": array.shape[0],
            "embedding": embedding_meta,
        }

        return array.astype(np.float32), metadata

    def _parse_csv(self, data: bytes, max_rows: int) -> pd.DataFrame:
        """Parse CSV bytes to DataFrame."""
        return pd.read_csv(io.BytesIO(data), nrows=max_rows)

    def _parse_json(self, data: bytes, max_rows: int) -> pd.DataFrame:
        """Parse JSON bytes (array of objects) to DataFrame."""
        records = json.loads(data.decode("utf-8"))
        if isinstance(records, dict):
            records = [records]
        return pd.DataFrame(records[:max_rows])

    def _parse_parquet(self, data: bytes, max_rows: int) -> pd.DataFrame:
        """Parse Parquet bytes to DataFrame."""
        return pd.read_parquet(io.BytesIO(data))

    def _dataframe_to_embedding(
        self, df: pd.DataFrame, target_dim: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert DataFrame to numeric embedding array.

        Strategy:
            - Numeric columns: z-score normalize
            - Categorical: one-hot encode (limited categories)
            - Text: character-level frequency encoding

        Returns:
            array: (features, samples) float32
            embedding_meta: dict describing encoding
        """
        features_list = []
        feature_names = []

        for col in df.columns:
            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                # Numeric: z-score normalize
                values = series.fillna(series.mean()).values
                std = values.std()
                if std > 0:
                    values = (values - values.mean()) / std
                features_list.append(values)
                feature_names.append(col)

            elif pd.api.types.is_categorical_dtype(series) or series.nunique() <= 10:
                # Categorical: one-hot (limited)
                dummies = pd.get_dummies(series, prefix=col)
                for dummy_col in dummies.columns:
                    features_list.append(dummies[dummy_col].values.astype(float))
                    feature_names.append(dummy_col)

            else:
                # Text: character-level frequency encoding
                text_features = self._text_to_features(series, target_dim // 4)
                for i in range(text_features.shape[1]):
                    features_list.append(text_features[:, i])
                    feature_names.append(f"{col}_charfreq_{i}")

        # Stack to (features, samples) - transpose for Bifröst spectral format
        if not features_list:
            raise ValueError("No features extracted from DataFrame")

        array = np.stack(features_list, axis=0)  # (features, samples)

        # Truncate or pad to target dimension
        if array.shape[0] > target_dim:
            array = array[:target_dim, :]
        elif array.shape[0] < target_dim:
            padding = np.zeros((target_dim - array.shape[0], array.shape[1]))
            array = np.concatenate([array, padding], axis=0)

        embedding_meta = {
            "strategy": "mixed_numeric_categorical_text",
            "original_columns": len(df.columns),
            "final_features": array.shape[0],
            "feature_names": feature_names[:target_dim],
        }

        return array, embedding_meta

    def _text_to_features(
        self, series: pd.Series, n_features: int
    ) -> np.ndarray:
        """
        Convert text series to character-level frequency features.

        Strategy:
            - Compute character frequency distribution
            - Use top N most common chars as features
        """
        # Simple char frequency encoding
        all_text = " ".join(series.fillna("").astype(str).str.lower())
        char_counts = {}
        for c in all_text:
            if c.isalnum() or c.isspace():
                char_counts[c] = char_counts.get(c, 0) + 1

        # Top N characters by frequency
        top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        top_chars = [c for c, _ in top_chars[:n_features]]

        if not top_chars:
            top_chars = ["a", "e", "i", "o", "u", " "][:n_features]

        # Compute per-row char frequencies
        features = []
        for text in series.fillna("").astype(str).str.lower():
            text_len = max(len(text), 1)
            freq = [text.count(c) / text_len for c in top_chars]
            features.append(freq)

        # Pad if needed
        result = np.array(features)
        if result.shape[1] < n_features:
            padding = np.zeros((result.shape[0], n_features - result.shape[1]))
            result = np.concatenate([result, padding], axis=1)

        return result


class TextTokenizer:
    """
    Tokenize raw text strings into numeric embeddings for Bifröst pipeline.

    Lightweight alternative to DataFrame-based decoding for:
        - Raw text documents
        - NLP preprocessing
        - Token-level frequency analysis
    """

    def __init__(self, vocab_size: int = 256, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Simple char-level vocab
        self.vocab = {chr(i): i for i in range(min(vocab_size, 256))}

    def tokenize(self, text: str) -> np.ndarray:
        """
        Tokenize text → (embedding_dim,) float32 array.

        Strategy:
            - Character-level encoding
            - Position-aware frequency distribution
            - Normalize to embedding_dim dimensions
        """
        text = text.lower()[:1000]  # Truncate long texts

        # Character frequency by position buckets
        bucket_size = max(len(text) // self.embedding_dim, 1)
        features = []

        for i in range(0, len(text), bucket_size):
            bucket = text[i : i + bucket_size]
            # Average char code in bucket
            avg_code = sum(ord(c) for c in bucket if c in self.vocab) / max(
                len(bucket), 1
            )
            features.append(avg_code / 255.0)  # Normalize

        # Pad or truncate to embedding_dim
        if len(features) < self.embedding_dim:
            features.extend([0.0] * (self.embedding_dim - len(features)))
        elif len(features) > self.embedding_dim:
            features = features[: self.embedding_dim]

        return np.array(features, dtype=np.float32)

    def tokenize_batch(self, texts: list[str]) -> np.ndarray:
        """Tokenize batch of texts → (batch, embedding_dim) array."""
        return np.stack([self.tokenize(t) for t in texts], axis=0)


class TensorDecoder(BaseDecoder):
    """
    Decode raw tensor formats (NPZ, HDF5) into numpy arrays.

    For direct tensor ingestion without text preprocessing.
    """

    @property
    def supported_formats(self) -> set[str]:
        """Return set of supported format strings."""
        return {"npz", "hdf5"}

    def decode(self, data: bytes, fmt: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode tensor file to numpy array.

        Args:
            data: Raw bytes of tensor file.
            fmt: Format string ('npz', 'hdf5').

        Returns:
            (array, metadata): float32 array and file metadata.
        """
        fmt = fmt.lower()

        if fmt == "npz":
            array, meta = self._decode_npz(data)
        elif fmt in ("hdf5", "h5"):
            array, meta = self._decode_hdf5(data)
        else:
            raise ValueError(f"Unsupported tensor format: {fmt}")

        return array.astype(np.float32), meta

    def _decode_npz(self, data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode NPZ (compressed numpy) format."""
        import io

        loaded = np.load(io.BytesIO(data))
        # Get first array if multiple
        keys = list(loaded.files)
        array = loaded[keys[0]]

        meta = {
            "format": "npz",
            "keys": keys,
            "shape": array.shape,
            "dtype": str(array.dtype),
        }
        return array, meta

    def _decode_hdf5(self, data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode HDF5 format."""
        import io

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 support. Install: pip install h5py")

        with h5py.File(io.BytesIO(data), "r") as f:
            array = None
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    array = f[key][()]
                    break

            if array is None:
                raise ValueError("No dataset found in HDF5 file")

            meta = {
                "format": "hdf5",
                "keys": list(f.keys()),
                "shape": array.shape,
                "dtype": str(array.dtype),
            }
            return array, meta

