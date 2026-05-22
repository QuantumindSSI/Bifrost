"""Tests for Phase 2: Text and Tensor modality support."""

import io
import json

import numpy as np
import pandas as pd
import pytest
import torch

from fbc.ingest import IngestPipeline, Modality, TextTokenizer
from fbc.ingest.decoders.text import TextDecoder, TensorDecoder
from fbc.bridge import bridge_to_s0
from fbc.pipeline import FBCPipeline


class TestTextDecoder:
    """Test text decoding for CSV, JSON, Parquet formats."""

    def test_decode_csv(self):
        """Test CSV text decoding."""
        decoder = TextDecoder()

        # Create sample CSV
        df = pd.DataFrame({
            'text': ['hello world', 'foo bar baz', 'test document'],
            'value': [1.0, 2.0, 3.0],
            'category': ['A', 'B', 'A'],
        })
        csv_bytes = df.to_csv(index=False).encode('utf-8')

        array, meta = decoder.decode(csv_bytes, 'csv')

        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert array.ndim == 2  # (features, samples)
        assert array.shape[1] == 3  # 3 rows
        assert meta['format'] == 'csv'
        assert meta['num_rows'] == 3

    def test_decode_json(self):
        """Test JSON text decoding."""
        decoder = TextDecoder()

        # Create sample JSON records
        records = [
            {'text': 'hello world', 'value': 1.0},
            {'text': 'another doc', 'value': 2.0},
        ]
        json_bytes = json.dumps(records).encode('utf-8')

        array, meta = decoder.decode(json_bytes, 'json')

        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert array.ndim == 2
        assert array.shape[1] == 2
        assert meta['format'] == 'json'

    def test_embedding_dim(self):
        """Test embedding dimension control."""
        decoder = TextDecoder()

        df = pd.DataFrame({'text': ['hello world'] * 5})
        csv_bytes = df.to_csv(index=False).encode('utf-8')

        array, _ = decoder.decode(csv_bytes, 'csv', embedding_dim=32)
        assert array.shape[0] == 32  # Should match target dim


class TestTextTokenizer:
    """Test TextTokenizer for raw text strings."""

    def test_tokenize_single(self):
        """Test single text tokenization."""
        tokenizer = TextTokenizer(vocab_size=256, embedding_dim=64)

        embedding = tokenizer.tokenize("hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (64,)

    def test_tokenize_batch(self):
        """Test batch tokenization."""
        tokenizer = TextTokenizer(vocab_size=256, embedding_dim=64)

        texts = ["hello", "world", "foo bar baz"]
        embeddings = tokenizer.tokenize_batch(texts)

        assert embeddings.shape == (3, 64)
        assert embeddings.dtype == np.float32

    def test_different_texts_produce_different_embeddings(self):
        """Verify different texts have different embeddings."""
        tokenizer = TextTokenizer(vocab_size=256, embedding_dim=64)

        embedding_a = tokenizer.tokenize("hello world")
        embedding_b = tokenizer.tokenize("completely different text")

        assert not np.allclose(embedding_a, embedding_b)


class TestTensorDecoder:
    """Test tensor file decoding (NPZ, HDF5)."""

    def test_decode_npz(self):
        """Test NPZ file decoding."""
        decoder = TensorDecoder()

        # Create sample NPZ
        array = np.random.randn(3, 64).astype(np.float32)
        buffer = io.BytesIO()
        np.savez(buffer, data=array)
        buffer.seek(0)

        result, meta = decoder.decode(buffer.read(), 'npz')

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_unsupported_format_raises(self):
        """Test that unsupported formats raise error."""
        decoder = TensorDecoder()

        with pytest.raises(ValueError, match="Unsupported"):
            decoder.decode(b'raw data', 'unsupported')


class TestTextBridge:
    """Test text data through bridge_to_s0."""

    def test_text_array_bridge(self):
        """Test bridge with text-decoded array."""
        decoder = TextDecoder()

        df = pd.DataFrame({
            'text': ['hello world', 'another document'],
            'value': [1.0, 2.0],
        })
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        array, meta = decoder.decode(csv_bytes, 'csv')

        # Bridge should handle text arrays
        meta['format'] = 'csv'
        signal, enriched_meta = bridge_to_s0(array, meta)

        assert isinstance(signal, torch.Tensor)
        assert signal.dtype == torch.float32
        assert signal.ndim == 2
        assert enriched_meta['modality'] == 'text'

    def test_raw_text_bridge(self):
        """Test bridge with raw string."""
        meta = {'format': 'txt'}
        signal, enriched_meta = bridge_to_s0("hello world test text", meta)

        assert isinstance(signal, torch.Tensor)
        assert signal.dtype == torch.float32
        assert enriched_meta['modality'] == 'text'


class TestTextPipeline:
    """Test full FBC pipeline with text data."""

    def test_text_through_full_pipeline(self):
        """Test complete pipeline: text → S0 → S1 → S2."""
        # Create sample text data
        decoder = TextDecoder()
        df = pd.DataFrame({
            'text': ['hello world'] * 5,
            'value': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        array, meta = decoder.decode(csv_bytes, 'csv', embedding_dim=64)

        # Bridge to S0
        meta['format'] = 'csv'
        signal, enriched_meta = bridge_to_s0(array, meta)

        # Run through FBC pipeline
        pipeline = FBCPipeline(
            n_fft_s0=64,
            n_fft_s1=32,
            d_model=64,
            use_mamba=False,  # CPU mode for testing
        )

        bound_st, coherence = pipeline(signal, enriched_meta)

        assert bound_st is not None
        assert coherence is not None
        assert bound_st.amplitude.ndim >= 2


class TestIngestPipelineText:
    """Test IngestPipeline with text modality."""

    def test_ingest_csv(self):
        """Test ingesting CSV via IngestPipeline."""
        pipeline = IngestPipeline()

        df = pd.DataFrame({
            'text': ['hello', 'world'],
            'value': [1.0, 2.0],
        })
        csv_bytes = df.to_csv(index=False).encode('utf-8')

        array, meta = pipeline.ingest(csv_bytes, Modality.TEXT, 'csv')

        assert isinstance(array, np.ndarray)
        assert meta['format'] == 'csv'

    def test_ingest_json(self):
        """Test ingesting JSON via IngestPipeline."""
        pipeline = IngestPipeline()

        records = [{'text': 'hello', 'value': 1.0}]
        json_bytes = json.dumps(records).encode('utf-8')

        array, meta = pipeline.ingest(json_bytes, Modality.TEXT, 'json')

        assert isinstance(array, np.ndarray)
        assert meta['format'] == 'json'
