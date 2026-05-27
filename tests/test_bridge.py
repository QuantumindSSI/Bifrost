"""Tests for the Ingest → S0 bridge adapter."""

import pytest
import numpy as np
import torch

from bifrost.bridge import bridge_to_canonicalizer


class TestBridgeAudio:
    def test_mono_1d(self):
        """Mono audio (samples,) → (1, samples)."""
        data = np.random.randn(16000).astype(np.float32)
        meta = {"format": "wav", "channels": 1, "sample_rate": 16000}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (1, 16000)
        assert m["channel_axis"] == 0
        assert m["channels"] == 1

    def test_stereo_scipy_layout(self):
        """Stereo scipy WAV (samples, 2) → (2, samples)."""
        data = np.random.randn(16000, 2).astype(np.float32)
        meta = {"format": "wav", "channels": 2, "sample_rate": 44100}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (2, 16000)
        assert m["channel_axis"] == 0
        assert m["channels"] == 2

    def test_stereo_librosa_layout(self):
        """Librosa layout (2, samples) stays (2, samples)."""
        data = np.random.randn(2, 16000).astype(np.float32)
        meta = {"format": "mp3", "channels": 2, "sample_rate": 44100}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (2, 16000)

    def test_metadata_enriched(self):
        data = np.random.randn(8000).astype(np.float32)
        meta = {"format": "wav", "channels": 1, "sample_rate": 8000}
        _, m = bridge_to_canonicalizer(data, meta)
        assert "channel_axis" in m
        assert "sample_axis" in m
        assert "num_samples" in m

    def test_output_is_torch(self):
        data = np.random.randn(8000).astype(np.float32)
        meta = {"format": "wav", "channels": 1, "sample_rate": 8000}
        signal, _ = bridge_to_canonicalizer(data, meta)
        assert isinstance(signal, torch.Tensor)
        assert signal.dtype == torch.float32


class TestBridgeImage:
    def test_grayscale(self):
        """(H, W) → (1, H*W)."""
        data = np.random.rand(64, 64).astype(np.float32)
        meta = {"format": "png", "height": 64, "width": 64, "channels": 1, "color_space": "grayscale"}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (1, 64 * 64)
        assert m["channels"] == 1

    def test_rgb(self):
        """(H, W, 3) → (3, H*W)."""
        data = np.random.rand(32, 32, 3).astype(np.float32)
        meta = {"format": "png", "height": 32, "width": 32, "channels": 3, "color_space": "rgb"}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (3, 32 * 32)
        assert m["channels"] == 3

    def test_rgba(self):
        """(H, W, 4) → (4, H*W)."""
        data = np.random.rand(16, 16, 4).astype(np.float32)
        meta = {"format": "png", "height": 16, "width": 16, "channels": 4, "color_space": "rgba"}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (4, 16 * 16)
        assert m["channels"] == 4

    def test_spatial_metadata_preserved(self):
        data = np.random.rand(100, 200, 3).astype(np.float32)
        meta = {"format": "jpg", "height": 100, "width": 200, "channels": 3}
        _, m = bridge_to_canonicalizer(data, meta)
        assert m["original_spatial"] == (100, 200)


class TestBridgeTensor:
    def test_1d_tensor(self):
        data = np.random.randn(500).astype(np.float32)
        meta = {"format": "npy", "shape": (500,)}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (1, 500)

    def test_2d_channels_last_heuristic(self):
        """(1000, 3) → should transpose to (3, 1000)."""
        data = np.random.randn(1000, 3).astype(np.float32)
        meta = {"format": "npy", "shape": (1000, 3)}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (3, 1000)

    def test_2d_explicit_channel_axis(self):
        """Respect metadata channel_axis."""
        data = np.random.randn(4, 256).astype(np.float32)
        meta = {"format": "npy", "shape": (4, 256), "channel_axis": 0}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (4, 256)

    def test_2d_channels_first_no_transpose(self):
        """(4, 10000) — first dim small, already channels-first."""
        data = np.random.randn(4, 10000).astype(np.float32)
        meta = {"format": "npy", "shape": (4, 10000)}
        signal, m = bridge_to_canonicalizer(data, meta)
        assert signal.shape == (4, 10000)


class TestBridgeRejectsNonNumeric:
    """Test that truly invalid data is rejected."""

    def test_rejects_dict(self):
        """Dicts are not valid input - no embedding possible."""
        with pytest.raises(TypeError, match="must be str or list"):
            bridge_to_canonicalizer({"key": "value"}, {"format": "json"})


class TestBridgeText:
    """Test Phase 2: Text modality support in bridge."""

    def test_accepts_string_with_text_format(self):
        """Raw strings with text format are now valid (Phase 2)."""
        signal, m = bridge_to_canonicalizer("raw text", {"format": "txt"})
        assert isinstance(signal, torch.Tensor)
        assert signal.dtype == torch.float32
        assert m["modality"] == "text"
        assert m["text_tokenizer"] == "char_level"

    def test_accepts_list_of_strings(self):
        """List of strings is valid text input."""
        signal, m = bridge_to_canonicalizer(["hello", "world"], {"format": "csv"})
        assert isinstance(signal, torch.Tensor)
        assert signal.ndim == 2  # (features, samples)
        assert signal.shape[1] == 2  # 2 texts
        assert m["modality"] == "text"

    def test_text_array_from_decoder(self):
        """Text-decoded numpy array passes through bridge."""
        from bifrost.ingest.decoders.text import TextDecoder

        decoder = TextDecoder()
        import pandas as pd
        df = pd.DataFrame({
            'text': ['hello world', 'test document'],
            'value': [1.0, 2.0],
        })
        import io
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        array, meta = decoder.decode(csv_bytes, 'csv')
        meta['format'] = 'csv'

        signal, m = bridge_to_canonicalizer(array, meta)
        assert isinstance(signal, torch.Tensor)
        assert signal.dtype == torch.float32
        assert m["modality"] == "text"


class TestBridgeToS0Integration:
    def test_audio_through_s0(self):
        """Full bridge → canonicalizer round-trip."""
        from bifrost.canonicalizer import SpectralCanonicalizer

        s0 = SpectralCanonicalizer(n_fft=256)
        data = np.random.randn(16000).astype(np.float32)
        meta = {"format": "wav", "channels": 1, "sample_rate": 16000}

        signal, enriched_meta = bridge_to_canonicalizer(data, meta)
        st = s0(signal, enriched_meta)
        st.validate()
        assert st.amplitude.shape[0] == 1  # 1 channel

    def test_stereo_through_s0(self):
        """Stereo scipy-layout → bridge → canonicalizer."""
        from bifrost.canonicalizer import SpectralCanonicalizer

        s0 = SpectralCanonicalizer(n_fft=256)
        data = np.random.randn(16000, 2).astype(np.float32)
        meta = {"format": "wav", "channels": 2, "sample_rate": 44100}

        signal, enriched_meta = bridge_to_canonicalizer(data, meta)
        st = s0(signal, enriched_meta)
        st.validate()
        assert st.amplitude.shape[0] == 2  # 2 channels

    def test_image_through_s0(self):
        """Image → bridge → canonicalizer."""
        from bifrost.canonicalizer import SpectralCanonicalizer

        s0 = SpectralCanonicalizer(n_fft=256)
        data = np.random.rand(32, 32, 3).astype(np.float32)
        meta = {"format": "png", "height": 32, "width": 32, "channels": 3}

        signal, enriched_meta = bridge_to_canonicalizer(data, meta)
        st = s0(signal, enriched_meta)
        st.validate()
        assert st.amplitude.shape[0] == 3  # 3 color channels
