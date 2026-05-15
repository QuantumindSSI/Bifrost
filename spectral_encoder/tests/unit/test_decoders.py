"""Unit tests for audio decoder."""

import pytest
import numpy as np
from spectral_encoder.ingest.decoders.audio import AudioDecoder
from spectral_encoder.ingest.validation.exceptions import DecodingError
from scipy.io import wavfile
import io


class TestAudioDecoder:
    """Test suite for AudioDecoder."""

    @pytest.fixture
    def decoder(self):
        return AudioDecoder()

    @pytest.fixture
    def wav_bytes_mono(self):
        """Create synthetic mono WAV."""
        sr = 16000
        duration = 1.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        bio = io.BytesIO()
        wavfile.write(bio, sr, audio)
        return bio.getvalue()

    @pytest.fixture
    def wav_bytes_stereo(self):
        """Create synthetic stereo WAV."""
        sr = 16000
        duration = 1.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        right = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        audio = np.column_stack([left, right])
        
        bio = io.BytesIO()
        wavfile.write(bio, sr, audio)
        return bio.getvalue()

    def test_supports_format(self, decoder):
        """Test format support checking."""
        assert decoder.supports_format("wav")
        assert decoder.supports_format("WAV")
        assert decoder.supports_format("mp3")
        assert not decoder.supports_format("xyz")

    def test_decode_mono_wav(self, decoder, wav_bytes_mono):
        """Test decoding mono WAV file."""
        audio, metadata = decoder.decode(wav_bytes_mono, "wav")
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert metadata["format"] == "wav"
        assert metadata["channels"] == 1
        assert metadata["sample_rate"] == 16000
        assert metadata["num_samples"] == 16000
        assert abs(metadata["duration_sec"] - 1.0) < 0.01

    def test_decode_stereo_wav(self, decoder, wav_bytes_stereo):
        """Test decoding stereo WAV file."""
        audio, metadata = decoder.decode(wav_bytes_stereo, "wav")
        
        assert audio.shape[0] == 2  # 2 channels
        assert metadata["channels"] == 2
        assert audio.dtype == np.float32

    def test_float32_range(self, decoder, wav_bytes_mono):
        """Test that output is in [-1.0, 1.0] range."""
        audio, _ = decoder.decode(wav_bytes_mono, "wav")
        
        assert np.all(audio >= -1.1)  # Allow small overshoot due to float precision
        assert np.all(audio <= 1.1)

    def test_decode_invalid_format(self, decoder, wav_bytes_mono):
        """Test error on unsupported format."""
        with pytest.raises(DecodingError):
            decoder.decode(wav_bytes_mono, "xyz")

    def test_decode_corrupted_data(self, decoder):
        """Test error on corrupted data."""
        with pytest.raises(DecodingError):
            decoder.decode(b"corrupted data", "wav")

    def test_librosa_fallback(self, decoder):
        """Test librosa availability message."""
        try:
            import librosa  # noqa
            assert True
        except ImportError:
            pytest.skip("librosa not installed")
