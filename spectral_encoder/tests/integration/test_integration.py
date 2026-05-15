"""Integration tests for the complete pipeline."""

import pytest
import numpy as np
import io
from scipy.io import wavfile
from spectral_encoder.ingest.pipeline import IngestPipeline, Modality
from spectral_encoder.ingest.validation.exceptions import ValidationError, DecodingError


class TestIngestPipeline:
    """Integration tests for IngestPipeline."""

    @pytest.fixture
    def pipeline(self):
        return IngestPipeline(strict_validation=False)

    @pytest.fixture
    def synthetic_wav(self):
        """Create synthetic WAV bytes."""
        sr = 16000
        samples = int(sr * 0.5)
        t = np.linspace(0, 0.5, samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        bio = io.BytesIO()
        wavfile.write(bio, sr, audio)
        return bio.getvalue()

    def test_pipeline_audio_decode_and_normalize(self, pipeline, synthetic_wav):
        """Test complete audio pipeline."""
        audio, metadata = pipeline.ingest(synthetic_wav, Modality.AUDIO, "wav")
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert metadata["sample_rate"] == 16000
        assert np.all(audio >= -1.1)
        assert np.all(audio <= 1.1)

    def test_pipeline_strict_mode(self):
        """Test strict validation mode."""
        pipeline = IngestPipeline(strict_validation=True)
        
        sr = 16000
        samples = 10
        t = np.linspace(0, 0.0001, samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        bio = io.BytesIO()
        wavfile.write(bio, sr, audio)
        wav_bytes = bio.getvalue()
        
        with pytest.raises(ValidationError):
            pipeline.ingest(wav_bytes, Modality.AUDIO, "wav")

    def test_pipeline_lenient_mode(self, pipeline, synthetic_wav):
        """Test lenient validation mode (should not raise)."""
        audio, metadata = pipeline.ingest(synthetic_wav, Modality.AUDIO, "wav")
        assert audio is not None

    def test_pipeline_corrupted_data(self, pipeline):
        """Test error handling for corrupted data."""
        with pytest.raises(DecodingError):
            pipeline.ingest(b"corrupted", Modality.AUDIO, "wav")
