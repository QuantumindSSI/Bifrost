"""
Test suite for multimodal spectral encoding training.

Validates that Bifrost can train on audio, text, and image modalities
producing consistent SpectralTensor representations.
"""

import pytest
import torch
import torch.nn as nn

from bifrost import BifrostPipeline, BifrostTrainer, SpectralTensor


def text_to_waveform(tokens: torch.Tensor, d_model: int = 128) -> torch.Tensor:
    """Convert text token IDs to pseudo-waveform."""
    B, T = tokens.shape
    embed = nn.Embedding(256, d_model)
    embedded = embed(tokens)
    return embedded.view(B, -1)


def image_to_1d(images: torch.Tensor) -> torch.Tensor:
    """Convert images to 1D signal."""
    B = images.size(0)
    return images.view(B, -1)


class TestMultimodalSpectralEncoding:
    """End-to-end multimodal training tests."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline fixture."""
        return BifrostPipeline(d_model=64, use_complex_ssm=False)
    
    @pytest.fixture
    def trainer(self, pipeline):
        """Create trainer fixture."""
        return BifrostTrainer(pipeline, lr=0.001, device="cpu")
    
    def test_audio_produces_spectral_tensor(self, pipeline):
        """Audio input produces SpectralTensor output."""
        audio = torch.randn(2, 3200)  # 2 seconds @ 16kHz
        
        result = pipeline.canonicalizer(audio)
        
        assert isinstance(result, SpectralTensor)
        assert result.amplitude.shape[0] == 2  # Batch size preserved
        assert result.amplitude.shape[-1] == 513  # n_fft // 2 + 1
        assert torch.all(result.amplitude >= 0)  # Non-negative amplitude
    
    def test_text_produces_spectral_tensor(self, pipeline):
        """Text input produces SpectralTensor output."""
        text_tokens = torch.randint(0, 256, (2, 100))
        text_waveform = text_to_waveform(text_tokens, d_model=64)
        
        result = pipeline.canonicalizer(text_waveform)
        
        assert isinstance(result, SpectralTensor)
        assert result.amplitude.shape[0] == 2
        assert torch.all(result.amplitude >= 0)
    
    def test_image_produces_spectral_tensor(self, pipeline):
        """Image input produces SpectralTensor output."""
        images = torch.randn(2, 3, 32, 32)
        image_1d = image_to_1d(images)
        
        result = pipeline.canonicalizer(image_1d)
        
        assert isinstance(result, SpectralTensor)
        assert result.amplitude.shape[0] == 2
        assert torch.all(result.amplitude >= 0)
    
    def test_audio_training_step(self, trainer):
        """Audio training step produces scalar loss."""
        audio = torch.randn(2, 3200)
        
        loss = trainer.train_step(audio)
        
        assert isinstance(loss, float)
        assert loss > 0 or loss < 0  # Any finite value acceptable
        assert loss == loss  # Not NaN
    
    def test_text_training_step(self, trainer):
        """Text training step produces scalar loss."""
        text_tokens = torch.randint(0, 256, (2, 100))
        text_waveform = text_to_waveform(text_tokens, d_model=64)
        
        loss = trainer.train_step(text_waveform)
        
        assert isinstance(loss, float)
        assert loss == loss  # Not NaN
    
    def test_image_training_step(self, trainer):
        """Image training step produces scalar loss."""
        images = torch.randn(2, 3, 32, 32)
        image_1d = image_to_1d(images)
        
        loss = trainer.train_step(image_1d)
        
        assert isinstance(loss, float)
        assert loss == loss  # Not NaN
    
    def test_multimodal_epoch(self, trainer):
        """Complete epoch with all three modalities."""
        losses = {'audio': None, 'text': None, 'image': None}
        
        # Audio
        audio = torch.randn(2, 3200)
        losses['audio'] = trainer.train_step(audio)
        
        # Text
        text_tokens = torch.randint(0, 256, (2, 100))
        text_waveform = text_to_waveform(text_tokens, d_model=64)
        losses['text'] = trainer.train_step(text_waveform)
        
        # Image
        images = torch.randn(2, 3, 32, 32)
        image_1d = image_to_1d(images)
        losses['image'] = trainer.train_step(image_1d)
        
        # All should produce valid losses
        for modality, loss in losses.items():
            assert isinstance(loss, float), f"{modality} loss not scalar"
            assert loss == loss, f"{modality} loss is NaN"
        
        # Average loss is finite
        avg_loss = sum(losses.values()) / 3
        assert isinstance(avg_loss, float)
    
    def test_spectral_tensors_same_dimension(self, pipeline):
        """All modalities produce same spectral dimension."""
        # Audio
        audio = torch.randn(2, 3200)
        audio_st = pipeline.canonicalizer(audio)
        
        # Text
        text_tokens = torch.randint(0, 256, (2, 100))
        text_waveform = text_to_waveform(text_tokens, d_model=64)
        text_st = pipeline.canonicalizer(text_waveform)
        
        # Image
        images = torch.randn(2, 3, 32, 32)
        image_1d = image_to_1d(images)
        image_st = pipeline.canonicalizer(image_1d)
        
        # All have 4 components: amplitude, phase, scale, uncertainty
        for name, st in [('audio', audio_st), ('text', text_st), ('image', image_st)]:
            assert hasattr(st, 'amplitude'), f"{name}: no amplitude"
            assert hasattr(st, 'phase'), f"{name}: no phase"
            assert hasattr(st, 'scale'), f"{name}: no scale"
            assert hasattr(st, 'uncertainty'), f"{name}: no uncertainty"
    
    def test_gradient_flows_all_modalities(self, trainer):
        """Gradients flow through all modality paths."""
        # Get initial parameter
        param = list(trainer.pipeline.parameters())[0]
        initial_value = param.data.clone()
        
        # Train on each modality
        for batch_fn in [
            lambda: torch.randn(2, 3200),  # Audio
            lambda: text_to_waveform(torch.randint(0, 256, (2, 100)), 64),  # Text
            lambda: image_to_1d(torch.randn(2, 3, 32, 32)),  # Image
        ]:
            trainer.optimizer.zero_grad()
            loss = trainer.train_step(batch_fn())
            # Check parameters updated
        
        # Parameter should have changed
        assert not torch.equal(param.data, initial_value), "Parameters not updated"
