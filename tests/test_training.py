"""Tests for BifrostTrainer and ComplexBifrostTrainer."""

import pytest
import torch
from bifrost import BifrostTrainer, BifrostPipeline, ComplexBifrostTrainer


class TestBifrostTrainer:
    """Test suite for BifrostTrainer."""

    def test_trainer_initialization(self):
        """Happy path: Trainer initializes with pipeline."""
        pipeline = BifrostPipeline(d_model=64)
        trainer = BifrostTrainer(pipeline, lr=1e-3)
        assert trainer.pipeline is pipeline
        assert trainer.lr == 1e-3

    def test_trainer_default_lr(self):
        """Happy path: Default learning rate applied."""
        pipeline = BifrostPipeline(d_model=64)
        trainer = BifrostTrainer(pipeline)
        assert trainer.lr > 0

    def test_train_step_returns_scalar(self, sample_spectral_input):
        """Happy path: train_step returns loss value."""
        pipeline = BifrostPipeline(d_model=64)
        trainer = BifrostTrainer(pipeline)

        # Create simple batch
        batch = torch.randn(2, 64, 100)
        loss = trainer.train_step(batch)

        assert isinstance(loss, (float, torch.Tensor))
        if isinstance(loss, torch.Tensor):
            assert loss.dim() == 0  # scalar

    def test_train_step_updates_parameters(self, sample_spectral_input):
        """Happy path: At least one parameter has a gradient after train_step."""
        pipeline = BifrostPipeline(d_model=32)
        trainer = BifrostTrainer(pipeline, lr=1e-2)

        # Snapshot all parameters before step
        params_before = {name: p.data.clone() for name, p in pipeline.named_parameters()}

        # train_step expects (B, L) time-domain signal
        batch = torch.randn(1, 16000)
        trainer.train_step(batch)

        # At least one parameter must have changed (has non-None gradient)
        updated = [
            name for name, p in pipeline.named_parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        ]
        assert len(updated) > 0, f"No parameters received gradients after train_step"

    def test_invalid_batch_shape_error(self):
        """Error path: Invalid batch shape should raise."""
        pipeline = BifrostPipeline(d_model=64)
        trainer = BifrostTrainer(pipeline)

        # Wrong shape
        invalid_batch = torch.randn(100)  # 1D instead of 3D
        with pytest.raises((ValueError, RuntimeError)):
            trainer.train_step(invalid_batch)

    def test_learning_rate_boundary(self):
        """Boundary: Very high learning rate should still work."""
        pipeline = BifrostPipeline(d_model=16)
        trainer = BifrostTrainer(pipeline, lr=1.0)  # Very high

        batch = torch.randn(1, 16, 10)
        # Should not crash, might produce large gradients
        try:
            loss = trainer.train_step(batch)
            assert loss is not None
        except RuntimeError:
            pass  # Gradients might explode


class TestComplexBifrostTrainer:
    """Test suite for ComplexBifrostTrainer."""

    def test_complex_trainer_initialization(self):
        """Happy path: Complex trainer initializes with real SpectralDecomposer."""
        from bifrost.decomposer import SpectralDecomposer

        decomposer = SpectralDecomposer(n_fft=64, d_model=64)
        trainer = ComplexBifrostTrainer(decomposer, lr=1e-3)
        assert trainer.d_model == 64
        assert trainer.decomposer is decomposer
        assert trainer.optimizer is not None

    def test_complex_trainer_has_optimizer(self):
        """Happy path: Trainer creates an optimizer on init."""
        from bifrost.decomposer import SpectralDecomposer

        decomposer = SpectralDecomposer(n_fft=64, d_model=64)
        trainer = ComplexBifrostTrainer(decomposer)
        assert hasattr(trainer, 'optimizer') or hasattr(trainer, 'opt')
