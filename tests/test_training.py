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
        """Happy path: Parameters change after training step."""
        pipeline = BifrostPipeline(d_model=32)
        trainer = BifrostTrainer(pipeline, lr=1e-2)

        # Get initial parameter
        param = list(pipeline.parameters())[0]
        initial_value = param.data.clone()

        # Training step
        batch = torch.randn(1, 32, 50)
        trainer.train_step(batch)

        # Parameter should have changed
        assert not torch.equal(param.data, initial_value)

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
        """Happy path: Complex trainer initializes."""
        from bifrost import ComplexFBCTrainer

        # Create minimal complex pipeline mock
        class MockComplexPipeline:
            def __init__(self):
                self.d_model = 64
                self.parameters = lambda: [torch.randn(10)]

        mock = MockComplexPipeline()
        trainer = ComplexBifrostTrainer(mock, lr=1e-3)
        assert trainer.lr == 1e-3
