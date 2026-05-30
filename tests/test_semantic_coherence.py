"""
Tests for Semantic Coherence Training Module

Per Agentic CTO-Persona policy C-03: Every function has tests for:
- Happy path
- Boundaries
- Error paths
"""

import torch
import pytest
from bifrost.semantic_coherence import (
    PhaseCoherenceExtractor,
    SupervisedSemanticCoherenceLoss,
    SemanticCoherenceHead,
    SemanticCoherenceTrainer,
    train_semantic_coherence,
)
from bifrost.pipeline import BifrostPipeline
from bifrost.spectral_tensor import SpectralTensor


class TestPhaseCoherenceExtractor:
    """Test PhaseCoherenceExtractor with various inputs."""
    
    def test_init(self):
        """Happy path: extractor initializes correctly."""
        extractor = PhaseCoherenceExtractor(d_model=128, coherence_dim=64)
        assert extractor.d_model == 128
        assert extractor.coherence_dim == 64
    
    def test_forward_shape(self):
        """Happy path: forward pass produces correct output shape."""
        extractor = PhaseCoherenceExtractor(d_model=128, coherence_dim=64)
        
        # Create mock spectral tensor
        batch_size = 4
        seq_len = 32
        d_model = 128
        
        spectral = SpectralTensor(
            amplitude=torch.randn(batch_size, seq_len, d_model),
            phase=torch.randn(batch_size, seq_len, d_model),
            scale=torch.ones(batch_size, seq_len, d_model),
            uncertainty=torch.zeros(batch_size, seq_len, d_model),
            metadata={},
        )
        
        coherence = extractor(spectral)
        
        assert coherence.shape == (batch_size, 64)
        assert coherence.dtype == torch.float32
    
    def test_normalization(self):
        """Happy path: output is L2 normalized."""
        extractor = PhaseCoherenceExtractor(d_model=64, coherence_dim=32)
        
        spectral = SpectralTensor(
            amplitude=torch.randn(2, 16, 64),
            phase=torch.randn(2, 16, 64),
            scale=torch.ones(2, 16, 64),
            uncertainty=torch.zeros(2, 16, 64),
            metadata={},
        )
        
        coherence = extractor(spectral)
        norms = torch.norm(coherence, dim=-1)
        
        # Should be close to 1.0 due to normalization
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_single_batch(self):
        """Boundary: single batch item."""
        extractor = PhaseCoherenceExtractor(d_model=64, coherence_dim=32)
        
        spectral = SpectralTensor(
            amplitude=torch.randn(1, 16, 64),
            phase=torch.randn(1, 16, 64),
            scale=torch.ones(1, 16, 64),
            uncertainty=torch.zeros(1, 16, 64),
            metadata={},
        )
        
        coherence = extractor(spectral)
        assert coherence.shape == (1, 32)
    
    def test_short_sequence(self):
        """Boundary: very short sequence."""
        extractor = PhaseCoherenceExtractor(d_model=64, coherence_dim=32)
        
        spectral = SpectralTensor(
            amplitude=torch.randn(2, 4, 64),
            phase=torch.randn(2, 4, 64),
            scale=torch.ones(2, 4, 64),
            uncertainty=torch.zeros(2, 4, 64),
            metadata={},
        )
        
        coherence = extractor(spectral)
        assert coherence.shape == (2, 32)
    
    def test_large_batch(self):
        """Boundary: large batch size."""
        extractor = PhaseCoherenceExtractor(d_model=64, coherence_dim=32)
        
        spectral = SpectralTensor(
            amplitude=torch.randn(100, 16, 64),
            phase=torch.randn(100, 16, 64),
            scale=torch.ones(100, 16, 64),
            uncertainty=torch.zeros(100, 16, 64),
            metadata={},
        )
        
        coherence = extractor(spectral)
        assert coherence.shape == (100, 32)


class TestSupervisedSemanticCoherenceLoss:
    """Test supervised contrastive loss."""
    
    def test_loss_reduction(self):
        """Happy path: loss decreases when same-class samples are pulled together."""
        loss_fn = SupervisedSemanticCoherenceLoss(temperature=0.1)
        
        # Create features where same-class samples are far apart initially
        features = torch.randn(4, 64)
        labels = torch.tensor([0, 0, 1, 1])
        
        # Compute loss
        loss1 = loss_fn(features, labels)
        
        # Manually pull same-class features together
        features_aligned = features.clone()
        features_aligned[0] = features[1]  # Make class 0 identical
        features_aligned[2] = features[3]  # Make class 1 identical
        
        loss2 = loss_fn(features_aligned, labels)
        
        # Loss should decrease when same-class features are closer
        assert loss2 < loss1
    
    def test_same_labels_zero_loss(self):
        """Boundary: all same labels should result in finite loss."""
        loss_fn = SupervisedSemanticCoherenceLoss()
        
        features = torch.randn(4, 64)
        labels = torch.tensor([0, 0, 0, 0])  # All same
        
        loss = loss_fn(features, labels)
        
        assert torch.isfinite(loss)
        assert loss.numel() == 1
    
    def test_all_different_labels(self):
        """Boundary: all different labels should still produce loss."""
        loss_fn = SupervisedSemanticCoherenceLoss()
        
        features = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 2, 3])  # All different
        
        loss = loss_fn(features, labels)
        
        assert torch.isfinite(loss)
        assert loss >= 0
    
    def test_gradient_flow(self):
        """Happy path: gradients flow back to features."""
        loss_fn = SupervisedSemanticCoherenceLoss()
        
        features = torch.randn(4, 64, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        
        loss = loss_fn(features, labels)
        loss.backward()
        
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()


class TestSemanticCoherenceHead:
    """Test semantic classification head."""
    
    def test_forward_shape(self):
        """Happy path: correct output shape."""
        head = SemanticCoherenceHead(coherence_dim=64, num_classes=10)
        
        features = torch.randn(4, 64)
        logits = head(features)
        
        assert logits.shape == (4, 10)
    
    def test_classification_accuracy(self):
        """Happy path: can fit simple classification task."""
        head = SemanticCoherenceHead(coherence_dim=32, num_classes=3)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        
        # Simple dataset: 3 classes with distinct feature patterns
        features = torch.randn(30, 32)
        # Make class 0 have positive first dim, class 1 negative, class 2 near zero
        features[:10, 0] = torch.abs(features[:10, 0]) + 1.0  # Class 0
        features[10:20, 0] = -torch.abs(features[10:20, 0]) - 1.0  # Class 1
        features[20:, 0] = features[20:, 0] * 0.1  # Class 2
        
        labels = torch.tensor([0]*10 + [1]*10 + [2]*10)
        
        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            logits = head(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            logits = head(features)
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
        
        # Should achieve decent accuracy
        assert accuracy > 0.7
    
    def test_dropout_active_during_training(self):
        """Happy path: dropout is active during training."""
        head = SemanticCoherenceHead(coherence_dim=32, num_classes=3)
        head.train()
        
        features = torch.randn(2, 32)
        
        # Multiple forward passes should give different results due to dropout
        logits1 = head(features)
        logits2 = head(features)
        
        # Should be different (with high probability due to dropout)
        assert not torch.allclose(logits1, logits2, atol=1e-4)
    
    def test_dropout_inactive_during_eval(self):
        """Happy path: dropout is inactive during eval."""
        head = SemanticCoherenceHead(coherence_dim=32, num_classes=3)
        head.eval()
        
        features = torch.randn(2, 32)
        
        # Multiple forward passes should give same results
        with torch.no_grad():
            logits1 = head(features)
            logits2 = head(features)
        
        # Should be identical
        assert torch.allclose(logits1, logits2)


class TestSemanticCoherenceTrainer:
    """Test end-to-end trainer."""
    
    def test_init(self):
        """Happy path: trainer initializes correctly."""
        pipeline = BifrostPipeline(n_fft_s0=256, d_model=64)
        
        trainer = SemanticCoherenceTrainer(
            pipeline=pipeline,
            num_classes=5,
            device="cpu",
        )
        
        assert trainer.num_classes == 5
        assert trainer.device == "cpu"
    
    def test_train_step(self):
        """Happy path: train step runs without error."""
        pipeline = BifrostPipeline(n_fft_s0=256, d_model=64)
        
        trainer = SemanticCoherenceTrainer(
            pipeline=pipeline,
            num_classes=3,
            device="cpu",
            lr=1e-3,
        )
        
        # Create synthetic batch
        signals = torch.randn(8, 512)
        labels = torch.randint(0, 3, (8,))
        
        # Train step
        metrics = trainer.train_step(signals, labels)
        
        # Check all expected metrics are present
        assert "total_loss" in metrics
        assert "semantic_loss" in metrics
        assert "contrastive_loss" in metrics
        assert "classification_loss" in metrics
        assert "classification_accuracy" in metrics
        
        # All values should be finite
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not (isinstance(value, float) and (value != value))  # Not NaN
    
    def test_train_step_gradient_update(self):
        """Happy path: parameters are updated after train step."""
        pipeline = BifrostPipeline(n_fft_s0=256, d_model=64)
        
        trainer = SemanticCoherenceTrainer(
            pipeline=pipeline,
            num_classes=3,
            device="cpu",
        )
        
        # Get initial parameter values
        initial_params = {
            name: param.clone() 
            for name, param in pipeline.named_parameters()
        }
        
        # Train step
        signals = torch.randn(4, 512)
        labels = torch.randint(0, 3, (4,))
        trainer.train_step(signals, labels)
        
        # Check parameters changed
        for name, param in pipeline.named_parameters():
            assert not torch.equal(param, initial_params[name])
    
    def test_evaluate_semantic_coherence(self):
        """Happy path: evaluation runs without error."""
        pipeline = BifrostPipeline(n_fft_s0=256, d_model=64)
        
        trainer = SemanticCoherenceTrainer(
            pipeline=pipeline,
            num_classes=3,
            device="cpu",
        )
        
        # Create test data
        test_signals = torch.randn(20, 512)
        test_labels = torch.randint(0, 3, (20,))
        
        # Evaluate
        metrics = trainer.evaluate_semantic_coherence(test_signals, test_labels)
        
        assert isinstance(metrics.contrastive_loss, float)
        assert isinstance(metrics.phase_similarity_accuracy, float)
        assert isinstance(metrics.semantic_retrieval_recall, float)
        assert isinstance(metrics.coherence_semantic_correlation, float)
        
        # Recall should be between 0 and 1
        assert 0 <= metrics.semantic_retrieval_recall <= 1


class TestTrainSemanticCoherence:
    """Test high-level training function."""
    
    def test_basic_training(self):
        """Happy path: complete training run."""
        pipeline = BifrostPipeline(n_fft_s0=256, d_model=64)
        
        # Create synthetic dataset
        n_samples = 30
        signals = [torch.randn(512) for _ in range(n_samples)]
        labels = [i % 3 for i in range(n_samples)]  # 3 classes
        
        # Train
        trainer = train_semantic_coherence(
            pipeline=pipeline,
            train_signals=signals,
            train_labels=labels,
            num_classes=3,
            epochs=2,
            batch_size=8,
            device="cpu",
        )
        
        assert isinstance(trainer, SemanticCoherenceTrainer)
        assert len(trainer.metrics_history) > 0
    
    def test_training_improves_coherence(self):
        """Happy path: training should improve semantic coherence metrics."""
        pipeline = BifrostPipeline(n_fft_s0=256, d_model=64)
        
        # Create dataset with clear class structure
        n_per_class = 20
        signals = []
        labels = []
        
        for class_id in range(3):
            for _ in range(n_per_class):
                # Add class-specific pattern
                base = torch.randn(512)
                base[class_id*100:(class_id+1)*100] += 2.0  # Class-specific activation
                signals.append(base)
                labels.append(class_id)
        
        # Evaluate before training
        trainer = SemanticCoherenceTrainer(
            pipeline=pipeline,
            num_classes=3,
            device="cpu",
        )
        
        test_signals = torch.stack([s for s in signals[:30]])
        test_labels = torch.tensor(labels[:30])
        
        metrics_before = trainer.evaluate_semantic_coherence(test_signals, test_labels)
        
        # Train
        train_semantic_coherence(
            pipeline=pipeline,
            train_signals=signals,
            train_labels=labels,
            num_classes=3,
            epochs=5,
            batch_size=10,
            device="cpu",
        )
        
        # Evaluate after training
        metrics_after = trainer.evaluate_semantic_coherence(test_signals, test_labels)
        
        # Coherence-semantic correlation should improve
        assert metrics_after.coherence_semantic_correlation > metrics_before.coherence_semantic_correlation or \
               metrics_after.phase_similarity_accuracy > metrics_before.phase_similarity_accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
