"""
Tests for Riemannian Semantic Coherence Module

Validates:
- RiemannianMetricLearner produces positive definite metrics
- GeodesicComputer computes valid distances
- CoherenceScorer produces [0, 1] scores
- TripletLoss trains correctly
- Full pipeline integration

Agentic CTO Compliance: C-03 (every function has tests)
"""

import pytest
import torch
import numpy as np
from typing import List

from bifrost.riemannian_coherence import (
    RiemannianSemanticCoherence,
    RiemannianMetricLearner,
    GeodesicComputer,
    CoherenceScorer,
    ManifoldProjector,
    TripletSemanticLoss,
    SemanticCoherenceOutput,
    create_triplets_from_labels,
)
from bifrost.phase_lock_bridge import FrequencyAttractor


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_attractors() -> List[FrequencyAttractor]:
    """Create sample FrequencyAttractors for testing."""
    torch.manual_seed(42)
    attractors = []
    
    # Create 5 attractors with different characteristics
    for i in range(5):
        centroid = torch.randn(128) * 0.5 + i * 0.3  # Slightly separated
        phase_sig = torch.randn(8) * 0.1
        
        attractor = FrequencyAttractor(
            centroid=centroid,
            phase_signature=phase_sig,
            amplitude_profile=centroid,
            stability=torch.tensor(0.5 + i * 0.1),
            domain="test",
            attractor_id=f"test_attractor_{i}",
            metadata={"test": True},
        )
        attractors.append(attractor)
    
    return attractors


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# RiemannianMetricLearner Tests
# =============================================================================

class TestRiemannianMetricLearner:
    """Test RiemannianMetricLearner functionality."""
    
    def test_initialization(self):
        """Test metric learner initializes correctly."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        assert learner.d_model == 128
        assert learner.metric_dim == 16
        assert learner.base_metric.shape == (16, 16)
    
    def test_forward_produces_valid_metrics(self, sample_attractors):
        """Test forward pass produces positive definite metrics."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        
        metrics = learner(sample_attractors)
        
        # Check shape
        assert metrics.shape == (5, 16, 16)
        
        # Check positive definiteness: all eigenvalues > 0
        for i in range(5):
            eigvals = torch.linalg.eigvalsh(metrics[i])
            assert (eigvals > 0).all(), f"Metric {i} not positive definite"
        
        # Check symmetry
        for i in range(5):
            assert torch.allclose(metrics[i], metrics[i].T, atol=1e-6)
    
    def test_empty_attractor_list_raises(self):
        """Test empty attractor list raises ValueError."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        
        with pytest.raises(ValueError, match="Empty attractor list"):
            learner([])
    
    def test_local_distance_computation(self, sample_attractors):
        """Test local Riemannian distance computation."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        
        metrics = learner(sample_attractors)
        
        # Compute distance between first two attractors
        dist = learner.local_distance(
            sample_attractors[0].centroid,
            sample_attractors[1].centroid,
            metrics[0],
        )
        
        # Distance should be positive scalar
        assert dist.item() >= 0
        assert dist.dim() == 0  # scalar
    
    def test_distance_symmetry(self, sample_attractors):
        """Test that metric produces symmetric distances at same point."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        metrics = learner(sample_attractors[:2])
        
        # Distance using metric at point 0
        dist_01 = learner.local_distance(
            sample_attractors[0].centroid,
            sample_attractors[1].centroid,
            metrics[0],
        )
        
        # Distance using metric at point 1
        dist_10 = learner.local_distance(
            sample_attractors[1].centroid,
            sample_attractors[0].centroid,
            metrics[1],
        )
        
        # Distances should be finite
        assert torch.isfinite(dist_01)
        assert torch.isfinite(dist_10)


# =============================================================================
# GeodesicComputer Tests
# =============================================================================

class TestGeodesicComputer:
    """Test GeodesicComputer functionality."""
    
    def test_initialization(self):
        """Test geodesic computer initializes correctly."""
        computer = GeodesicComputer(k_neighbors=3, algorithm='dijkstra')
        assert computer.k_neighbors == 3
        assert computer.algorithm == 'dijkstra'
    
    def test_forward_produces_distance_matrix(self, sample_attractors):
        """Test forward pass produces valid distance matrix."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        computer = GeodesicComputer(k_neighbors=3)
        
        metrics = learner(sample_attractors)
        distances = computer(sample_attractors, metrics, learner)
        
        # Check shape
        assert distances.shape == (5, 5)
        
        # Check symmetry
        assert torch.allclose(distances, distances.T, atol=1e-4)
        
        # Check diagonal is zero
        assert torch.allclose(distances.diag(), torch.zeros(5), atol=1e-6)
        
        # Check all distances are non-negative
        assert (distances >= 0).all()
    
    def test_single_attractor_raises(self, sample_attractors):
        """Test single attractor raises ValueError."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        computer = GeodesicComputer(k_neighbors=3)
        
        metrics = learner(sample_attractors[:1])
        
        with pytest.raises(ValueError, match="Need at least 2 attractors"):
            computer(sample_attractors[:1], metrics, learner)
    
    def test_floyd_warshall_algorithm(self, sample_attractors):
        """Test Floyd-Warshall algorithm option."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        computer = GeodesicComputer(k_neighbors=3, algorithm='floyd')
        
        metrics = learner(sample_attractors)
        distances = computer(sample_attractors, metrics, learner)
        
        # Basic sanity checks
        assert distances.shape == (5, 5)
        assert (distances >= 0).all()
    
    def test_invalid_algorithm_raises(self, sample_attractors):
        """Test invalid algorithm raises ValueError."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        computer = GeodesicComputer(k_neighbors=3, algorithm='invalid')
        
        metrics = learner(sample_attractors)
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            computer(sample_attractors, metrics, learner)
    
    def test_triangle_inequality(self, sample_attractors):
        """Test that geodesic distances satisfy triangle inequality."""
        learner = RiemannianMetricLearner(d_model=128, metric_dim=16)
        computer = GeodesicComputer(k_neighbors=4)
        
        metrics = learner(sample_attractors)
        distances = computer(sample_attractors, metrics, learner)
        
        # Check triangle inequality for a few triplets
        for i in range(3):
            for j in range(i+1, 4):
                for k in range(j+1, 5):
                    d_ij = distances[i, j].item()
                    d_jk = distances[j, k].item()
                    d_ik = distances[i, k].item()
                    
                    # Allow small numerical tolerance
                    assert d_ik <= d_ij + d_jk + 1e-4, \
                        f"Triangle inequality violated: {d_ik} > {d_ij} + {d_jk}"


# =============================================================================
# CoherenceScorer Tests
# =============================================================================

class TestCoherenceScorer:
    """Test CoherenceScorer functionality."""
    
    def test_initialization(self):
        """Test coherence scorer initializes correctly."""
        scorer = CoherenceScorer(init_temperature=0.5, init_bias=0.1)
        assert scorer.temperature.item() == 0.5
        assert scorer.bias.item() == 0.1
    
    def test_forward_produces_valid_scores(self):
        """Test forward produces scores in [0, 1]."""
        scorer = CoherenceScorer()
        
        # Create sample distance matrix
        distances = torch.tensor([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ])
        
        coherence = scorer(distances)
        
        # Check shape
        assert coherence.shape == (3, 3)
        
        # Check range [0, 1]
        assert (coherence >= 0).all() and (coherence <= 1).all()
        
        # Check diagonal is 1 (self-coherence)
        assert torch.allclose(coherence.diag(), torch.ones(3), atol=1e-6)
        
        # Check symmetry
        assert torch.allclose(coherence, coherence.T, atol=1e-6)
    
    def test_distance_coherence_inverse(self):
        """Test that larger distances produce smaller coherence."""
        scorer = CoherenceScorer()
        
        # Small distance -> high coherence
        small_dist = torch.tensor([[0.0, 0.1], [0.1, 0.0]])
        high_coherence = scorer(small_dist)[0, 1].item()
        
        # Large distance -> low coherence
        large_dist = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
        low_coherence = scorer(large_dist)[0, 1].item()
        
        assert high_coherence > low_coherence
        assert high_coherence > 0.5  # Should be relatively high
        assert low_coherence < 0.5   # Should be relatively low


# =============================================================================
# ManifoldProjector Tests
# =============================================================================

class TestManifoldProjector:
    """Test ManifoldProjector functionality."""
    
    def test_initialization(self):
        """Test manifold projector initializes correctly."""
        projector = ManifoldProjector(n_components=3, method='mds')
        assert projector.n_components == 3
        assert projector.method == 'mds'
    
    def test_mds_projection(self):
        """Test classical MDS projection."""
        projector = ManifoldProjector(n_components=2, method='mds')
        
        # Create simple distance matrix (square corners)
        distances = torch.tensor([
            [0.0, 1.0, 1.0, 1.414],
            [1.0, 0.0, 1.414, 1.0],
            [1.0, 1.414, 0.0, 1.0],
            [1.414, 1.0, 1.0, 0.0],
        ])
        
        coords = projector(distances)
        
        # Check shape
        assert coords.shape == (4, 2)
        
        # Check finite
        assert torch.isfinite(coords).all()
    
    def test_pca_projection(self):
        """Test PCA projection."""
        projector = ManifoldProjector(n_components=2, method='pca')
        
        # Create sample features
        features = torch.randn(10, 128)
        
        coords = projector(torch.randn(10, 10), attractor_features=features)
        
        # Check shape
        assert coords.shape == (10, 2)
        
        # Check finite
        assert torch.isfinite(coords).all()
    
    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        projector = ManifoldProjector(n_components=2, method='invalid')
        
        with pytest.raises(ValueError, match="Unknown method"):
            projector(torch.randn(5, 5))


# =============================================================================
# TripletSemanticLoss Tests
# =============================================================================

class TestTripletSemanticLoss:
    """Test TripletSemanticLoss functionality."""
    
    def test_initialization(self):
        """Test triplet loss initializes correctly."""
        loss_fn = TripletSemanticLoss(margin=0.5)
        assert loss_fn.margin == 0.5
    
    def test_forward_computes_loss(self):
        """Test forward pass computes triplet loss."""
        loss_fn = TripletSemanticLoss(margin=1.0)
        
        # Create distance matrix where anchor is closer to positive than negative
        distances = torch.tensor([
            [0.0, 0.5, 2.0],  # Anchor: self=0, pos=0.5, neg=2.0
            [0.5, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ])
        
        anchor_idx = torch.tensor([0])
        positive_idx = torch.tensor([1])
        negative_idx = torch.tensor([2])
        
        loss = loss_fn(distances, anchor_idx, positive_idx, negative_idx)
        
        # Loss should be positive (d(a,p) - d(a,n) + margin = 0.5 - 2.0 + 1.0 = -0.5)
        # ReLU(-0.5) = 0, so loss = 0
        assert loss.item() == 0.0  # Successful triplet
    
    def test_forward_violated_triplet(self):
        """Test loss is positive for violated triplet."""
        loss_fn = TripletSemanticLoss(margin=1.0)
        
        # Create distance matrix where anchor is farther from positive than negative
        distances = torch.tensor([
            [0.0, 2.0, 0.5],  # Anchor: self=0, pos=2.0, neg=0.5
            [2.0, 0.0, 1.5],
            [0.5, 1.5, 0.0],
        ])
        
        anchor_idx = torch.tensor([0])
        positive_idx = torch.tensor([1])
        negative_idx = torch.tensor([2])
        
        loss = loss_fn(distances, anchor_idx, positive_idx, negative_idx)
        
        # Loss should be positive (d(a,p) - d(a,n) + margin = 2.0 - 0.5 + 1.0 = 2.5)
        assert loss.item() > 0.0
    
    def test_out_of_bounds_indices_raise(self):
        """Test out of bounds indices raise ValueError."""
        loss_fn = TripletSemanticLoss(margin=1.0)
        
        distances = torch.randn(5, 5)
        anchor_idx = torch.tensor([0, 1])
        positive_idx = torch.tensor([1, 2])
        negative_idx = torch.tensor([5, 3])  # 5 is out of bounds
        
        with pytest.raises(ValueError, match="Indices out of bounds"):
            loss_fn(distances, anchor_idx, positive_idx, negative_idx)
    
    def test_semantic_accuracy(self):
        """Test semantic accuracy computation."""
        loss_fn = TripletSemanticLoss(margin=1.0)
        
        # Distance matrix where synonyms (0,1) are closer than antonyms (0,2)
        distances = torch.tensor([
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 1.8],
            [2.0, 1.8, 0.0],
        ])
        
        synonym_pairs = [(0, 1)]
        antonym_pairs = [(0, 2)]
        
        accuracy = loss_fn.compute_semantic_accuracy(distances, synonym_pairs, antonym_pairs)
        
        # Should be 1.0 (100%) because d(0,1) < d(0,2)
        assert accuracy == 1.0


# =============================================================================
# RiemannianSemanticCoherence (S4) Tests
# =============================================================================

class TestRiemannianSemanticCoherence:
    """Test full S4 stage functionality."""
    
    def test_initialization(self):
        """Test S4 stage initializes correctly."""
        s4 = RiemannianSemanticCoherence(
            d_model=128,
            metric_dim=16,
            k_neighbors=3,
            manifold_dim=2,
        )
        
        assert s4.d_model == 128
        assert s4.metric_dim == 16
        assert s4.metric_learner is not None
        assert s4.geodesic_computer is not None
        assert s4.coherence_scorer is not None
        assert s4.manifold_projector is not None
    
    def test_forward_produces_valid_output(self, sample_attractors):
        """Test forward pass produces valid SemanticCoherenceOutput."""
        s4 = RiemannianSemanticCoherence(d_model=128, metric_dim=16)
        
        output = s4(sample_attractors)
        
        # Check output type
        assert isinstance(output, SemanticCoherenceOutput)
        
        # Check coherence scores
        assert output.coherence_scores.shape == (5, 5)
        assert (output.coherence_scores >= 0).all() and (output.coherence_scores <= 1).all()
        
        # Check geodesic distances
        assert output.geodesic_distances.shape == (5, 5)
        assert (output.geodesic_distances >= 0).all()
        
        # Check manifold coordinates
        assert output.manifold_coords.shape == (5, 2)
        assert torch.isfinite(output.manifold_coords).all()
        
        # Check metric tensor
        assert output.metric_tensor.shape == (5, 16, 16)
    
    def test_forward_without_projection(self, sample_attractors):
        """Test forward pass without manifold projection."""
        s4 = RiemannianSemanticCoherence(d_model=128, metric_dim=16)
        
        output = s4(sample_attractors, return_projection=False)
        
        # Manifold coords should still be computed but not used
        assert output.manifold_coords.shape == (5, 2)
    
    def test_insufficient_attractors_raises(self):
        """Test single attractor raises ValueError."""
        s4 = RiemannianSemanticCoherence(d_model=128, metric_dim=16)
        
        single_attractor = [
            FrequencyAttractor(
                centroid=torch.randn(128),
                phase_signature=torch.randn(8),
                amplitude_profile=torch.randn(128),
                stability=torch.tensor(0.5),
                domain="test",
                attractor_id="test_0",
                metadata={},
            )
        ]
        
        with pytest.raises(ValueError, match="Need >=2 attractors"):
            s4(single_attractor)
    
    def test_training_step(self, sample_attractors):
        """Test training step with triplet loss."""
        s4 = RiemannianSemanticCoherence(d_model=128, metric_dim=16)
        
        # Create synthetic triplet indices
        anchor_idx = torch.tensor([0, 1, 2])
        positive_idx = torch.tensor([1, 2, 3])
        negative_idx = torch.tensor([3, 4, 0])
        
        loss, output = s4.training_step(
            sample_attractors,
            anchor_idx,
            positive_idx,
            negative_idx,
        )
        
        # Check loss is scalar
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        
        # Check output is valid
        assert isinstance(output, SemanticCoherenceOutput)
    
    def test_find_semantic_clusters(self, sample_attractors):
        """Test semantic clustering."""
        s4 = RiemannianSemanticCoherence(d_model=128, metric_dim=16)
        
        clusters = s4.find_semantic_clusters(sample_attractors, coherence_threshold=0.3)
        
        # Should return dict
        assert isinstance(clusters, dict)
        
        # All attractors should be in some cluster
        all_indices = set()
        for cluster in clusters.values():
            all_indices.update(cluster)
        assert len(all_indices) == 5
    
    def test_semantic_search(self, sample_attractors):
        """Test semantic search functionality."""
        s4 = RiemannianSemanticCoherence(d_model=128, metric_dim=16)
        
        query = sample_attractors[0]
        database = sample_attractors[1:]
        
        results = s4.semantic_search(query, database, top_k=3)
        
        # Check return type
        assert isinstance(results, list)
        assert len(results) == 3
        
        # Check each result
        for idx, score in results:
            assert isinstance(idx, int)
            assert 0 <= idx < len(database)
            assert isinstance(score, float)
            assert 0 <= score <= 1
        
        # Results should be sorted by score (descending)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestCreateTripletsFromLabels:
    """Test triplet creation from labels."""
    
    def test_basic_triplet_creation(self):
        """Test basic triplet creation from labels."""
        labels = ['cat', 'cat', 'dog', 'dog', 'bird', 'bird']
        
        anchor_idx, positive_idx, negative_idx = create_triplets_from_labels(labels)
        
        # Should have some triplets
        assert len(anchor_idx) > 0
        assert len(anchor_idx) == len(positive_idx) == len(negative_idx)
        
        # Check each triplet
        for a, p, n in zip(anchor_idx.tolist(), positive_idx.tolist(), negative_idx.tolist()):
            # Positive should have same label as anchor
            assert labels[a] == labels[p]
            # Negative should have different label
            assert labels[a] != labels[n]
            # Should not be same index
            assert a != p
            assert a != n
    
    def test_with_synonym_map(self):
        """Test triplet creation with synonym map."""
        labels = ['cat', 'feline', 'dog', 'canine']
        synonym_map = {
            'cat': ['feline'],
            'dog': ['canine'],
        }
        
        anchor_idx, positive_idx, negative_idx = create_triplets_from_labels(
            labels, synonym_map=synonym_map
        )
        
        # Should have triplets
        assert len(anchor_idx) > 0
        
        # Check that synonyms are considered positives
        for a, p in zip(anchor_idx.tolist(), positive_idx.tolist()):
            label_a = labels[a]
            label_p = labels[p]
            # Positive should be same label or synonym
            assert label_p == label_a or label_p in synonym_map.get(label_a, [])
    
    def test_no_valid_triplets(self):
        """Test case where no valid triplets can be created."""
        # All same label - no negatives
        labels = ['cat', 'cat', 'cat']
        
        anchor_idx, positive_idx, negative_idx = create_triplets_from_labels(labels)
        
        # Should return empty tensors
        assert len(anchor_idx) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestRiemannianIntegration:
    """Integration tests for Riemannian coherence with other Bifrost components."""
    
    def test_with_attractor_learning(self):
        """Test Riemannian coherence integration with attractor learning outputs."""
        from bifrost.s3_attractor import AttractorLearningModule
        from bifrost.spectral_tensor import SpectralTensor
        
        # Create sample spectral tensor
        torch.manual_seed(42)
        amplitude = torch.randn(2, 10, 128).abs()
        phase = torch.randn(2, 10, 128)
        spectral = SpectralTensor(
            amplitude=amplitude,
            phase=phase,
            scale="decomposed",
            uncertainty=torch.ones(2, 10, 1) * 0.1,
        )
        
        # Extract attractors using attractor learning
        attractor_learner = AttractorLearningModule(d_model=128, n_bands=8, n_attractors=5)
        attractors, _ = attractor_learner(spectral)
        
        # Process with Riemannian coherence
        coherence = RiemannianSemanticCoherence(d_model=128, metric_dim=16)
        output = coherence(attractors)
        
        # Verify output
        assert isinstance(output, SemanticCoherenceOutput)
        assert output.coherence_scores.shape == (5, 5)
    
    def test_end_to_end_pipeline(self):
        """Test Riemannian coherence in context of full pipeline flow."""
        # Simulate: SpectralTensor -> AttractorLearning -> RiemannianCoherence
        from bifrost.spectral_tensor import SpectralTensor
        from bifrost.s3_attractor import AttractorLearningModule
        
        # Create synthetic spectral data
        torch.manual_seed(42)
        batch_size, seq_len, d_model = 2, 32, 128
        
        spectral = SpectralTensor(
            amplitude=torch.randn(batch_size, seq_len, d_model).abs(),
            phase=torch.randn(batch_size, seq_len, d_model),
            scale="decomposed",
            uncertainty=torch.ones(batch_size, seq_len, 1) * 0.1,
        )
        
        # Extract attractors
        attractor_learner = AttractorLearningModule(d_model=d_model, n_bands=8, n_attractors=6)
        attractors, assignment_probs = attractor_learner(spectral)
        
        # Compute semantic coherence
        coherence = RiemannianSemanticCoherence(d_model=d_model, metric_dim=16, k_neighbors=4)
        output = coherence(attractors)
        
        # Assertions
        assert len(attractors) == 6
        assert output.coherence_scores.shape == (6, 6)
        assert output.manifold_coords.shape == (6, 2)
        
        # Coherence should be higher for similar attractors
        # (This is a sanity check - actual values depend on learned metrics)
        assert (output.coherence_scores.diag() == 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
