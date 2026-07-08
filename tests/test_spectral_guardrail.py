"""Tests for spectral guardrail module."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from bifrost.agent.spectral_guardrail import (
    AttentionGraph,
    SpectralAnalyzer,
    SpectralGuardrail,
    AgentMonitor,
    SpectralFeatures,
)


class TestAttentionGraph:
    """Tests for AttentionGraph."""

    def test_build_graph_3d_attention(self):
        """Test graph construction from multi-head attention."""
        # [n_heads=2, seq_len=4, seq_len=4]
        attention = torch.rand(2, 4, 4)
        adj = AttentionGraph.build_graph(attention)

        assert adj.shape == (4, 4)
        # Should be symmetric
        assert torch.allclose(adj, adj.T, atol=1e-5)
        # Diagonal should be zero (no self-loops)
        assert torch.allclose(adj.diagonal(), torch.zeros(4))

    def test_build_graph_2d_attention(self):
        """Test graph construction from single-head attention."""
        attention = torch.rand(4, 4)
        adj = AttentionGraph.build_graph(attention)

        assert adj.shape == (4, 4)
        assert torch.allclose(adj, adj.T, atol=1e-5)

    def test_compute_laplacian(self):
        """Test Laplacian computation."""
        # Simple graph: 0-1-2-3
        adj = torch.zeros(4, 4)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0

        laplacian = AttentionGraph.compute_laplacian(adj)

        assert laplacian.shape == (4, 4)
        # Diagonal should be degrees
        assert laplacian[0, 0] == 1.0  # node 0 has degree 1
        assert laplacian[1, 1] == 2.0  # node 1 has degree 2
        # Off-diagonal should be -adjacency
        assert laplacian[0, 1] == -1.0

    def test_laplacian_eigenvalues(self):
        """Test eigenvalue computation."""
        adj = torch.zeros(4, 4)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0

        laplacian = AttentionGraph.compute_laplacian(adj)
        eigenvalues = AttentionGraph.laplacian_eigenvalues(laplacian, top_k=4)

        # First eigenvalue should be 0 (connected graph)
        assert eigenvalues[0] < 1e-5
        # All eigenvalues should be non-negative
        assert (eigenvalues >= -1e-5).all()


class TestSpectralAnalyzer:
    """Tests for SpectralAnalyzer."""

    def test_compute_hfer(self):
        """Test HFER computation."""
        analyzer = SpectralAnalyzer()
        hidden = torch.randn(8, 16)
        adj = torch.rand(8, 8)
        adj = (adj + adj.T) / 2
        adj.fill_diagonal_(0)
        laplacian = AttentionGraph.compute_laplacian(adj)

        hfer = analyzer.compute_hfer(hidden, laplacian)
        assert 0.0 <= hfer <= 1.0

    def test_compute_hfer_short_sequence(self):
        """Test HFER with very short sequence."""
        analyzer = SpectralAnalyzer()
        hidden = torch.randn(2, 16)
        adj = torch.rand(2, 2)
        adj = (adj + adj.T) / 2
        adj.fill_diagonal_(0)
        laplacian = AttentionGraph.compute_laplacian(adj)

        hfer = analyzer.compute_hfer(hidden, laplacian)
        # Should return default for short sequences
        assert 0.0 <= hfer <= 1.0

    def test_compute_spectral_entropy(self):
        """Test spectral entropy computation."""
        analyzer = SpectralAnalyzer()
        hidden = torch.randn(8, 16)
        adj = torch.rand(8, 8)
        adj = (adj + adj.T) / 2
        adj.fill_diagonal_(0)
        laplacian = AttentionGraph.compute_laplacian(adj)

        entropy = analyzer.compute_spectral_entropy(hidden, laplacian)
        assert 0.0 <= entropy <= 1.0

    def test_compute_smoothness(self):
        """Test smoothness computation."""
        analyzer = SpectralAnalyzer()
        hidden = torch.randn(8, 16)
        adj = torch.rand(8, 8)
        adj = (adj + adj.T) / 2
        adj.fill_diagonal_(0)
        laplacian = AttentionGraph.compute_laplacian(adj)

        smoothness = analyzer.compute_smoothness(hidden, laplacian)
        assert isinstance(smoothness, float)

    def test_compute_fiedler_value(self):
        """Test Fiedler value computation."""
        analyzer = SpectralAnalyzer()
        adj = torch.zeros(4, 4)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        laplacian = AttentionGraph.compute_laplacian(adj)

        fiedler = analyzer.compute_fiedler_value(laplacian)
        # Fiedler value for a path graph should be positive
        assert fiedler > 0

    def test_analyze_returns_all_features(self):
        """Test that analyze returns all spectral features."""
        analyzer = SpectralAnalyzer()
        hidden = torch.randn(8, 16)
        attention = torch.rand(2, 8, 8)

        features = analyzer.analyze(hidden, attention)

        assert isinstance(features, SpectralFeatures)
        assert 0.0 <= features.hfer <= 1.0
        assert 0.0 <= features.spectral_entropy <= 1.0
        assert isinstance(features.smoothness, float)
        assert isinstance(features.fiedler_value, float)


class TestSpectralGuardrail:
    """Tests for SpectralGuardrail with mock model."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns fake attentions and hidden states."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = 24
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock()

        # Mock forward pass output
        def mock_forward(*args, **kwargs):
            output = MagicMock()
            batch, seq_len = args[0].shape if args else kwargs["input_ids"].shape

            # Create fake attention weights [n_layers][batch, n_heads, seq, seq]
            n_heads = 4
            output.attentions = tuple(
                torch.rand(1, n_heads, seq_len, seq_len)
                for _ in range(24)
            )
            # Create fake hidden states [n_layers+1][batch, seq, dim]
            hidden_dim = 896
            output.hidden_states = tuple(
                torch.randn(1, seq_len, hidden_dim)
                for _ in range(25)
            )
            return output

        model.side_effect = mock_forward
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer that returns SimpleNamespace with input_ids."""
        def mock_call(text, return_tensors="pt", **kwargs):
            # Simple tokenization: split by spaces and hash to token ids
            tokens = text.split()
            token_ids = [hash(t) % 1000 + 100 for t in tokens]
            if not token_ids:
                token_ids = [100]
            return SimpleNamespace(input_ids=torch.tensor([token_ids]))

        tokenizer = MagicMock()
        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        return tokenizer

    def test_guardrail_initialization(self, mock_model, mock_tokenizer):
        """Test guardrail initialization."""
        guardrail = SpectralGuardrail(
            mock_model, mock_tokenizer, device="cpu"
        )
        assert guardrail.hfer_threshold == 0.25
        assert len(guardrail.monitor_layers) > 0

    def test_check_generation(self, mock_model, mock_tokenizer):
        """Test generation checking."""
        guardrail = SpectralGuardrail(
            mock_model, mock_tokenizer, device="cpu"
        )
        result = guardrail.check_generation("test prompt", "test output")

        assert isinstance(result, SpectralFeatures)
        assert 0.0 <= result.hfer <= 1.0
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.confidence, float)

    def test_check_context(self, mock_model, mock_tokenizer):
        """Test context checking."""
        guardrail = SpectralGuardrail(
            mock_model, mock_tokenizer, device="cpu"
        )
        result = guardrail.check_context("context here", "statement here")

        assert isinstance(result, SpectralFeatures)
        assert 0.0 <= result.hfer <= 1.0


class TestAgentMonitor:
    """Tests for AgentMonitor."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.num_hidden_layers = 24
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock()

        def mock_forward(*args, **kwargs):
            output = MagicMock()
            batch, seq_len = args[0].shape if args else kwargs["input_ids"].shape
            n_heads = 4
            output.attentions = tuple(
                torch.rand(1, n_heads, seq_len, seq_len) for _ in range(24)
            )
            output.hidden_states = tuple(
                torch.randn(1, seq_len, 896) for _ in range(25)
            )
            return output

        model.side_effect = mock_forward
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        def mock_call(text, return_tensors="pt", **kwargs):
            tokens = text.split()
            token_ids = [hash(t) % 1000 + 100 for t in tokens]
            if not token_ids:
                token_ids = [100]
            return SimpleNamespace(input_ids=torch.tensor([token_ids]))
        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call
        return tokenizer

    def test_check_step(self, mock_model, mock_tokenizer):
        """Test step checking."""
        monitor = AgentMonitor(mock_model, mock_tokenizer, device="cpu")
        result = monitor.check_step("context", "output")

        assert isinstance(result, SpectralFeatures)
        assert len(monitor.history) == 1

    def test_check_tool_call(self, mock_model, mock_tokenizer):
        """Test tool call checking."""
        monitor = AgentMonitor(mock_model, mock_tokenizer, device="cpu")
        result = monitor.check_tool_call(
            "A search tool", "query", "result"
        )

        assert isinstance(result, SpectralFeatures)

    def test_get_health_summary_empty(self, mock_model, mock_tokenizer):
        """Test health summary with no history."""
        monitor = AgentMonitor(mock_model, mock_tokenizer, device="cpu")
        summary = monitor.get_health_summary()

        assert summary["status"] == "no_data"
        assert summary["steps"] == 0

    def test_get_health_summary_with_data(self, mock_model, mock_tokenizer):
        """Test health summary with history."""
        monitor = AgentMonitor(mock_model, mock_tokenizer, device="cpu")
        monitor.check_step("context1", "output1")
        monitor.check_step("context2", "output2")

        summary = monitor.get_health_summary()
        assert summary["steps"] == 2
        assert "avg_hfer" in summary
        assert "hfer_trend" in summary
        assert len(summary["hfer_trend"]) == 2

    def test_reset(self, mock_model, mock_tokenizer):
        """Test reset."""
        monitor = AgentMonitor(mock_model, mock_tokenizer, device="cpu")
        monitor.check_step("context", "output")
        assert len(monitor.history) == 1

        monitor.reset()
        assert len(monitor.history) == 0
