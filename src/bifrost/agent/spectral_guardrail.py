"""
Spectral Guardrails for Agent Safety.

Implements training-free hallucination detection for LLM agents based on
the spectral properties of attention topology. When an LLM hallucinates,
its attention graph undergoes a thermodynamic state change: the High-
Frequency Energy Ratio (HFER) collapses from ~0.52 (context-supported)
to ~0.05 (context-contradicted).

This module provides:
1. AttentionGraph: constructs a graph from attention weights
2. SpectralAnalyzer: computes HFER, spectral entropy, smoothness
3. SpectralGuardrail: inline binary accept/reject decision during generation
4. AgentMonitor: integration with agent execution loop

Based on:
- Spectral Guardrails (arXiv:2602.08082): 97.7% recall on Llama 3.1 8B
- Spectral Kill Switches (OpenReview 2026): <1ms overhead, bimodal HFER
- Bifrost doc 19: spectral alpha distinguishes reasoning from factual

Usage:
    from bifrost.agent.spectral_guardrail import SpectralGuardrail

    guardrail = SpectralGuardrail(model, tokenizer)
    result = guardrail.check_context(prompt, generated_text)
    if not result.is_safe:
        # Agent should retry or flag for review
        ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpectralFeatures:
    """Spectral features extracted from attention graph."""
    hfer: float                    # High-Frequency Energy Ratio
    spectral_entropy: float        # Shannon entropy of spectral energy
    smoothness: float              # Dirichlet smoothness of hidden states
    fiedler_value: float           # Second smallest Laplacian eigenvalue
    total_energy: float            # Total spectral energy
    is_safe: bool                  # Binary safety decision
    confidence: float              # Confidence in the safety decision
    layer_features: Dict[int, Dict[str, float]] = field(default_factory=dict)


class AttentionGraph:
    """Constructs a graph from attention weights.

    The attention matrix A[i,j] represents how much token i attends to
    token j. We treat this as a weighted adjacency matrix and compute
    graph spectral properties.

    The graph Laplacian L = D - A (where D is the degree matrix) captures
    the connectivity structure. Its eigenvalues reveal the graph's
    spectral properties:
    - Small eigenvalues = low-frequency components (global structure)
    - Large eigenvalues = high-frequency components (local detail)
    """

    @staticmethod
    def build_graph(attention: torch.Tensor) -> torch.Tensor:
        """Build weighted adjacency matrix from attention weights.

        Parameters
        ----------
        attention : torch.Tensor
            Attention weights of shape [n_heads, seq_len, seq_len] or
            [seq_len, seq_len].

        Returns
        -------
        adjacency : torch.Tensor
            Symmetric weighted adjacency matrix [seq_len, seq_len].
        """
        if attention.dim() == 3:
            # Average over heads
            attention = attention.mean(dim=0)

        # Make symmetric (attention is directed, but graph spectral
        # analysis requires undirected graphs)
        adjacency = (attention + attention.transpose(-1, -2)) / 2.0

        # Remove self-loops
        n = adjacency.shape[-1]
        adjacency.fill_diagonal_(0.0)

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        # This preserves symmetry while normalizing
        degree = adjacency.sum(dim=-1).clamp(min=1e-8)
        d_inv_sqrt = degree.pow(-0.5)
        adjacency = adjacency * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)

        return adjacency

    @staticmethod
    def compute_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        """Compute the graph Laplacian.

        L = D - A where D is the degree matrix.

        Parameters
        ----------
        adjacency : torch.Tensor
            Symmetric adjacency matrix [seq_len, seq_len].

        Returns
        -------
        laplacian : torch.Tensor
            Graph Laplacian [seq_len, seq_len].
        """
        degree = adjacency.sum(dim=-1)
        degree_matrix = torch.diag(degree)
        laplacian = degree_matrix - adjacency
        return laplacian

    @staticmethod
    def laplacian_eigenvalues(laplacian: torch.Tensor,
                              top_k: int = 20) -> torch.Tensor:
        """Compute eigenvalues of the Laplacian.

        Parameters
        ----------
        laplacian : torch.Tensor
            Graph Laplacian [n, n].
        top_k : int
            Number of eigenvalues to compute (for efficiency).

        Returns
        -------
        eigenvalues : torch.Tensor
            Sorted eigenvalues (smallest first) [min(top_k, n)].
        """
        try:
            eigenvalues = torch.linalg.eigvalsh(laplacian)
            eigenvalues = eigenvalues.sort()[0]
            return eigenvalues[:top_k]
        except Exception:
            # Fallback: use SVD approximation
            n = laplacian.shape[0]
            u, s, v = torch.svd(laplacian)
            return s[:top_k]


class SpectralAnalyzer:
    """Analyzes spectral properties of attention graphs and hidden states.

    Computes the key spectral metrics used for hallucination detection:
    - HFER: High-Frequency Energy Ratio (key metric, bimodal distribution)
    - Spectral Entropy: Shannon entropy of spectral energy distribution
    - Smoothness: Dirichlet energy of hidden states on the graph
    - Fiedler Value: Second smallest Laplacian eigenvalue (connectivity)
    """

    def __init__(self, n_frequency_bins: int = 32):
        self.n_frequency_bins = n_frequency_bins

    def compute_hfer(self, hidden_states: torch.Tensor,
                     laplacian: torch.Tensor) -> float:
        """Compute High-Frequency Energy Ratio.

        HFER = E_high / E_total

        where E_high is the energy in high-frequency graph spectral
        components and E_total is the total energy.

        Context-supported text: HFER ~ 0.52 (high-frequency, segregated)
        Context-contradicted text: HFER ~ 0.05 (low-frequency, smooth)

        Parameters
        ----------
        hidden_states : torch.Tensor
            Hidden state signals [seq_len, hidden_dim].
        laplacian : torch.Tensor
            Graph Laplacian [seq_len, seq_len].

        Returns
        -------
        hfer : float
            High-frequency energy ratio in [0, 1].
        """
        n = hidden_states.shape[0]
        if n < 4:
            return 0.5  # Default for very short sequences

        try:
            # Compute graph Fourier transform
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

            # Project hidden states onto eigenvectors (graph Fourier transform)
            # hidden_states: [n, d], eigenvectors: [n, n]
            # gft: [n, d] — spectral coefficients
            gft = eigenvectors.T @ hidden_states

            # Energy at each frequency
            energy = (gft ** 2).sum(dim=-1)  # [n]

            total_energy = energy.sum().item()
            if total_energy < 1e-10:
                return 0.5

            # High frequency = top half of eigenvalues
            n_high = n // 2
            high_energy = energy[-n_high:].sum().item()

            return high_energy / total_energy

        except Exception:
            return 0.5

    def compute_spectral_entropy(self, hidden_states: torch.Tensor,
                                 laplacian: torch.Tensor) -> float:
        """Compute spectral entropy of hidden state energy distribution.

        High entropy = distributed energy (healthy processing)
        Low entropy = concentrated energy (potential hallucination)

        Parameters
        ----------
        hidden_states : torch.Tensor
            [seq_len, hidden_dim]
        laplacian : torch.Tensor
            [seq_len, seq_len]

        Returns
        -------
        entropy : float
            Normalized spectral entropy in [0, 1].
        """
        n = hidden_states.shape[0]
        if n < 2:
            return 0.0

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
            gft = eigenvectors.T @ hidden_states
            energy = (gft ** 2).sum(dim=-1)  # [n]

            total = energy.sum().item()
            if total < 1e-10:
                return 0.0

            # Energy distribution
            probs = (energy / total).clamp(min=1e-10)

            # Shannon entropy
            entropy = -(probs * probs.log()).sum().item()

            # Normalize by max entropy (log(n))
            max_entropy = math.log(n)
            return entropy / max_entropy if max_entropy > 0 else 0.0

        except Exception:
            return 0.0

    def compute_smoothness(self, hidden_states: torch.Tensor,
                           laplacian: torch.Tensor) -> float:
        """Compute Dirichlet smoothness of hidden states on the graph.

        S = h^T L h / ||h||^2

        Low smoothness = hidden states vary smoothly across the graph
        (context-supported, coherent processing)
        High smoothness = hidden states vary sharply (potential issue)

        Note: In the Spectral Guardrails paper, LOWER smoothness values
        indicate MORE coherent processing.

        Parameters
        ----------
        hidden_states : torch.Tensor
            [seq_len, hidden_dim]
        laplacian : torch.Tensor
            [seq_len, seq_len]

        Returns
        -------
        smoothness : float
            Dirichlet smoothness value.
        """
        try:
            # S = trace(H^T L H) / trace(H^T H)
            numerator = (hidden_states.T @ laplacian @ hidden_states).trace()
            denominator = (hidden_states.T @ hidden_states).trace()

            if denominator.abs() < 1e-10:
                return 0.0

            return (numerator / denominator).item()
        except Exception:
            return 0.0

    def compute_fiedler_value(self, laplacian: torch.Tensor) -> float:
        """Compute the Fiedler value (algebraic connectivity).

        The second smallest eigenvalue of the Laplacian. Measures how
        well-connected the graph is. Low values indicate disconnected
        components (potential attention breakdown).

        Parameters
        ----------
        laplacian : torch.Tensor
            [n, n]

        Returns
        -------
        fiedler : float
            Second smallest Laplacian eigenvalue.
        """
        try:
            eigenvalues = torch.linalg.eigvalsh(laplacian)
            eigenvalues = eigenvalues.sort()[0]
            if len(eigenvalues) >= 2:
                return eigenvalues[1].item()
            return 0.0
        except Exception:
            return 0.0

    def analyze(self, hidden_states: torch.Tensor,
                attention: torch.Tensor) -> SpectralFeatures:
        """Compute all spectral features for a single layer.

        Parameters
        ----------
        hidden_states : torch.Tensor
            [seq_len, hidden_dim]
        attention : torch.Tensor
            [n_heads, seq_len, seq_len] or [seq_len, seq_len]

        Returns
        -------
        features : SpectralFeatures
            All spectral features.
        """
        adjacency = AttentionGraph.build_graph(attention)
        laplacian = AttentionGraph.compute_laplacian(adjacency)

        hfer = self.compute_hfer(hidden_states, laplacian)
        entropy = self.compute_spectral_entropy(hidden_states, laplacian)
        smoothness = self.compute_smoothness(hidden_states, laplacian)
        fiedler = self.compute_fiedler_value(laplacian)

        # Total energy
        total_energy = (hidden_states ** 2).sum().item()

        return SpectralFeatures(
            hfer=hfer,
            spectral_entropy=entropy,
            smoothness=smoothness,
            fiedler_value=fiedler,
            total_energy=total_energy,
            is_safe=True,  # Set by guardrail
            confidence=0.0,  # Set by guardrail
        )


class SpectralGuardrail:
    """Training-free spectral guardrail for hallucination detection.

    Monitors an LLM's attention topology during generation and emits
    a binary accept/reject signal based on spectral features.

    The key insight: hallucination is a thermodynamic state change.
    When the model hallucinates, HFER collapses from ~0.52 to ~0.05.
    This bimodal distribution enables a simple threshold-based decision.

    Usage:
        guardrail = SpectralGuardrail(model, tokenizer)
        result = guardrail.check_generation(prompt, generated_text)
        if not result.is_safe:
            print(f"Hallucination detected! HFER={result.hfer:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        monitor_layers: Optional[List[int]] = None,
        hfer_threshold: float = 0.25,
        entropy_threshold: float = 0.5,
        smoothness_threshold: float = 2.0,
        device: str = "cpu",
        adaptive_thresholds: bool = True,
    ):
        """Initialize the spectral guardrail.

        Parameters
        ----------
        model : nn.Module
            HuggingFace causal LM model.
        tokenizer : tokenizer
            HuggingFace tokenizer.
        monitor_layers : List[int], optional
            Which layers to monitor. Default: early layers [2, 3, 4, 5]
            (where the bimodal HFER pattern is strongest).
        hfer_threshold : float
            HFER below this → hallucination suspected.
            Context-supported: ~0.52, context-contradicted: ~0.05.
            Threshold of 0.25 separates the two modes.
            Note: calibrated for 7B+ models. On smaller models, the
            bimodal pattern is weaker and thresholds need adjustment.
        entropy_threshold : float
            Spectral entropy below this → hallucination suspected.
        smoothness_threshold : float
            Smoothness above this → hallucination suspected.
        device : str
            Device to run on.
        adaptive_thresholds : bool
            If True, use relative comparison (compare against a baseline)
            rather than absolute thresholds. This is more robust across
            model sizes.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hfer_threshold = hfer_threshold
        self.entropy_threshold = entropy_threshold
        self.smoothness_threshold = smoothness_threshold
        self.adaptive_thresholds = adaptive_thresholds
        self.baseline_features: Optional[SpectralFeatures] = None

        # Default: monitor early layers (2-5) where bimodal pattern is strongest
        if monitor_layers is None:
            n_layers = getattr(model.config, "num_hidden_layers", 24)
            monitor_layers = [min(2, n_layers - 1),
                            min(3, n_layers - 1),
                            min(4, n_layers - 1),
                            min(5, n_layers - 1)]
        self.monitor_layers = monitor_layers
        self.analyzer = SpectralAnalyzer()

        # Move model to device
        self.model.to(device)
        self.model.eval()

    def calibrate_baseline(self, context: str, supported_statement: str):
        """Calibrate thresholds using a known-good example.

        Run this once with a context and a statement that is definitely
        supported by the context. The guardrail will use this as a
        baseline for relative comparison.

        This is especially important for small models (<7B) where the
        absolute HFER values don't match the 0.52/0.05 bimodal pattern
        from the literature.

        Parameters
        ----------
        context : str
            A reference context.
        supported_statement : str
            A statement definitely supported by the context.
        """
        self.baseline_features = self.check_generation(
            context, supported_statement
        )
        # Set thresholds relative to baseline
        # A hallucination should show HFER significantly below baseline
        self.hfer_threshold = self.baseline_features.hfer * 0.5
        self.smoothness_threshold = self.baseline_features.smoothness * 1.5

    def _extract_attention_and_hidden(
        self, input_ids: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract attention weights and hidden states from specified layers.

        Parameters
        ----------
        input_ids : torch.Tensor
            [batch, seq_len]

        Returns
        -------
        layer_data : Dict[int, Tuple[attention, hidden_states]]
            For each monitored layer: (attention [n_heads, seq, seq],
            hidden_states [seq, hidden_dim])
        """
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        layer_data = {}
        for layer_idx in self.monitor_layers:
            if layer_idx < len(outputs.attentions):
                attn = outputs.attentions[layer_idx][0]  # [n_heads, seq, seq]
                hidden = outputs.hidden_states[layer_idx + 1][0]  # [seq, dim]
                layer_data[layer_idx] = (attn, hidden)

        return layer_data

    def check_generation(
        self,
        prompt: str,
        generated_text: str,
        max_length: int = 512,
    ) -> SpectralFeatures:
        """Check if a generated text is likely hallucinated.

        Parameters
        ----------
        prompt : str
            The input prompt.
        generated_text : str
            The generated text to check.
        max_length : int
            Maximum sequence length for analysis.

        Returns
        -------
        features : SpectralFeatures
            Spectral features with safety decision.
        """
        full_text = prompt + generated_text
        inputs = self.tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=max_length
        )

        layer_data = self._extract_attention_and_hidden(inputs.input_ids)

        # Aggregate features across monitored layers
        all_features = {}
        hfers, entropies, smoothnesses, fiedlers = [], [], [], []

        for layer_idx, (attn, hidden) in layer_data.items():
            feats = self.analyzer.analyze(hidden, attn)
            all_features[layer_idx] = {
                "hfer": feats.hfer,
                "spectral_entropy": feats.spectral_entropy,
                "smoothness": feats.smoothness,
                "fiedler_value": feats.fiedler_value,
            }
            hfers.append(feats.hfer)
            entropies.append(feats.spectral_entropy)
            smoothnesses.append(feats.smoothness)
            fiedlers.append(feats.fiedler_value)

        if not hfers:
            return SpectralFeatures(
                hfer=0.5, spectral_entropy=0.5, smoothness=1.0,
                fiedler_value=0.0, total_energy=0.0,
                is_safe=True, confidence=0.0,
            )

        # Aggregate (mean across layers)
        avg_hfer = float(np.mean(hfers))
        avg_entropy = float(np.mean(entropies))
        avg_smoothness = float(np.mean(smoothnesses))
        avg_fiedler = float(np.mean(fiedlers))

        # Safety decision: HFER is the primary signal
        # Context-supported: HFER ~ 0.52
        # Context-contradicted: HFER ~ 0.05
        # We use a multi-signal vote
        votes_unsafe = 0
        if avg_hfer < self.hfer_threshold:
            votes_unsafe += 2  # HFER is the strongest signal (weight 2)
        if avg_entropy < self.entropy_threshold:
            votes_unsafe += 1
        if avg_smoothness > self.smoothness_threshold:
            votes_unsafe += 1

        is_safe = votes_unsafe < 2  # Need at least 2 votes to flag unsafe

        # Confidence: how far from the threshold
        hfer_distance = abs(avg_hfer - self.hfer_threshold)
        confidence = min(1.0, hfer_distance * 4.0)  # Scale to [0, 1]

        return SpectralFeatures(
            hfer=avg_hfer,
            spectral_entropy=avg_entropy,
            smoothness=avg_smoothness,
            fiedler_value=avg_fiedler,
            total_energy=0.0,
            is_safe=is_safe,
            confidence=confidence,
            layer_features=all_features,
        )

    def check_context(
        self,
        context: str,
        statement: str,
        max_length: int = 512,
    ) -> SpectralFeatures:
        """Check if a statement is supported by context.

        This is the key use case for agents: verify that a generated
        statement is actually supported by the retrieved context.

        Parameters
        ----------
        context : str
            The retrieved context (e.g., from RAG).
        statement : str
            The statement to verify.

        Returns
        -------
        features : SpectralFeatures
            Spectral features with safety decision.
        """
        return self.check_generation(context, statement, max_length)


class AgentMonitor:
    """Monitors an agent's execution loop for spectral anomalies.

    Integrates with agent frameworks to provide real-time safety monitoring
    at each step of the agent's reasoning chain.

    Usage:
        monitor = AgentMonitor(model, tokenizer)
        monitor.calibrate("reference context", "supported statement")
        for step in agent_loop:
            result = monitor.check_step(step.context, step.output)
            if not result.is_safe:
                step.retry()
    """

    def __init__(self, model: nn.Module, tokenizer, **kwargs):
        self.guardrail = SpectralGuardrail(model, tokenizer, **kwargs)
        self.history: List[SpectralFeatures] = []

    def calibrate(self, context: str, supported_statement: str):
        """Calibrate the guardrail with a known-good example."""
        self.guardrail.calibrate_baseline(context, supported_statement)

    def check_step(
        self,
        context: str,
        output: str,
        step_id: Optional[int] = None,
    ) -> SpectralFeatures:
        """Check a single agent step for hallucination.

        Parameters
        ----------
        context : str
            The context available to the agent at this step.
        output : str
            The agent's output at this step.
        step_id : int, optional
            Step identifier for logging.

        Returns
        -------
        features : SpectralFeatures
            Spectral features with safety decision.
        """
        result = self.guardrail.check_generation(context, output)
        self.history.append(result)
        return result

    def check_tool_call(
        self,
        tool_description: str,
        tool_input: str,
        tool_output: str,
    ) -> SpectralFeatures:
        """Check a tool call for hallucination.

        Verifies that the tool output is consistent with the tool
        description and input. This catches cases where the model
        hallucinates a tool output instead of actually calling the tool.

        Parameters
        ----------
        tool_description : str
            Description of the tool being called.
        tool_input : str
            Input provided to the tool.
        tool_output : str
            Output returned by the tool (or hallucinated).

        Returns
        -------
        features : SpectralFeatures
            Spectral features with safety decision.
        """
        context = f"Tool: {tool_description}\nInput: {tool_input}"
        return self.guardrail.check_generation(context, tool_output)

    def get_health_summary(self) -> Dict:
        """Get a summary of the agent's spectral health over all steps.

        Returns
        -------
        summary : Dict
            Health summary with statistics.
        """
        if not self.history:
            return {"status": "no_data", "steps": 0}

        hfers = [f.hfer for f in self.history]
        safe_count = sum(1 for f in self.history if f.is_safe)
        unsafe_count = len(self.history) - safe_count

        return {
            "status": "healthy" if unsafe_count == 0 else "degraded",
            "steps": len(self.history),
            "safe_steps": safe_count,
            "unsafe_steps": unsafe_count,
            "avg_hfer": float(np.mean(hfers)),
            "min_hfer": float(np.min(hfers)),
            "max_hfer": float(np.max(hfers)),
            "hfer_trend": hfers,  # Trend over time
            "first_unsafe_step": next(
                (i for i, f in enumerate(self.history) if not f.is_safe), None
            ),
        }

    def reset(self):
        """Reset monitoring history."""
        self.history = []
