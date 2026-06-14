"""
Phase 1: Evaluation Framework

Comprehensive metrics for Phase-LM training:
- Coherence metrics (phase alignment quality)
- Cross-modal alignment verification  
- Structure preservation metrics
- Baseline comparisons (ResonanceAttention vs dot-product)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    
    # Coherence metrics
    coherence_mean: float = 0.0
    coherence_std: float = 0.0
    coherence_max: float = 0.0
    
    # Phase alignment
    phase_alignment_score: float = 0.0
    phase_variance: float = 0.0
    
    # Structure preservation
    attractor_preservation: float = 0.0
    temporal_consistency: float = 0.0
    
    # Cross-modal metrics
    cross_modal_alignment: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Attention mechanism comparison
    attention_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging"""
        result = {
            'coherence_mean': self.coherence_mean,
            'coherence_std': self.coherence_std,
            'coherence_max': self.coherence_max,
            'phase_alignment_score': self.phase_alignment_score,
            'phase_variance': self.phase_variance,
            'attractor_preservation': self.attractor_preservation,
            'temporal_consistency': self.temporal_consistency,
        }
        
        # Add cross-modal metrics
        for (m1, m2), score in self.cross_modal_alignment.items():
            result[f'cross_modal_{m1}_vs_{m2}'] = score
        
        # Add attention metrics
        for name, score in self.attention_scores.items():
            result[f'attention_{name}'] = score
        
        return result


class CoherenceMetrics:
    """Compute coherence-related metrics"""
    
    @staticmethod
    def phase_alignment_score(
        phase_output: torch.Tensor,
        target_phase: torch.Tensor,
        coherence_weights: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Measure phase alignment between output and target.
        
        High score = output phase closely matches target
        (weighted by coherence if provided)
        
        Parameters
        ----------
        phase_output : torch.Tensor
            Predicted phase [batch, time, freq]
        target_phase : torch.Tensor
            Ground truth phase [batch, time, freq]
        coherence_weights : Optional[torch.Tensor]
            Coherence weights for importance weighting
        
        Returns
        -------
        float
            Phase alignment score [0-1]
        """
        # Compute phase distance (circular)
        phase_diff = torch.abs(
            torch.angle(torch.exp(1j * (phase_output - target_phase)))
        )  # [0, π]
        
        # Convert to alignment score [0, 1] (0 diff = 1 score)
        alignment = 1.0 - (phase_diff / np.pi)
        
        # Weight by coherence if provided
        if coherence_weights is not None:
            alignment = (alignment * coherence_weights).sum() / (coherence_weights.sum() + 1e-8)
        else:
            alignment = alignment.mean()
        
        return float(alignment.item())
    
    @staticmethod
    def phase_variance(
        phase_output: torch.Tensor,
        coherence_threshold: float = 0.5,
    ) -> float:
        """
        Measure variance in predicted phase.
        
        High variance = phase is varied (good)
        Low variance = phase is constant (collapse)
        
        Parameters
        ----------
        phase_output : torch.Tensor
            Predicted phase [batch, time, freq]
        coherence_threshold : float
            Ignore frequencies with coherence < threshold
        
        Returns
        -------
        float
            Phase variance score
        """
        # Unwrap phase to handle circular nature
        phase_unwrapped = torch.angle(torch.exp(1j * phase_output))
        
        # Compute variance over time and frequency
        variance = phase_unwrapped.var(dim=(0, 1)).mean()
        
        return float(variance.item())
    
    @staticmethod
    def coherence_statistics(
        coherence: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Compute coherence statistics.
        
        Parameters
        ----------
        coherence : torch.Tensor
            Coherence scores [batch, time, freq] or [batch, time]
        
        Returns
        -------
        Tuple[float, float, float]
            Mean, std, max coherence
        """
        mean = float(coherence.mean().item())
        std = float(coherence.std().item())
        max_val = float(coherence.max().item())
        
        return mean, std, max_val


class StructurePreservationMetrics:
    """Measure how well structure is preserved through encoding/decoding"""
    
    @staticmethod
    def attractor_preservation(
        original_attractors: List[Dict],
        reconstructed_attractors: List[Dict],
        frequency_tolerance: float = 0.05,  # 5% frequency tolerance
        amplitude_tolerance: float = 0.1,   # 10% amplitude tolerance
    ) -> float:
        """
        Measure percentage of attractors preserved during reconstruction.
        
        Parameters
        ----------
        original_attractors : List[Dict]
            Original attractors with keys: time, frequency, amplitude, coherence
        reconstructed_attractors : List[Dict]
            Reconstructed attractors
        frequency_tolerance : float
            Relative tolerance for frequency matching
        amplitude_tolerance : float
            Relative tolerance for amplitude matching
        
        Returns
        -------
        float
            Preservation ratio [0-1]
        """
        if not original_attractors:
            return 1.0
        
        preserved = 0
        
        for orig in original_attractors:
            # Find closest match in reconstructed
            best_distance = float('inf')
            
            for recon in reconstructed_attractors:
                # Time proximity (must be within ±1 frame)
                time_dist = abs(orig['time'] - recon['time'])
                if time_dist > 1:
                    continue
                
                # Frequency proximity
                freq_error = abs(orig['frequency'] - recon['frequency']) / (orig['frequency'] + 1e-8)
                if freq_error > frequency_tolerance:
                    continue
                
                # Amplitude proximity
                amp_error = abs(orig['amplitude'] - recon['amplitude']) / (orig['amplitude'] + 1e-8)
                if amp_error > amplitude_tolerance:
                    continue
                
                best_distance = 0
                break
            
            if best_distance == 0:
                preserved += 1
        
        return preserved / len(original_attractors)
    
    @staticmethod
    def temporal_consistency(
        features: torch.Tensor,
        window_size: int = 3,
    ) -> float:
        """
        Measure temporal consistency of features.
        
        High consistency = features change smoothly over time
        
        Parameters
        ----------
        features : torch.Tensor
            Features over time [batch, time, dim]
        window_size : int
            Window size for consistency checking
        
        Returns
        -------
        float
            Consistency score [0-1]
        """
        # Compute frame-to-frame differences
        diffs = torch.abs(features[:, 1:] - features[:, :-1])  # [batch, time-1, dim]
        
        # Smooth using moving average
        consistency = 1.0 / (1.0 + diffs.mean())
        
        return float(consistency.item())


class CrossModalMetrics:
    """Measure alignment between modalities"""
    
    @staticmethod
    def cross_modal_alignment(
        embeddings_m1: torch.Tensor,
        embeddings_m2: torch.Tensor,
        modality_1: str = "audio",
        modality_2: str = "video",
        metric: str = "cosine",
    ) -> float:
        """
        Measure alignment between two modalities.
        
        High alignment = embeddings from same content are similar
        
        Parameters
        ----------
        embeddings_m1 : torch.Tensor
            Embeddings from modality 1 [batch, dim]
        embeddings_m2 : torch.Tensor
            Embeddings from modality 2 [batch, dim]
        modality_1 : str
            Name of modality 1
        modality_2 : str
            Name of modality 2
        metric : str
            "cosine" or "euclidean"
        
        Returns
        -------
        float
            Alignment score [0-1]
        """
        # Normalize embeddings
        m1_norm = F.normalize(embeddings_m1, p=2, dim=-1)
        m2_norm = F.normalize(embeddings_m2, p=2, dim=-1)
        
        if metric == "cosine":
            # Cosine similarity [-1, 1] → [0, 1]
            similarity = torch.mm(m1_norm, m2_norm.t())
            alignment = (similarity.diag().mean() + 1.0) / 2.0
        else:  # euclidean
            # Euclidean distance: lower is better
            distances = torch.norm(m1_norm - m2_norm, dim=-1)
            alignment = 1.0 / (1.0 + distances.mean())
        
        return float(alignment.item())


class AttentionComparison:
    """Compare attention mechanisms"""
    
    @staticmethod
    def compare_attention_patterns(
        resonance_attention: torch.Tensor,
        dot_product_attention: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compare attention patterns between ResonanceAttention and dot-product.
        
        Parameters
        ----------
        resonance_attention : torch.Tensor
            Attention weights from ResonanceAttention [batch, heads, time, time]
        dot_product_attention : torch.Tensor
            Attention weights from dot-product attention [batch, heads, time, time]
        
        Returns
        -------
        Dict[str, float]
            Metrics comparing the two attention patterns
        """
        # Normalize both
        ra_norm = F.normalize(resonance_attention.reshape(-1, resonance_attention.shape[-1]), p=2, dim=-1)
        dp_norm = F.normalize(dot_product_attention.reshape(-1, dot_product_attention.shape[-1]), p=2, dim=-1)
        
        # Compute metrics
        cosine_sim = F.cosine_similarity(ra_norm, dp_norm).mean()
        
        # KL divergence (treating as distributions)
        ra_prob = F.softmax(resonance_attention, dim=-1).flatten()
        dp_prob = F.softmax(dot_product_attention, dim=-1).flatten()
        kl_div = F.kl_div(dp_prob.log(), ra_prob, reduction='mean')
        
        # Entropy difference
        ra_entropy = -(ra_prob * torch.log(ra_prob + 1e-8)).sum()
        dp_entropy = -(dp_prob * torch.log(dp_prob + 1e-8)).sum()
        entropy_diff = float((ra_entropy - dp_entropy).item())
        
        return {
            'cosine_similarity': float(cosine_sim.item()),
            'kl_divergence': float(kl_div.item()),
            'entropy_difference': entropy_diff,
        }


class Phase1Evaluator:
    """
    Complete Phase 1 evaluation suite.
    
    Combines all metrics for comprehensive evaluation.
    """
    
    def __init__(self):
        self.coherence_metrics = CoherenceMetrics()
        self.structure_metrics = StructurePreservationMetrics()
        self.cross_modal_metrics = CrossModalMetrics()
        self.attention_metrics = AttentionComparison()
    
    def evaluate(
        self,
        phase_output: torch.Tensor,
        target_phase: torch.Tensor,
        coherence: torch.Tensor,
        attractors_original: Optional[List[Dict]] = None,
        attractors_reconstructed: Optional[List[Dict]] = None,
        features: Optional[torch.Tensor] = None,
        cross_modal_pairs: Optional[Dict] = None,
        attention_comparison: Optional[Dict] = None,
    ) -> EvaluationMetrics:
        """
        Run complete Phase 1 evaluation.
        
        Parameters
        ----------
        phase_output : torch.Tensor
            Predicted phase
        target_phase : torch.Tensor
            Ground truth phase
        coherence : torch.Tensor
            Coherence scores
        attractors_original : Optional[List[Dict]]
            Original attractors
        attractors_reconstructed : Optional[List[Dict]]
            Reconstructed attractors
        features : Optional[torch.Tensor]
            Features for temporal consistency
        cross_modal_pairs : Optional[Dict]
            Cross-modal embedding pairs
        attention_comparison : Optional[Dict]
            Attention patterns to compare
        
        Returns
        -------
        EvaluationMetrics
            Comprehensive evaluation results
        """
        metrics = EvaluationMetrics()
        
        # Coherence metrics
        metrics.coherence_mean, metrics.coherence_std, metrics.coherence_max = \
            self.coherence_metrics.coherence_statistics(coherence)
        
        metrics.phase_alignment_score = self.coherence_metrics.phase_alignment_score(
            phase_output, target_phase, coherence
        )
        
        metrics.phase_variance = self.coherence_metrics.phase_variance(phase_output)
        
        # Structure preservation
        if attractors_original is not None and attractors_reconstructed is not None:
            metrics.attractor_preservation = self.structure_metrics.attractor_preservation(
                attractors_original, attractors_reconstructed
            )
        
        if features is not None:
            metrics.temporal_consistency = self.structure_metrics.temporal_consistency(features)
        
        # Cross-modal alignment
        if cross_modal_pairs is not None:
            for (m1, m2), (emb1, emb2) in cross_modal_pairs.items():
                score = self.cross_modal_metrics.cross_modal_alignment(emb1, emb2, m1, m2)
                metrics.cross_modal_alignment[(m1, m2)] = score
        
        # Attention comparison
        if attention_comparison is not None:
            metrics.attention_scores = self.attention_metrics.compare_attention_patterns(
                attention_comparison.get('resonance_attention'),
                attention_comparison.get('dot_product_attention'),
            )
        
        return metrics
