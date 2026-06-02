"""
Semantic Coherence Metrics for PhaseLLM

Implements standard NLP coherence metrics:
- BERT-based coherence: Measures semantic similarity between sentences
- Entity Grid coherence: Measures entity consistency across discourse
- Perplexity: Language model quality metric

References:
- Barzilay & Lapata (2008) - Modeling Local Coherence
- Roder et al. (2015) - Exploring the Space of Topic Coherence Measures
"""

import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F


class SemanticCoherenceMetrics:
    """
    Compute semantic coherence metrics for text.
    
    Metrics:
    - bert_coherence: BERT-based sentence similarity (requires BERT model)
    - entity_grid_coherence: Entity consistency across discourse
    - perplexity: Language model quality
    """
    
    def __init__(self, bert_model=None):
        """
        Initialize semantic coherence metrics.
        
        Args:
            bert_model: Optional BERT model for semantic similarity
        """
        self.bert_model = bert_model
        self.entity_cache = {}
    
    def compute_bert_coherence(
        self,
        sentences: List[str],
        embeddings: torch.Tensor = None
    ) -> float:
        """
        Compute BERT-based coherence as average sentence similarity.
        
        Args:
            sentences: List of sentences
            embeddings: Pre-computed BERT embeddings (B, T, d_model)
            
        Returns:
            coherence_score: Average cosine similarity between consecutive sentences
        """
        if embeddings is None:
            # Fallback: return 0.0 if no embeddings provided
            return 0.0
        
        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = F.cosine_similarity(
                embeddings[i:i+1], 
                embeddings[i+1:i+2], 
                dim=-1
            ).item()
            similarities.append(sim)
        
        # Return mean similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def compute_entity_grid_coherence(
        self,
        sentences: List[str],
        entities: List[List[str]] = None
    ) -> float:
        """
        Compute Entity Grid coherence (Barzilay & Lapata 2008).
        
        Measures entity consistency across discourse using transition patterns.
        
        Args:
            sentences: List of sentences
            entities: List of entities per sentence (optional)
            
        Returns:
            coherence_score: Entity grid coherence [0, 1]
        """
        if entities is None:
            # Simple entity extraction: extract capitalized words
            entities = []
            for sent in sentences:
                # Extract capitalized words as entities
                words = sent.split()
                sent_entities = [w for w in words if w[0].isupper() and len(w) > 1]
                entities.append(sent_entities)
        
        if not entities or len(entities) < 2:
            return 0.0
        
        # Build entity grid: track transitions
        transitions = 0
        total_transitions = 0
        
        for i in range(len(entities) - 1):
            current_entities = set(entities[i])
            next_entities = set(entities[i + 1])
            
            for entity in current_entities:
                total_transitions += 1
                if entity in next_entities:
                    transitions += 1  # Entity continues
                # else: entity disappears (coherence penalty)
        
        # Coherence = ratio of continuing entities
        return transitions / total_transitions if total_transitions > 0 else 0.0
    
    def compute_perplexity(
        self,
        loss: float
    ) -> float:
        """
        Compute perplexity from cross-entropy loss.
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            perplexity: exp(loss)
        """
        return math.exp(loss)
    
    def compute_all_metrics(
        self,
        sentences: List[str],
        embeddings: torch.Tensor = None,
        entities: List[List[str]] = None,
        loss: float = None
    ) -> Dict[str, float]:
        """
        Compute all semantic coherence metrics.
        
        Args:
            sentences: List of sentences
            embeddings: Pre-computed BERT embeddings
            entities: List of entities per sentence
            loss: Cross-entropy loss for perplexity
            
        Returns:
            metrics: Dict with all coherence scores
        """
        metrics = {}
        
        # BERT coherence
        metrics['bert_coherence'] = self.compute_bert_coherence(
            sentences, embeddings
        )
        
        # Entity grid coherence
        metrics['entity_grid_coherence'] = self.compute_entity_grid_coherence(
            sentences, entities
        )
        
        # Perplexity
        if loss is not None:
            metrics['perplexity'] = self.compute_perplexity(loss)
        
        return metrics


class PhaseSemanticCorrelation:
    """
    Compute correlation between phase coherence and semantic coherence.
    
    Metrics:
    - pearson_correlation: Linear correlation coefficient
    - spearman_correlation: Rank correlation coefficient
    - mutual_information: Statistical dependence
    """
    
    def __init__(self):
        """Initialize correlation tracker."""
        self.phase_history = []
        self.semantic_history = []
    
    def update(
        self,
        phase_coherence: float,
        semantic_coherence: float
    ):
        """
        Update history with new measurements.
        
        Args:
            phase_coherence: Phase coherence score
            semantic_coherence: Semantic coherence score
        """
        self.phase_history.append(phase_coherence)
        self.semantic_history.append(semantic_coherence)
    
    def compute_pearson_correlation(self) -> float:
        """
        Compute Pearson correlation coefficient.
        
        Returns:
            correlation: Pearson r in [-1, 1]
        """
        if len(self.phase_history) < 2:
            return 0.0
        
        import numpy as np
        
        phase = np.array(self.phase_history)
        semantic = np.array(self.semantic_history)
        
        # Compute correlation
        correlation_matrix = np.corrcoef(phase, semantic)
        return correlation_matrix[0, 1]
    
    def compute_spearman_correlation(self) -> float:
        """
        Compute Spearman rank correlation coefficient.
        
        Returns:
            correlation: Spearman rho in [-1, 1]
        """
        if len(self.phase_history) < 2:
            return 0.0
        
        import numpy as np
        from scipy.stats import spearmanr
        
        correlation, _ = spearmanr(self.phase_history, self.semantic_history)
        return correlation
    
    def compute_mutual_information(self, bins: int = 10) -> float:
        """
        Compute mutual information between phase and semantic coherence.
        
        Args:
            bins: Number of bins for histogram
            
        Returns:
            mi: Mutual information in bits
        """
        if len(self.phase_history) < 10:
            return 0.0
        
        import numpy as np
        from scipy.stats import entropy
        
        # Discretize into bins
        phase_hist, _ = np.histogram(self.phase_history, bins=bins, density=True)
        semantic_hist, _ = np.histogram(self.semantic_history, bins=bins, density=True)
        
        # 2D histogram
        hist_2d, _, _ = np.histogram2d(
            self.phase_history, 
            self.semantic_history, 
            bins=bins, 
            density=True
        )
        
        # Compute mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 0 and phase_hist[i] > 0 and semantic_hist[j] > 0:
                    mi += hist_2d[i, j] * np.log2(
                        hist_2d[i, j] / (phase_hist[i] * semantic_hist[j])
                    )
        
        return mi
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of correlation metrics.
        
        Returns:
            summary: Dict with all correlation metrics
        """
        summary = {
            'n_samples': len(self.phase_history),
            'mean_phase': sum(self.phase_history) / len(self.phase_history) if self.phase_history else 0.0,
            'mean_semantic': sum(self.semantic_history) / len(self.semantic_history) if self.semantic_history else 0.0,
            'pearson_correlation': self.compute_pearson_correlation(),
        }
        
        # Add Spearman if scipy available
        try:
            summary['spearman_correlation'] = self.compute_spearman_correlation()
        except ImportError:
            summary['spearman_correlation'] = None
        
        # Add mutual information
        summary['mutual_information'] = self.compute_mutual_information()
        
        return summary
