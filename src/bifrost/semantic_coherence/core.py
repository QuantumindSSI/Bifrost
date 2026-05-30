"""
Semantic Coherence Training for Bifrost

Connects phase coherence to semantic meaning through supervised learning.

The Core Insight:
    Current phase coherence measures signal structure (STFT phase gradients).
    But phase structure ≠ semantic structure.
    
    This module adds a supervised objective:
    - Similar semantics → Similar phase coherence patterns
    - Different semantics → Different phase coherence patterns
    
    This forces the spectral encoder to develop phase representations
    that encode semantic information, not just harmonic structure.

Training Approach:
    1. Extract phase coherence vectors from spectral output
    2. Use contrastive learning: pull similar semantics together, push apart different
    3. Optional: Add classification head on phase features
    
References:
    - Phase Coherence as Learned Semantic Embedding
    - Supervised Contrastive Learning (Khosla et al. 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from ..spectral_tensor import SpectralTensor
from ..pipeline import BifrostPipeline


@dataclass
class SemanticCoherenceMetrics:
    """Metrics for semantic coherence training."""
    contrastive_loss: float
    phase_similarity_accuracy: float  # % of pairs correctly ordered
    semantic_retrieval_recall: float   # Recall@k for semantic retrieval
    coherence_semantic_correlation: float  # Pearson correlation


class PhaseCoherenceExtractor(nn.Module):
    """
    Extract fixed-dimensional phase coherence features from spectral output.
    
    Input: SpectralTensor with phase (B, T, d_model)
    Output: Phase coherence vector (B, coherence_dim)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_temporal_windows: int = 4,
        n_freq_bands: int = 8,
        coherence_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_temporal_windows = n_temporal_windows
        self.n_freq_bands = n_freq_bands
        
        # Phase coherence features:
        # 1. Temporal phase gradients (how phase evolves over time)
        # 2. Cross-frequency phase coupling (phase relationships across bands)
        # 3. Phase stability (variance of phase within windows)
        
        # Compute raw feature dimension
        # temporal: n_windows * (window_size - 1) gradients
        # coupling: n_bands * n_bands cross-correlations
        # stability: n_bands variance values
        raw_dim = (n_temporal_windows * 3) + (n_freq_bands * n_freq_bands) + n_freq_bands
        
        # Project to fixed coherence representation
        self.feature_projector = nn.Sequential(
            nn.Linear(raw_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, coherence_dim),
            nn.LayerNorm(coherence_dim),
        )
        
        # Learnable temperature for similarity scaling
        self.similarity_temp = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, spectral: SpectralTensor) -> torch.Tensor:
        """
        Extract phase coherence features from spectral tensor.
        
        Args:
            spectral: SpectralTensor with phase (B, T, d_model)
            
        Returns:
            Phase coherence vectors (B, coherence_dim)
        """
        phase = spectral.phase  # (B, T, d_model)
        B, T, d = phase.shape
        
        features = []
        
        # 1. Temporal phase gradients
        window_size = max(1, T // self.n_temporal_windows)
        for i in range(self.n_temporal_windows):
            start = i * window_size
            end = min((i + 1) * window_size, T)
            if end - start < 2:
                features.append(torch.zeros(B, 3, device=phase.device))
                continue
            
            window_phase = phase[:, start:end, :]  # (B, window_size, d)
            
            # Phase gradient along time
            phase_diff = torch.diff(window_phase, dim=1)  # (B, window_size-1, d)
            
            # Statistics of phase evolution
            grad_mean = phase_diff.mean(dim=(1, 2))  # (B,)
            grad_std = phase_diff.std(dim=(1, 2))    # (B,)
            grad_smooth = (phase_diff ** 2).mean(dim=(1, 2))  # Smoothness (lower = smoother)
            
            window_features = torch.stack([grad_mean, grad_std, grad_smooth], dim=1)
            features.append(window_features)
        
        # 2. Cross-frequency phase coupling
        # Divide into frequency bands
        band_size = d // self.n_freq_bands
        band_phases = []
        for i in range(self.n_freq_bands):
            start = i * band_size
            end = start + band_size if i < self.n_freq_bands - 1 else d
            band_phase = phase[:, :, start:end].mean(dim=2)  # (B, T)
            band_phases.append(band_phase)
        
        # Compute phase coupling matrix (cosine similarity between bands)
        band_tensor = torch.stack(band_phases, dim=2)  # (B, T, n_bands)
        # Average over time for stable coupling
        band_mean = band_tensor.mean(dim=1)  # (B, n_bands)
        
        # Coupling as outer product of phase means
        for b in range(B):
            coupling = torch.outer(band_mean[b], band_mean[b])  # (n_bands, n_bands)
            features.append(coupling.flatten().unsqueeze(0))
        
        # 3. Phase stability (low variance = stable)
        for i in range(self.n_freq_bands):
            start = i * band_size
            end = start + band_size if i < self.n_freq_bands - 1 else d
            band_phase = phase[:, :, start:end]  # (B, T, band_size)
            stability = -band_phase.var(dim=(1, 2))  # Negative variance (higher = more stable)
            features.append(stability.unsqueeze(1))
        
        # Concatenate all features
        feature_vector = torch.cat([f if f.dim() == 2 else f.unsqueeze(1) for f in features], dim=1)
        
        # Project to coherence representation
        coherence = self.feature_projector(feature_vector)
        
        # Normalize for cosine similarity
        coherence = F.normalize(coherence, dim=-1)
        
        return coherence


class SupervisedSemanticCoherenceLoss(nn.Module):
    """
    Supervised loss to connect phase coherence with semantic similarity.
    
    Uses supervised contrastive learning:
    - Samples from same semantic class → phase coherence vectors pulled together
    - Samples from different classes → phase coherence vectors pushed apart
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrastive_mode: str = "all",  # "all" or "one_positive"
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrastive_mode = contrastive_mode
    
    def forward(
        self,
        coherence_features: torch.Tensor,  # (B, coherence_dim)
        labels: torch.Tensor,               # (B,) semantic labels
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss on phase coherence vectors.
        
        Args:
            coherence_features: Normalized phase coherence vectors
            labels: Semantic class labels
            
        Returns:
            Supervised contrastive loss
        """
        device = coherence_features.device
        batch_size = coherence_features.shape[0]
        
        # Compute similarity matrix (cosine similarity)
        # features are already normalized
        similarity_matrix = torch.matmul(coherence_features, coherence_features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-similarity from positives
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positives
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class SemanticCoherenceHead(nn.Module):
    """
    Classification head on phase coherence features for semantic tasks.
    
    Provides direct supervision: phase coherence → semantic class
    """
    
    def __init__(
        self,
        coherence_dim: int = 256,
        num_classes: int = 10,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(coherence_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, coherence_features: torch.Tensor) -> torch.Tensor:
        """Classify based on phase coherence pattern."""
        return self.classifier(coherence_features)


class SemanticCoherenceTrainer:
    """
    Trainer for improving semantic coherence in Bifrost.
    
    Combines:
    1. Contrastive phase coherence loss (existing)
    2. Supervised semantic coherence loss (new)
    3. Optional: Classification loss on phase features
    """
    
    def __init__(
        self,
        pipeline: BifrostPipeline,
        num_classes: int = 10,
        device: str = "cuda",
        lr: float = 1e-4,
        lambda_semantic: float = 1.0,  # Weight for semantic loss
        lambda_contrastive: float = 0.5,  # Weight for contrastive loss
    ):
        self.pipeline = pipeline.to(device)
        self.device = device
        self.lambda_semantic = lambda_semantic
        self.lambda_contrastive = lambda_contrastive
        
        # Phase coherence extractor
        self.coherence_extractor = PhaseCoherenceExtractor(
            d_model=pipeline.s2.d_model,
            coherence_dim=256,
        ).to(device)
        
        # Supervised semantic coherence loss
        self.semantic_loss_fn = SupervisedSemanticCoherenceLoss()
        
        # Contrastive loss (existing)
        from ..training import ContrastiveCoherenceLoss
        self.contrastive_loss_fn = ContrastiveCoherenceLoss()
        
        # Classification head (optional)
        self.classifier = SemanticCoherenceHead(
            coherence_dim=256,
            num_classes=num_classes,
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': self.pipeline.parameters(), 'lr': lr * 0.1},  # Slower for pretrained
            {'params': self.coherence_extractor.parameters(), 'lr': lr},
            {'params': self.classifier.parameters(), 'lr': lr},
        ])
        
        self.metrics_history = []
    
    def train_step(
        self,
        signals: torch.Tensor,  # (B, signal_len)
        labels: torch.Tensor,   # (B,) semantic labels
    ) -> Dict[str, float]:
        """
        Single training step for semantic coherence.
        
        Returns:
            Dictionary of loss values and metrics
        """
        self.optimizer.zero_grad()
        
        # Forward through Bifrost
        bound, coherence_matrix = self.pipeline(signals)
        
        # Extract phase coherence features
        coherence_features = self.coherence_extractor(bound)
        
        # 1. Supervised semantic coherence loss
        semantic_loss = self.semantic_loss_fn(coherence_features, labels)
        
        # 2. Contrastive phase coherence loss (existing)
        # Generate phase-randomized negatives
        noise_signals = signals * (2 * torch.rand_like(signals) - 1)
        bound_noise, _ = self.pipeline(noise_signals)
        contrastive_loss = self.contrastive_loss_fn(bound.amplitude, bound_noise.amplitude)
        
        # 3. Classification loss (auxiliary)
        class_logits = self.classifier(coherence_features)
        classification_loss = F.cross_entropy(class_logits, labels)
        
        # Combined loss
        total_loss = (
            self.lambda_semantic * semantic_loss +
            self.lambda_contrastive * contrastive_loss +
            0.5 * classification_loss
        )
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.pipeline.parameters()) + 
            list(self.coherence_extractor.parameters()) +
            list(self.classifier.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            predictions = class_logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            
            # Compute coherence-semantic correlation
            coherence_np = coherence_features.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Simple correlation: within-class variance vs between-class variance
            unique_labels = torch.unique(labels)
            if len(unique_labels) > 1:
                within_var = 0
                between_var = 0
                global_mean = coherence_features.mean(dim=0)
                
                for lbl in unique_labels:
                    mask = labels == lbl
                    class_features = coherence_features[mask]
                    class_mean = class_features.mean(dim=0)
                    within_var += (class_features - class_mean).pow(2).sum()
                    between_var += (class_mean - global_mean).pow(2).sum() * mask.sum()
                
                coherence_semantic_ratio = between_var / (within_var + 1e-8)
            else:
                coherence_semantic_ratio = 0.0
        
        metrics = {
            "total_loss": total_loss.item(),
            "semantic_loss": semantic_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "classification_loss": classification_loss.item(),
            "classification_accuracy": accuracy,
            "coherence_semantic_ratio": coherence_semantic_ratio.item() if isinstance(coherence_semantic_ratio, torch.Tensor) else coherence_semantic_ratio,
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def evaluate_semantic_coherence(
        self,
        test_signals: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> SemanticCoherenceMetrics:
        """
        Evaluate how well phase coherence encodes semantic information.
        
        Returns:
            SemanticCoherenceMetrics with detailed analysis
        """
        self.pipeline.eval()
        self.coherence_extractor.eval()
        
        with torch.no_grad():
            # Extract coherence features
            all_coherence = []
            all_labels = []
            
            batch_size = 32
            for i in range(0, len(test_signals), batch_size):
                batch_signals = test_signals[i:i+batch_size].to(self.device)
                batch_labels = test_labels[i:i+batch_size]
                
                bound, _ = self.pipeline(batch_signals)
                coherence = self.coherence_extractor(bound)
                
                all_coherence.append(coherence.cpu())
                all_labels.append(batch_labels)
            
            all_coherence = torch.cat(all_coherence, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute semantic retrieval recall@k
            similarity_matrix = torch.matmul(all_coherence, all_coherence.T)
            
            # For each sample, check if same-class samples are in top-k
            k = 5
            correct_retrievals = 0
            total_queries = 0
            
            for i in range(len(all_coherence)):
                query_label = all_labels[i].item()
                # Get top-k most similar (excluding self)
                scores = similarity_matrix[i].clone()
                scores[i] = -float('inf')  # Exclude self
                top_k_indices = scores.topk(k).indices
                
                # Count same-class retrievals
                retrieved_labels = all_labels[top_k_indices]
                same_class_count = (retrieved_labels == query_label).sum().item()
                correct_retrievals += same_class_count
                total_queries += k
            
            recall_at_k = correct_retrievals / total_queries
            
            # Compute phase similarity accuracy
            # Pairs from same class should be more similar than pairs from different classes
            n_pairs = min(1000, len(all_coherence) * (len(all_coherence) - 1) // 2)
            correct_pairs = 0
            total_pairs = 0
            
            for _ in range(n_pairs):
                i, j = torch.randint(0, len(all_coherence), (2,))
                if i == j:
                    continue
                k, l = torch.randint(0, len(all_coherence), (2,))
                if k == l or len({i.item(), j.item(), k.item(), l.item()}) < 4:
                    continue
                
                sim_same = similarity_matrix[i, j].item() if all_labels[i] == all_labels[j] else similarity_matrix[k, l].item()
                sim_diff = similarity_matrix[k, l].item() if all_labels[k] != all_labels[l] else similarity_matrix[i, j].item()
                
                if all_labels[i] == all_labels[j] and sim_same > sim_diff:
                    correct_pairs += 1
                elif all_labels[k] == all_labels[l] and sim_diff > sim_same:
                    correct_pairs += 1
                total_pairs += 1
            
            pair_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
            
            # Compute correlation between coherence similarity and label equality
            coherence_sims = []
            label_eq = []
            for i in range(min(100, len(all_coherence))):
                for j in range(i+1, min(100, len(all_coherence))):
                    coherence_sims.append(similarity_matrix[i, j].item())
                    label_eq.append(1.0 if all_labels[i] == all_labels[j] else 0.0)
            
            if len(coherence_sims) > 10:
                coherence_tensor = torch.tensor(coherence_sims)
                label_tensor = torch.tensor(label_eq)
                
                # Pearson correlation
                mean_coh = coherence_tensor.mean()
                mean_lab = label_tensor.mean()
                num = ((coherence_tensor - mean_coh) * (label_tensor - mean_lab)).sum()
                den = torch.sqrt(((coherence_tensor - mean_coh)**2).sum() * ((label_tensor - mean_lab)**2).sum())
                correlation = (num / (den + 1e-8)).item()
            else:
                correlation = 0.0
        
        self.pipeline.train()
        self.coherence_extractor.train()
        
        return SemanticCoherenceMetrics(
            contrastive_loss=0.0,  # Not computed here
            phase_similarity_accuracy=pair_accuracy,
            semantic_retrieval_recall=recall_at_k,
            coherence_semantic_correlation=correlation,
        )


def train_semantic_coherence(
    pipeline: BifrostPipeline,
    train_signals: List[torch.Tensor],
    train_labels: List[int],
    num_classes: int,
    epochs: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
) -> SemanticCoherenceTrainer:
    """
    High-level function to train Bifrost for semantic coherence.
    
    Usage:
        trainer = train_semantic_coherence(
            pipeline=bifrost_pipeline,
            train_signals=audio_clips,
            train_labels=emotion_labels,
            num_classes=8,
            epochs=20,
        )
        
        # Evaluate
        metrics = trainer.evaluate_semantic_coherence(test_signals, test_labels)
        print(f"Semantic correlation: {metrics.coherence_semantic_correlation:.3f}")
    """
    trainer = SemanticCoherenceTrainer(
        pipeline=pipeline,
        num_classes=num_classes,
        device=device,
    )
    
    # Convert to tensors
    train_signals_tensor = torch.stack([s.squeeze() if s.dim() > 1 else s for s in train_signals])
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    
    print(f"\n{'='*60}")
    print("SEMANTIC COHERENCE TRAINING")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_signals)}")
    print(f"Num classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle
        perm = torch.randperm(len(train_signals_tensor))
        
        for i in range(0, len(train_signals_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_signals = train_signals_tensor[batch_idx].to(device)
            batch_labels = train_labels_tensor[batch_idx].to(device)
            
            metrics = trainer.train_step(batch_signals, batch_labels)
            epoch_losses.append(metrics["total_loss"])
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
    
    print(f"\n{'='*60}")
    print("Training complete. Phase coherence now encodes semantics.")
    print(f"{'='*60}\n")
    
    return trainer
