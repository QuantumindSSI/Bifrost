"""
S3: Phase-Lock Attractor Learning Module

Implements learned attractor dynamics to replace placeholder values.
Provides stability scores, phase coherence tracking, and attractor clustering.

References:
    - Phase-Lock Bridge concept from Bifrost architecture
    - Attractor dynamics in neural oscillators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..spectral_tensor import SpectralTensor


@dataclass
class FrequencyAttractor:
    """Represents a learned frequency attractor with dynamics."""
    centroid: torch.Tensor           # (d_model,) - amplitude profile
    phase_signature: torch.Tensor  # (n_bands,) - phase band means
    amplitude_profile: torch.Tensor  # (d_model,)
    stability: torch.Tensor          # scalar, learned
    domain: str
    attractor_id: str
    metadata: Dict
    
    # Learned dynamics
    coherence_history: List[float] = None
    drift_velocity: torch.Tensor = None
    
    def __post_init__(self):
        if self.coherence_history is None:
            self.coherence_history = []
        if self.drift_velocity is None:
            self.drift_velocity = torch.zeros_like(self.centroid)


class AttractorLearningModule(nn.Module):
    """
    Learned attractor dynamics module for S3 Phase-Lock Bridge.
    
    Replaces placeholder stability=0.5 with learned stability based on:
    1. Temporal consistency of attractor position
    2. Phase coherence within attractor region
    3. Contrast with neighboring attractors
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_bands: int = 8,
        n_attractors: int = 16,
        stability_threshold: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_bands = n_bands
        self.n_attractors = n_attractors
        self.stability_threshold = stability_threshold
        
        # Learnable attractor prototypes (centroids in frequency space)
        self.attractor_prototypes = nn.Parameter(
            torch.randn(n_attractors, d_model) * 0.1
        )
        
        # Phase pattern prototypes for each attractor
        self.phase_prototypes = nn.Parameter(
            torch.randn(n_attractors, n_bands) * 0.1
        )
        
        # Stability predictor network
        # Input: [attractor_features (n_bands), phase_coherence (n_bands), temporal_variance (1)]
        # Total input size: 2 * n_bands + 1
        self.stability_predictor = nn.Sequential(
            nn.Linear(2 * n_bands + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Stability in [0, 1]
        )
        
        # Temporal tracking for stability calculation
        self.register_buffer('attractor_history', torch.zeros(n_attractors, 10, d_model))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
        # Learned similarity temperature
        self.similarity_temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, spectral: SpectralTensor) -> Tuple[List[FrequencyAttractor], torch.Tensor]:
        """
        Extract attractors from spectral tensor with learned stability.
        
        Args:
            spectral: Input SpectralTensor with (B, T, d_model) or (B, d_model)
            
        Returns:
            attractors: List of FrequencyAttractor with learned stability
            assignment_probs: (B, n_attractors) soft assignment to attractors
        """
        # Handle both 2D and 3D inputs
        if spectral.amplitude.dim() == 3:
            B, T, d = spectral.amplitude.shape
            amp = spectral.amplitude.reshape(-1, d)  # (B*T, d)
            phase = spectral.phase.reshape(-1, d)
        else:
            B = spectral.amplitude.shape[0]
            T = 1
            amp = spectral.amplitude
            phase = spectral.phase
        
        # === COMPUTE ATTRACTOR ASSIGNMENTS ===
        # Soft assignment based on amplitude similarity to prototypes
        amp_normalized = F.normalize(amp, dim=-1)
        proto_normalized = F.normalize(self.attractor_prototypes, dim=-1)
        
        # Cosine similarity with learnable temperature
        similarities = torch.matmul(amp_normalized, proto_normalized.T) / F.softplus(self.similarity_temp)
        assignment_probs = F.softmax(similarities, dim=-1)  # (B*T, n_attractors)
        
        # === EXTRACT PHASE BAND FEATURES ===
        band_size = max(1, self.d_model // self.n_bands)
        phase_bands = []
        for b in range(self.n_bands):
            start = b * band_size
            end = start + band_size if b < self.n_bands - 1 else self.d_model
            phase_bands.append(phase[:, start:end].mean(dim=-1))
        phase_features = torch.stack(phase_bands, dim=-1)  # (B*T, n_bands)
        
        # === COMPUTE LEARNED STABILITY ===
        attractors = []
        for i in range(self.n_attractors):
            # Weighted centroid based on assignments
            weights = assignment_probs[:, i:i+1]  # (B*T, 1)
            centroid = (amp * weights).sum(dim=0) / (weights.sum() + 1e-8)
            
            # Phase signature weighted by assignment
            phase_sig = (phase_features * weights).sum(dim=0) / (weights.sum() + 1e-8)
            
            # Temporal consistency (lower variance = higher stability)
            temporal_var = self._compute_temporal_variance(i, centroid)
            
            # Phase coherence within attractor
            phase_coherence = torch.cos(phase_sig - self.phase_prototypes[i]).mean()
            
            # Predict stability
            stability_input = torch.cat([
                centroid[:self.n_bands],  # Use first bands for efficiency
                phase_sig,
                temporal_var.unsqueeze(0),
            ])
            stability = self.stability_predictor(stability_input)
            
            # Update history for next iteration
            self._update_history(i, centroid.detach())
            
            attractor = FrequencyAttractor(
                centroid=centroid,
                phase_signature=phase_sig,
                amplitude_profile=centroid,
                stability=stability.item(),
                domain="learned",
                attractor_id=f"s3_attractor_{i:03d}",
                metadata={
                    "assignment_mass": weights.sum().item(),
                    "phase_coherence": phase_coherence.item(),
                    "temporal_variance": temporal_var.item(),
                },
                coherence_history=[phase_coherence.item()],
                drift_velocity=self.attractor_history[i, -1] - centroid.detach(),
            )
            attractors.append(attractor)
        
        # Reshape assignment probs back to (B, T, n_attractors) if needed
        if T > 1:
            assignment_probs = assignment_probs.reshape(B, T, -1)
        else:
            assignment_probs = assignment_probs.reshape(B, -1)
        
        return attractors, assignment_probs
    
    def _compute_temporal_variance(self, attractor_idx: int, current: torch.Tensor) -> torch.Tensor:
        """Compute temporal variance for stability estimation."""
        history = self.attractor_history[attractor_idx]  # (10, d_model)
        
        # Check if history has been initialized
        if history.abs().sum() < 1e-6:
            return torch.tensor(1.0, device=current.device)  # High variance initially
        
        # Variance from history mean
        mean_pos = history.mean(dim=0)
        variance = ((current - mean_pos) ** 2).mean()
        
        return variance
    
    def _update_history(self, attractor_idx: int, centroid: torch.Tensor):
        """Update attractor position history for temporal tracking."""
        ptr = self.history_ptr.item() % 10
        self.attractor_history[attractor_idx, ptr] = centroid
        self.history_ptr[0] = (ptr + 1) % 10
    
    def compute_attractor_loss(
        self,
        attractors: List[FrequencyAttractor],
        assignment_probs: torch.Tensor,
        target_coherence: float = 0.7,
    ) -> torch.Tensor:
        """
        Compute training loss for attractor learning.
        
        Loss components:
        1. Coherence loss: Encourage high phase coherence
        2. Separation loss: Encourage distinct attractors
        3. Stability loss: Encourage high stability scores
        4. Coverage loss: Ensure all attractors are used
        """
        losses = []
        
        # 1. Coherence loss (encourage phase alignment within attractors)
        coherences = torch.tensor([a.metadata["phase_coherence"] for a in attractors])
        coherence_loss = F.mse_loss(coherences, torch.full_like(coherences, target_coherence))
        losses.append(coherence_loss)
        
        # 2. Separation loss (attractors should be distinct)
        # Compute pairwise distances between prototypes
        protos = self.attractor_prototypes
        distances = torch.cdist(protos, protos, p=2)  # (n_attractors, n_attractors)
        # Mask diagonal
        mask = torch.eye(self.n_attractors, device=distances.device) * 1e8
        min_distance = (distances + mask).min(dim=1)[0].mean()
        # Encourage minimum distance of 1.0
        separation_loss = F.relu(1.0 - min_distance).mean()
        losses.append(separation_loss)
        
        # 3. Stability loss (encourage high stability predictions)
        stabilities = torch.tensor([a.stability for a in attractors])
        stability_loss = (1.0 - stabilities).mean()  # Penalize low stability
        losses.append(stability_loss)
        
        # 4. Coverage loss (all attractors should have some assignment)
        # assignment_probs shape: (B, n_attractors) or (B, T, n_attractors)
        if assignment_probs.dim() == 3:
            usage = assignment_probs.mean(dim=(0, 1))  # (n_attractors,)
        else:
            usage = assignment_probs.mean(dim=0)  # (n_attractors,)
        coverage_loss = -torch.log(usage + 1e-8).mean()  # Encourage uniform usage
        losses.append(coverage_loss * 0.1)  # Weighted down
        
        return sum(losses)


class PhaseLockBridge(nn.Module):
    """
    Complete S3 Phase-Lock Bridge with learned attractor dynamics.
    
    Integrates attractor learning into the Bifrost pipeline.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_bands: int = 8,
        n_attractors: int = 16,
    ):
        super().__init__()
        self.attractor_learner = AttractorLearningModule(
            d_model=d_model,
            n_bands=n_bands,
            n_attractors=n_attractors,
        )
    
    def forward(self, spectral: SpectralTensor) -> Tuple[List[FrequencyAttractor], torch.Tensor]:
        """Extract learned attractors from spectral representation."""
        return self.attractor_learner(spectral)
    
    def get_bridge_summary(self, attractors: List[FrequencyAttractor]) -> Dict:
        """Get summary statistics of attractor dynamics."""
        stabilities = [a.stability for a in attractors]
        coherences = [a.metadata.get("phase_coherence", 0.0) for a in attractors]
        usages = [a.metadata.get("assignment_mass", 0.0) for a in attractors]
        
        return {
            "n_attractors": len(attractors),
            "mean_stability": sum(stabilities) / len(stabilities),
            "max_stability": max(stabilities),
            "min_stability": min(stabilities),
            "mean_coherence": sum(coherences) / len(coherences),
            "active_attractors": sum(1 for u in usages if u > 0.1),
            "stability_std": torch.tensor(stabilities).std().item(),
        }
