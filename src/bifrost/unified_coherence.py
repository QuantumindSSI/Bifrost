"""
Unified Coherence Metric — maps modality-specific coherence features
to a shared coherence space where cross-modal comparison is possible.

All MSC instances (CBMPC, PhaseCongruency, WaveletCoherence) produce
coherence features in different dimensions. This module projects them
to a shared space where:
    - Same-semantic samples from different modalities map nearby
    - Different-semantic samples map far apart

The projection is learned via contrastive learning on cross-modal pairs.

This module is the operationalization of Claim C3: the same coherence
principle captures semantic structure across all modalities.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedCoherenceMetric(nn.Module):
    """Maps modality-specific coherence features to a shared coherence space.

    Parameters
    ----------
    audio_dim : int
        Dimension of audio coherence features (CBMPC output).
    image_dim : int
        Dimension of image coherence features (PhaseCongruency output).
    sensor_dim : int
        Dimension of sensor coherence features (WaveletCoherence output).
    target_dim : int
        Dimension of the shared coherence space.
    hidden_dim : int
        Hidden dimension of the projection MLPs.
    """

    def __init__(
        self,
        audio_dim: int,
        image_dim: int,
        sensor_dim: int,
        target_dim: int = 256,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim

        # Modality-specific projection MLPs
        self.audio_proj = self._build_proj(audio_dim, hidden_dim, target_dim)
        self.image_proj = self._build_proj(image_dim, hidden_dim, target_dim)
        self.sensor_proj = self._build_proj(sensor_dim, hidden_dim, target_dim)

        # Modality embedding (learned modality token)
        self.modality_embed = nn.Embedding(3, target_dim)

    def _build_proj(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        """Build a 2-layer MLP projection."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        coherence_features: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Project coherence features to the shared coherence space.

        Parameters
        ----------
        coherence_features : torch.Tensor
            Modality-specific coherence features. Shape (B, modality_dim).
        modality : str
            One of "audio", "image", "sensor".

        Returns
        -------
        torch.Tensor
            Normalized coherence embedding in shared space. Shape (B, target_dim).
        """
        mod_idx = {"audio": 0, "image": 1, "sensor": 2}.get(modality)
        if mod_idx is None:
            raise ValueError(f"Unknown modality: {modality}")
        mod_token = self.modality_embed(
            torch.tensor(mod_idx, device=coherence_features.device)
        )

        if modality == "audio":
            projected = self.audio_proj(coherence_features)
        elif modality == "image":
            projected = self.image_proj(coherence_features)
        elif modality == "sensor":
            projected = self.sensor_proj(coherence_features)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Add modality token (allows the model to know which modality
        # the features came from — but in the shared space, semantically
        # similar features from different modalities should still cluster)
        projected = projected + mod_token

        # Normalize to unit hypersphere
        return F.normalize(projected, dim=-1)

    def coherence_similarity(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity in unified coherence space.

        Since features are L2-normalized, this is equivalent to dot product.
        Range: [-1, 1]. 1 = identical, 0 = orthogonal, -1 = opposite.
        """
        return F.cosine_similarity(feat_a, feat_b, dim=-1)

    def cross_modal_distance(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """Euclidean distance in unified coherence space.

        Since features are L2-normalized, this is related to cosine similarity:
        d = sqrt(2 - 2 * cos_sim)
        """
        return torch.norm(feat_a - feat_b, dim=-1)


class CrossModalCoherenceLoss(nn.Module):
    """Contrastive loss for cross-modal coherence alignment.

    Corresponding pairs (same semantic category, different modality)
    should have high coherence similarity. Non-corresponding pairs
    should have low similarity.

    This is the training objective for the UnifiedCoherenceMetric.

    Parameters
    ----------
    temperature : float
        Temperature for the contrastive loss (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        audio_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-modal contrastive loss.

        Parameters
        ----------
        audio_features : torch.Tensor
            Audio coherence embeddings in shared space. Shape (B, target_dim).
        image_features : torch.Tensor
            Image coherence embeddings in shared space. Shape (B, target_dim).
        labels : torch.Tensor
            Semantic labels. Shape (B,). Same-label pairs are positives.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        B = audio_features.shape[0]

        # Compute similarity matrix (audio x image)
        sim_matrix = torch.matmul(audio_features, image_features.T) \
                     / self.temperature  # (B, B)

        # Positive mask: same label
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # (B, B)

        # InfoNCE-style loss
        # For each audio sample, maximize similarity with same-label images
        # and minimize with different-label images
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

        # Mean log probability of positive pairs
        pos_count = pos_mask.sum(dim=1) + 1e-8
        loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()

    def forward_triplet(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """Triplet loss for coherence alignment.

        Parameters
        ----------
        anchor : torch.Tensor
            Anchor coherence embedding.
        positive : torch.Tensor
            Positive (same semantic) coherence embedding.
        negative : torch.Tensor
            Negative (different semantic) coherence embedding.
        margin : float
            Margin between positive and negative distances.
        """
        d_pos = torch.norm(anchor - positive, dim=-1)
        d_neg = torch.norm(anchor - negative, dim=-1)
        return F.relu(d_pos - d_neg + margin).mean()


class CoherenceClassifier(nn.Module):
    """Classifier on coherence features for within-modal validation.

    This is a simple linear classifier on top of coherence features,
    used to validate that coherence features carry semantic structure
    within each modality.

    Parameters
    ----------
    feature_dim : int
        Dimension of coherence features.
    n_classes : int
        Number of classes.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class UnifiedCoherenceClassifier(nn.Module):
    """Classifier on unified coherence embeddings for cross-modal validation.

    This classifier operates in the shared coherence space, allowing it
    to classify samples from any modality after projection.

    Parameters
    ----------
    unified_metric : UnifiedCoherenceMetric
        The trained unified coherence metric.
    n_classes : int
        Number of classes.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        unified_metric: UnifiedCoherenceMetric,
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.unified_metric = unified_metric
        self.classifier = nn.Sequential(
            nn.Linear(unified_metric.target_dim, unified_metric.target_dim),
            nn.LayerNorm(unified_metric.target_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(unified_metric.target_dim, n_classes),
        )

    def forward(
        self,
        coherence_features: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Classify coherence features from any modality.

        Parameters
        ----------
        coherence_features : torch.Tensor
            Modality-specific coherence features. Shape (B, modality_dim).
        modality : str
            One of "audio", "image", "sensor".

        Returns
        -------
        torch.Tensor
            Logits. Shape (B, n_classes).
        """
        unified = self.unified_metric(coherence_features, modality)
        return self.classifier(unified)
