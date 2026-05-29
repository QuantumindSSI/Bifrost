"""
ContrastivePhaseLoss — Proper contrastive objective for phase coherence learning.

This loss forces the model to discriminate between phase-coherent (harmonic)
and phase-incoherent (randomized) signals — preventing collapse mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastivePhaseLoss(nn.Module):
    """
    Contrastive loss for phase coherence discrimination.

    Maximizes the gap between phase-coherent and phase-incoherent signals:
        L = -log(σ(coherence_real - coherence_noise))

    This forces the model to output HIGHER coherence for structured signals.
    """

    def __init__(self, margin: float = 0.5, temperature: float = 1.0) -> None:
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        coherence_real: torch.Tensor,  # (B, H, T, T) from harmonic signals
        coherence_noise: torch.Tensor,  # (B, H, T, T) from phase-randomized signals
    ) -> torch.Tensor:
        """
        Compute contrastive phase loss.

        Args:
            coherence_real: Coherence from phase-consistent (harmonic) signals
            coherence_noise: Coherence from phase-randomized signals

        Returns:
            Scalar loss (lower = better discrimination)
        """
        # Mean coherence per sample (before softmax)
        real_mean = coherence_real.mean(dim=(-2, -1))  # (B, H)
        noise_mean = coherence_noise.mean(dim=(-2, -1))  # (B, H)

        # Contrastive: real should be higher than noise by margin
        # Using hinge loss formulation
        diff = real_mean - noise_mean - self.margin  # (B, H)

        # Soft hinge: smooth transition at margin boundary
        loss = F.softplus(-diff / self.temperature).mean()

        return loss


class InfoNCESpectralLoss(nn.Module):
    """
    InfoNCE-style loss for spectral coherence.

    Treats phase-coherent samples as positives, phase-randomized as negatives.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        coherence_real: torch.Tensor,  # (B, H, T, T)
        coherence_noise: torch.Tensor,  # (B, H, T, T)
    ) -> torch.Tensor:
        """
        InfoNCE loss: treat each real-noise pair as positive-negative.
        """
        B, H, T, _ = coherence_real.shape

        # Flatten coherence to (B, H, T*T) for comparison
        real_flat = coherence_real.reshape(B, H, -1).mean(dim=-1)  # (B, H)
        noise_flat = coherence_noise.reshape(B, H, -1).mean(dim=-1)  # (B, H)

        # Stack: [real, noise] for each sample
        # Shape: (B, 2, H)
        logits = torch.stack([real_flat, noise_flat], dim=1) / self.temperature

        # Labels: real is class 0 for all samples
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)

        # Cross-entropy: pushes real high, noise low
        loss = F.cross_entropy(logits.mean(dim=-1), labels)

        return loss


def compute_coherence_discrimination_gap(
    coherence_real: torch.Tensor,
    coherence_noise: torch.Tensor,
) -> dict[str, float]:
    """
    Compute discrimination metrics between real and noise coherence.

    Returns dict with:
        - real_mean: Mean coherence for real signals
        - noise_mean: Mean coherence for noise signals
        - gap: Difference (should be positive)
        - ratio: real_mean / noise_mean (should be > 1.0)
    """
    real_mean = coherence_real.mean().item()
    noise_mean = coherence_noise.mean().item()
    gap = real_mean - noise_mean
    ratio = real_mean / (noise_mean + 1e-8)

    return {
        "real_mean": real_mean,
        "noise_mean": noise_mean,
        "gap": gap,
        "ratio": ratio,
    }
