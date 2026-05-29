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

    Uses variance ratio: phase-coherent signals should have LOWER variance
    in their attention patterns (more focused) compared to phase-randomized.

    This forces the model to learn structured attention for real signals.
    """

    def __init__(self, margin: float = 0.1, temperature: float = 0.5) -> None:
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        coherence_real: torch.Tensor,  # (B, H, T, T) from harmonic signals (pre-softmax)
        coherence_noise: torch.Tensor,  # (B, H, T, T) from phase-randomized signals
    ) -> torch.Tensor:
        """
        Compute contrastive phase loss using variance of attention.

        Strategy: Real signals should produce more focused (lower variance)
        attention patterns compared to randomized phase signals.

        Args:
            coherence_real: Coherence from phase-consistent (harmonic) signals
            coherence_noise: Coherence from phase-randomized signals

        Returns:
            Scalar loss (lower = better discrimination)
        """
        # Apply softmax to get proper attention weights
        attn_real = torch.softmax(coherence_real / self.temperature, dim=-1)
        attn_noise = torch.softmax(coherence_noise / self.temperature, dim=-1)

        # Compute variance of attention weights (lower = more focused)
        real_var = attn_real.var(dim=-1).mean()  # Scalar
        noise_var = attn_noise.var(dim=-1).mean()  # Scalar

        # Real should have LOWER variance than noise
        # Loss: penalize when real_var >= noise_var - margin
        var_diff = noise_var - real_var - self.margin

        # Hinge loss: we want noise_var - real_var > margin
        loss = F.relu(-var_diff)

        # Add regularization to prevent collapse
        entropy_real = -(attn_real * torch.log(attn_real + 1e-8)).sum(dim=-1).mean()
        entropy_noise = -(attn_noise * torch.log(attn_noise + 1e-8)).sum(dim=-1).mean()

        # Penalize uniform attention (high entropy)
        entropy_penalty = F.relu(entropy_real - 2.0) + F.relu(entropy_noise - 2.0)

        total_loss = loss + 0.1 * entropy_penalty

        return total_loss


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
