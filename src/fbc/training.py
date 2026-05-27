"""
FBCTrainer — basic training infrastructure for phase coherence learning.

Self-supervised next-frame prediction enables the dual-stream SSM
to learn temporal phase coherence patterns.

Usage:
    from fbc.training import FBCTrainer
    from fbc.pipeline import FBCPipeline

    pipeline = FBCPipeline(d_model=128, use_mamba=True)
    trainer = FBCTrainer(pipeline, lr=1e-3)

    # Train on audio sequences
    for epoch in range(100):
        for batch in dataloader:
            loss = trainer.train_step(batch)
        print(f"Epoch {epoch}: loss={loss:.4f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .spectral_tensor import SpectralTensor
from .pipeline import FBCPipeline


class NextFramePredictionLoss(nn.Module):
    """
    Self-supervised loss: predict frame t+1 from frame t.

    MSE on both amplitude and phase encourages the SSM to learn
    temporal coherence patterns. Phase coherence emerges as the
    phase SSM learns to predict consistent phase progressions.
    """

    def __init__(self, phase_weight: float = 0.5) -> None:
        super().__init__()
        self.phase_weight = phase_weight

    def forward(
        self,
        pred: SpectralTensor,
        target: SpectralTensor,
    ) -> torch.Tensor:
        """
        Compute next-frame prediction loss.

        Args:
            pred: Predicted SpectralTensor (frame t+1)
            target: Target SpectralTensor (actual frame t+1)

        Returns:
            Combined MSE loss on amplitude and phase
        """
        # Amplitude loss (standard MSE)
        amp_loss = F.mse_loss(pred.amplitude, target.amplitude)

        # Phase loss (circular MSE - handle phase wraparound)
        phase_diff = pred.phase - target.phase
        # Wrap to [-π, π]
        phase_diff = torch.atan2(phase_diff.sin(), phase_diff.cos())
        phase_loss = (phase_diff ** 2).mean()

        return amp_loss + self.phase_weight * phase_loss


class FBCTrainer:
    """
    Basic trainer for FBC pipeline with phase coherence learning.

    Implements next-frame prediction training with Adam optimizer,
    gradient clipping, and warmup learning rate schedule.

    Parameters
    ----------
    pipeline : FBCPipeline
        The FBC pipeline to train
    lr : float
        Learning rate (default: 1e-3)
    weight_decay : float
        Weight decay for regularization (default: 0.01)
    grad_clip : float
        Gradient clipping threshold (default: 1.0)
    warmup_steps : int
        Number of warmup steps for LR schedule (default: 1000)
    device : str
        Device to train on (default: auto-detect)
    """

    def __init__(
        self,
        pipeline: FBCPipeline,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        device: Optional[str] = None,
    ) -> None:
        self.pipeline = pipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)

        self.lr = lr
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps

        # Criterion: next-frame prediction
        self.criterion = NextFramePredictionLoss(phase_weight=0.5)

        # Optimizer: Adam with weight decay
        self.optimizer = Adam(
            pipeline.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

        # Learning rate warmup
        self.scheduler = self._create_scheduler()

        self.step_count = 0
        self.loss_history: List[float] = []

    def _create_scheduler(self) -> LambdaLR:
        """Create warmup learning rate scheduler."""
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0
        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Single training step with next-frame prediction.

        Args:
            signal: Audio tensor (B, T, n_freq) or (B, L) time-domain
            metadata: Optional metadata dict

        Returns:
            Loss value (float)
        """
        self.pipeline.train()
        signal = signal.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass through pipeline
        bound_st, coherence = self.pipeline(signal, metadata)

        # Next-frame prediction loss
        # Use all frames except last as input, all frames except first as target
        B, T, D = bound_st.amplitude.shape

        if T < 2:
            # Single frame - can't do next-frame prediction
            # Fall back to reconstruction loss against input
            target = bound_st
        else:
            # Create shifted target for next-frame prediction
            # Current implementation: simple shift
            target_amplitude = torch.cat([
                bound_st.amplitude[:, 1:, :],
                bound_st.amplitude[:, -1:, :]  # Repeat last frame
            ], dim=1)
            target_phase = torch.cat([
                bound_st.phase[:, 1:, :],
                bound_st.phase[:, -1:, :]
            ], dim=1)
            target = SpectralTensor(
                amplitude=target_amplitude.detach(),
                phase=target_phase.detach(),
                scale=bound_st.scale,
                uncertainty=bound_st.uncertainty,
            )

        loss = self.criterion(bound_st, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.pipeline.parameters(),
            self.grad_clip,
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        self.step_count += 1
        loss_val = loss.item()
        self.loss_history.append(loss_val)

        return loss_val

    def eval_step(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluation step (no gradient computation).

        Returns:
            Dict with loss and coherence statistics
        """
        self.pipeline.eval()
        signal = signal.to(self.device)

        with torch.no_grad():
            bound_st, coherence = self.pipeline(signal, metadata)

            # Next-frame prediction loss
            B, T, D = bound_st.amplitude.shape
            if T < 2:
                target = bound_st
            else:
                target_amplitude = torch.cat([
                    bound_st.amplitude[:, 1:, :],
                    bound_st.amplitude[:, -1:, :]
                ], dim=1)
                target_phase = torch.cat([
                    bound_st.phase[:, 1:, :],
                    bound_st.phase[:, -1:, :]
                ], dim=1)
                target = SpectralTensor(
                    amplitude=target_amplitude,
                    phase=target_phase,
                    scale=bound_st.scale,
                    uncertainty=bound_st.uncertainty,
                )

            loss = self.criterion(bound_st, target)

            # Coherence statistics
            coherence_stats = {
                "coherence_mean": coherence.mean().item(),
                "coherence_std": coherence.std().item(),
                "coherence_max": coherence.max().item(),
            }

            # Per-head diagonal ratio (self-attention strength)
            for h in range(coherence.shape[1]):
                head_coh = coherence[0, h]  # (T, T)
                diag = head_coh.diagonal().mean().item()
                offdiag = head_coh[~torch.eye(head_coh.shape[0], dtype=torch.bool)].mean().item()
                coherence_stats[f"head_{h}_ratio"] = diag / offdiag if offdiag > 0 else 1.0

        return {
            "loss": loss.item(),
            **coherence_stats,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "pipeline_state": self.pipeline.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "step_count": self.step_count,
            "loss_history": self.loss_history,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.pipeline.load_state_dict(checkpoint["pipeline_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.step_count = checkpoint["step_count"]
        self.loss_history = checkpoint["loss_history"]


def train_fbc_simple(
    pipeline: FBCPipeline,
    dataloader: Any,
    epochs: int = 100,
    device: Optional[str] = None,
) -> FBCTrainer:
    """
    Simple training loop for FBC pipeline.

    Args:
        pipeline: FBCPipeline instance
        dataloader: PyTorch DataLoader with audio sequences
        epochs: Number of training epochs
        device: Device to train on

    Returns:
        Trained FBCTrainer instance
    """
    trainer = FBCTrainer(pipeline, device=device)

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                signal, metadata = batch[0], batch[1] if len(batch) > 1 else None
            else:
                signal, metadata = batch, None

            loss = trainer.train_step(signal, metadata)
            epoch_losses.append(loss)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: loss={loss:.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch}/{epochs}: avg_loss={avg_loss:.4f}")

        # Evaluate every 10 epochs
        if epoch % 10 == 0 and len(dataloader) > 0:
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                signal, metadata = batch[0], batch[1] if len(batch) > 1 else None
            else:
                signal, metadata = batch, None

            stats = trainer.eval_step(signal, metadata)
            print(f"  Eval: loss={stats['loss']:.4f}")
            print(f"  Coherence: mean={stats['coherence_mean']:.4f}, std={stats['coherence_std']:.4f}")
            for h in range(4):
                print(f"  Head {h} ratio: {stats.get(f'head_{h}_ratio', 0):.3f}")

    return trainer
