"""
BifröstTrainer — basic training infrastructure for phase coherence learning.

Self-supervised next-frame prediction enables the dual-stream SSM
to learn temporal phase coherence patterns.

Usage:
    from bifrost.training import FBCTrainer, BifrostTrainer
    from bifrost.pipeline import FBCPipeline, BifrostPipeline

    pipeline = BifrostPipeline(d_model=128, use_mamba=True)
    trainer = BifrostTrainer(pipeline, lr=1e-3)

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
import math as _math
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .spectral_tensor import SpectralTensor
from .pipeline import FBCPipeline


class ContrastiveCoherenceLoss(nn.Module):
    """
    Contrastive phase coherence loss.

    Trains the pipeline to produce HIGH coherence for the input signal
    and LOW coherence for a phase-randomised version of the same signal.
    This is a non-collapsible objective: the model cannot drive both
    terms to zero simultaneously.

    Loss = -log(sigma(coh_real - coh_noise - margin))

    where coh_real = mean coherence on real harmonic signal,
          coh_noise = mean coherence on phase-randomised noise,
          margin = target separation (default 0.1).

    Parameters
    ----------
    margin : float
        Minimum required coherence gap between real and noise (default 0.1).
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        coh_real: torch.Tensor,
        coh_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive coherence loss.

        Coherence matrices are softmax outputs — their mean is always 1/T
        (rows sum to 1), so mean() gives zero gradient regardless of structure.

        Instead, we maximise the VARIANCE of coh_real (structured attention has
        high variance — peaked at coherent pairs) and minimise the variance of
        coh_noise (white noise should produce near-uniform attention, low variance).

        Args:
            coh_real:  (B, H, T, T) softmax coherence weights from real signal.
            coh_noise: (B, H, T, T) softmax coherence weights from noise signal.

        Returns:
            Scalar loss. Minimising this maximises var(coh_real) - var(coh_noise).
        """
        # Maximise variance of real coherence (structured = peaked attention).
        # Minimise variance of noise coherence (noise = uniform attention).
        # Loss = -var_real + var_noise + margin.
        # Always has nonzero gradient through -var_real, even when already satisfied.
        # This prevents dead gradients when the margin condition is met.
        var_real = coh_real.var()
        # Detach var_noise: it is a fixed reference baseline, not a gradient target.
        # Without detach, the optimizer minimises loss by growing var_noise (making
        # noise attention structured) instead of growing var_real (making harmonic
        # attention structured). Only var_real should receive gradients.
        var_noise = coh_noise.var().detach()
        loss = -var_real + var_noise + self.margin
        return loss


class NextFramePredictionLoss(nn.Module):
    """
    MSE next-frame prediction loss (kept for API compatibility).

    Note: this loss can collapse to zero on periodic signals. Prefer
    ContrastiveCoherenceLoss for phase coherence training.
    """

    def __init__(self, phase_weight: float = 0.5) -> None:
        super().__init__()
        self.phase_weight = phase_weight

    def forward(
        self,
        pred: SpectralTensor,
        target: SpectralTensor,
    ) -> torch.Tensor:
        amp_loss = F.mse_loss(pred.amplitude, target.amplitude)
        phase_diff = torch.atan2(
            (pred.phase - target.phase).sin(),
            (pred.phase - target.phase).cos(),
        )
        phase_loss = (phase_diff ** 2).mean()
        return amp_loss + self.phase_weight * phase_loss


class FBCTrainer:
    """
    Basic trainer for Bifröst pipeline with phase coherence learning.

    Implements next-frame prediction training with Adam optimizer,
    gradient clipping, and warmup learning rate schedule.

    Parameters
    ----------
    pipeline : FBCPipeline
        The Bifröst pipeline to train
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

        # Criterion: contrastive coherence — variance of real > variance of noise.
        # margin=1e-4 is in variance units (softmax variance ~1e-5 to 1e-3 range).
        self.criterion = ContrastiveCoherenceLoss(margin=1e-4)

        # Freeze tau and band_weights in ResonanceAttention.
        # These control softmax sharpness globally — when learned they equalise
        # attention variance across ALL inputs (real and noise), collapsing the gap.
        # Only SSM weights, projections, and binding parameters should be trained.
        frozen_names = {"tau", "band_weights"}
        trainable_params = [
            p for name, p in pipeline.named_parameters()
            if not any(name.endswith(f".{fn}") or name == fn for fn in frozen_names)
        ]
        for name, p in pipeline.named_parameters():
            if any(name.endswith(f".{fn}") or name == fn for fn in frozen_names):
                p.requires_grad_(False)

        # Optimizer: Adam with weight decay
        self.optimizer = Adam(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

        # Learning rate warmup
        self.scheduler = self._create_scheduler()

        self.step_count = 0
        self.loss_history: List[float] = []

    def _create_scheduler(self) -> LambdaLR:
        """Create warmup + cosine decay scheduler.

        Warmup for warmup_steps, then cosine decay over 50x that many steps.
        Long decay keeps gradients active well past the plateau region.
        """
        total_steps = self.warmup_steps * 50

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return 0.01 + 0.99 * 0.5 * (1.0 + _math.cos(_math.pi * min(progress, 1.0)))

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
        if signal.dim() < 2:
            raise ValueError(
                f"train_step expects signal with dim >= 2, got dim={signal.dim()}. "
                "Pass (B, L) time-domain or (B, T, n_freq) spectral input."
            )
        self.pipeline.train()
        signal = signal.to(self.device)

        self.optimizer.zero_grad()

        # --- Positive: run real signal through pipeline ---
        _, coh_real = self.pipeline(signal, metadata)

        # --- Negative: white noise with same RMS as the real signal ---
        # Phase-randomised signals share the same amplitude spectrum as the original
        # and are indistinguishable to an STFT-based canonicalizer (S0 throws phase away
        # during amplitude normalisation, then recomputes phase from the STFT).
        # White noise has a FLAT amplitude spectrum — no harmonic peaks — so it produces
        # a fundamentally different SpectralTensor that the SSM can learn to distinguish.
        rms = signal.std(dim=-1, keepdim=True).clamp(min=1e-8)
        if signal.dim() == 2:
            noise_signal = torch.randn_like(signal) * rms
        else:
            noise_signal = torch.randn_like(signal) * rms

        with torch.no_grad():
            self.pipeline.eval()
            _, coh_noise = self.pipeline(noise_signal.detach(), metadata)
        # Restore full train mode on every submodule explicitly
        self.pipeline.train()
        for module in self.pipeline.modules():
            module.training = True

        loss = self.criterion(coh_real, coh_noise)

        # NaN guard: skip step if loss is NaN (prevents gradient corruption)
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            self.step_count += 1
            self.loss_history.append(float('nan'))
            return float('nan')

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.pipeline.parameters(),
            self.grad_clip,
        )

        # Skip optimizer step if gradients exploded (secondary NaN guard)
        if torch.isfinite(grad_norm):
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
    Simple training loop for Bifröst pipeline.

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
