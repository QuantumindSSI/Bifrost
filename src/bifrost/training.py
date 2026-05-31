"""
BifröstTrainer — basic training infrastructure for phase coherence learning.

Self-supervised next-frame prediction enables the dual-stream SSM
to learn temporal phase coherence patterns.

Usage:
    from bifrost.training import BifrostTrainer, BifrostTrainer
    from bifrost.pipeline import BifrostPipeline, BifrostPipeline

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
from .pipeline import BifrostPipeline


class ContrastiveCoherenceLoss(nn.Module):
    """
    Contrastive ratio loss: maximise var_real / var_noise.

    Trains the pipeline to produce higher output variance for harmonic
    signals (structured coherence) than for phase-randomised noise
    (uniform coherence). Uses a ratio formulation to prevent the model
    from increasing both variances together, which collapses the gap.

    Loss = log(var_noise) - log(var_real) = -log(var_real / var_noise)

    Minimising this maximises the variance ratio. The noise variance is
    detached (fixed baseline), so gradients flow only to increase real
    variance relative to noise.

    Parameters
    ----------
    margin : float
        Unused (kept for API compatibility).
    """

    def __init__(self, margin: float = 0.0) -> None:
        """
        Initialize contrastive coherence loss.

        Parameters
        ----------
        margin : float
            Unused (kept for API compatibility).

        Complexity
        ----------
        O(1) - initialization only.

        Side Effects
        ------------
        None.
        """
        super().__init__()
        # Margin kept for API compatibility; not used in ratio loss.

    def forward(
        self,
        feat_real: torch.Tensor,
        feat_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive coherence loss on output feature amplitudes.

        When coherence is peaked (structured real signal), the attention-weighted
        aggregation in W_v produces high-variance output features — similar frames
        reinforce each other. When coherence is uniform (noise), aggregation averages
        everything and produces low-variance output.

        Using feat.amplitude.var() instead of coherence_weights.var():
        - Has a real gradient path: W_v → amp_proj → loss
        - Is not destroyed by softmax compression (75000:1 dynamic range loss)
        - The canonical STFT phase controls coherence; W_v learns what to route

        Args
        ----
        feat_real : torch.Tensor or SpectralTensor
            (B, T, D) amplitude from real signal. Must be finite.
        feat_noise : torch.Tensor or SpectralTensor
            (B, T, D) amplitude from noise signal. Must be finite.

        Returns
        -------
        torch.Tensor
            Scalar loss. Minimising this maximises var(feat_real) - var(feat_noise).

        Raises
        ------
        ValueError
            If inputs contain NaN or Inf values.

        Complexity
        ----------
        O(N) - where N is total number of elements in input tensors.

        Side Effects
        ------------
        None.
        """
        amp_real = feat_real.amplitude if hasattr(feat_real, 'amplitude') else feat_real
        amp_noise = feat_noise.amplitude if hasattr(feat_noise, 'amplitude') else feat_noise
        if not torch.isfinite(amp_real).all():
            raise ValueError("feat_real contains NaN or Inf values")
        if not torch.isfinite(amp_noise).all():
            raise ValueError("feat_noise contains NaN or Inf values")
        var_real = amp_real.var()
        var_noise = amp_noise.var().detach()
        # Normalised gap loss: maximise var_real / var_noise, not var_real alone.
        # Raw variance loss (-var_real + var_noise) allows both to increase together,
        # collapsing the gap. Ratio loss forces discrimination: increasing var_real
        # while var_noise stays fixed maximises the ratio.
        # Loss = -log(var_real / var_noise) = log(var_noise) - log(var_real).
        # Minimising this maximises the ratio. Epsilon prevents log(0).
        eps = 1e-8
        loss = (var_noise + eps).log() - (var_real + eps).log()
        return loss


class NextFramePredictionLoss(nn.Module):
    """
    MSE next-frame prediction loss (kept for API compatibility).

    Note: this loss can collapse to zero on periodic signals. Prefer
    ContrastiveCoherenceLoss for phase coherence training.
    """

    def __init__(self, phase_weight: float = 0.5) -> None:
        """
        Initialize next-frame prediction loss.

        Parameters
        ----------
        phase_weight : float
            Weight for phase loss component. Must be in [0, 1].

        Raises
        ------
        ValueError
            If phase_weight not in [0, 1].

        Complexity
        ----------
        O(1) - initialization only.

        Side Effects
        ------------
        None.
        """
        if not (0.0 <= phase_weight <= 1.0):
            raise ValueError(f"phase_weight must be in [0, 1], got {phase_weight}")
        super().__init__()
        self.phase_weight = phase_weight

    def forward(
        self,
        pred: SpectralTensor,
        target: SpectralTensor,
    ) -> torch.Tensor:
        """
        Compute next-frame prediction loss.

        Args
        ----
        pred : SpectralTensor
            Predicted spectral tensor. Must be finite.
        target : SpectralTensor
            Target spectral tensor. Must be finite.

        Returns
        -------
        torch.Tensor
            Combined amplitude and phase loss.

        Raises
        ------
        ValueError
            If inputs contain NaN or Inf values.

        Complexity
        ----------
        O(N) - where N is total number of elements in input tensors.

        Side Effects
        ------------
        None.
        """
        if not torch.isfinite(pred.amplitude).all():
            raise ValueError("pred.amplitude contains NaN or Inf values")
        if not torch.isfinite(pred.phase).all():
            raise ValueError("pred.phase contains NaN or Inf values")
        if not torch.isfinite(target.amplitude).all():
            raise ValueError("target.amplitude contains NaN or Inf values")
        if not torch.isfinite(target.phase).all():
            raise ValueError("target.phase contains NaN or Inf values")
        amp_loss = F.mse_loss(pred.amplitude, target.amplitude)
        phase_diff = torch.atan2(
            (pred.phase - target.phase).sin(),
            (pred.phase - target.phase).cos(),
        )
        phase_loss = (phase_diff ** 2).mean()
        return amp_loss + self.phase_weight * phase_loss


class BifrostTrainer:
    """
    Basic trainer for Bifröst pipeline with phase coherence learning.

    Implements next-frame prediction training with Adam optimizer,
    gradient clipping, and warmup learning rate schedule.

    Parameters
    ----------
    pipeline : BifrostPipeline
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
        pipeline: BifrostPipeline,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize Bifrost trainer.

        Parameters
        ----------
        pipeline : BifrostPipeline
            The Bifröst pipeline to train.
        lr : float
            Learning rate. Must be > 0.
        weight_decay : float
            Weight decay for regularization. Must be >= 0.
        grad_clip : float
            Gradient clipping threshold. Must be > 0.
        warmup_steps : int
            Number of warmup steps for LR schedule. Must be >= 0.
        device : str, optional
            Device to train on. Defaults to auto-detect.

        Raises
        ------
        ValueError
            If lr <= 0, weight_decay < 0, grad_clip <= 0, or warmup_steps < 0.

        Complexity
        ----------
        O(1) - initialization only.

        Side Effects
        ------------
        Moves pipeline to device, initializes optimizer and scheduler.
        """
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {grad_clip}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.pipeline = pipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)

        self.lr = lr
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps

        # Criterion: contrastive ratio loss on output feature amplitudes.
        # Maximises var_real / var_noise, preventing joint variance increase.
        self.criterion = ContrastiveCoherenceLoss()

        # Freeze band_weights only. tau is now UNFROZEN.
        # With parameter-free coherence (from canonical STFT phase), tau cannot collapse
        # coherence to uniform — the phase differences are fixed by S0, not by tau.
        # tau now provides the only gradient path: weights = softmax(precomp_coh / tau)
        # carries grad_fn via tau, so var_real = weights.var() has a gradient.
        # band_weights stays frozen: it controls multi-band weighting and is irrelevant
        # in the precomputed-coherence path.
        frozen_names = {"band_weights"}
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
        """Create linear warmup scheduler with constant LR after warmup.

        Cosine decay was removed: tau and band_weights are frozen, so there is
        no tau-overshoot risk. The decay was killing gradients past epoch 33,
        causing attention variance to collapse toward uniform even though the
        gap remained positive. Constant LR after warmup keeps gradients active
        for the full 200 epochs.
        """
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
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
        if signal.dim() < 2:
            raise ValueError(
                f"train_step expects signal with dim >= 2, got dim={signal.dim()}. "
                "Pass (B, L) time-domain or (B, T, n_freq) spectral input."
            )
        self.pipeline.train()
        signal = signal.to(self.device)

        self.optimizer.zero_grad()

        # --- Positive: run real signal through pipeline ---
        bound_real, _ = self.pipeline(signal, metadata)
        
        # Get real signal's canonical representation for negative sample construction
        with torch.no_grad():
            canonical_real = self.pipeline.canonicalizer(signal, metadata)

        # --- Negative: STFT-domain phase randomisation ---
        # Goal: identical per-frame amplitude spectrum, destroyed inter-frame phase coherence.
        # White noise is too spectrally different (flat vs harmonic peaks) — W_q/W_k learn
        # amplitude shape discrimination, not phase coherence. Temporal shuffling destroys
        # frame structure entirely, making the task trivial then forgettable.
        #
        # STFT phase randomisation keeps |STFT(x)[f,t]| identical to the original but
        # assigns a uniform-random phase to each (freq, frame) bin independently.
        # The reconstructed waveform has the same per-frame spectral envelope but no
        # consistent phase relationship across frames — forcing the SSM to detect
        # inter-frame phase coherence, which is the actual training objective.
        if signal.dim() == 2:
            # (B, L) time-domain: STFT phase randomisation — same amplitude spectrum,
            # destroyed inter-frame phase coherence. Forces SSM to detect phase structure.
            n_fft = 1024
            hop = n_fft // 4
            B, L = signal.shape
            win = torch.hann_window(n_fft, device=signal.device)
            spec = torch.stft(
                signal,
                n_fft=n_fft,
                hop_length=hop,
                return_complex=True,
                window=win,
                pad_mode='reflect',
            )  # (B, F, T)
            rand_phase = torch.rand_like(spec.real) * 2.0 * _math.pi
            noise_spec = torch.polar(spec.abs(), rand_phase)
            noise_signal = torch.istft(
                noise_spec,
                n_fft=n_fft,
                hop_length=hop,
                window=win,
                length=L,
            )
        else:
            # (B, T, D) spectral input: RMS-matched white noise fallback
            rms = signal.std(dim=-1, keepdim=True).clamp(min=1e-8)
            noise_signal = torch.randn_like(signal) * rms

        # --- Negative: same decomposed amplitude, destroyed phase coherence ---
        # The key insight: W_v learns amplitude shortcuts if decomposed amplitude
        # differs between positive and negative. We must keep decomposed amplitude
        # identical, varying ONLY the coherence pattern (via canonical phase).
        #
        # Implementation: run canonicalizer on phase-randomized signal to get
        # destroyed phase, but use the REAL signal's decomposed amplitude.
        with torch.no_grad():
            self.pipeline.eval()
            # Get phase-randomized canonical representation
            canonical_rand = self.pipeline.canonicalizer(noise_signal.detach(), metadata)
            # Get real signal's decomposed representation
            if self.pipeline.use_complex_ssm:
                decomposed_real_neg, _ = self.pipeline.decomposer(canonical_real, None)
            else:
                decomposed_real_neg = self.pipeline.decomposer(canonical_real)
            # Bind with real amplitude but random phase → destroyed coherence
            bound_noise, _ = self.pipeline.binding(
                decomposed_real_neg,
                input_proj=self.pipeline._decomp_to_bind_proj,
                canonical_phase=canonical_rand.phase,
            )
        # Restore full train mode on every submodule explicitly
        self.pipeline.train()
        for module in self.pipeline.modules():
            module.training = True

        loss = self.criterion(bound_real, bound_noise)

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
    pipeline: BifrostPipeline,
    dataloader: Any,
    epochs: int = 100,
    device: Optional[str] = None,
) -> BifrostTrainer:
    """
    Simple training loop for Bifröst pipeline.

    Args:
        pipeline: BifrostPipeline instance
        dataloader: PyTorch DataLoader with audio sequences
        epochs: Number of training epochs
        device: Device to train on

    Returns:
        Trained BifrostTrainer instance
    """
    trainer = BifrostTrainer(pipeline, device=device)

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
