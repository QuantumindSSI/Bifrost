"""
ComplexBifrostTrainer — Training infrastructure for ComplexSpectralDecomposer.

Implements the correct training objective for phase coherence learning:
- Complex-valued next-step prediction: |pred_z - target_z|^2
- Coherence metrics: diagonal attention ratio from complex state evolution
- Phase gradient analysis across time frames

This replaces the broken training that used independent per-frame phase computation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .spectral_tensor import SpectralTensor
from .decomposer.complex_decomposer import ComplexSpectralDecomposer
from .resonance_attention import SpectralBinding


class ComplexNextStepLoss(nn.Module):
    """
    Complex-valued next-step prediction loss.

    Measures the squared magnitude of complex prediction error:
        L = |pred_z - target_z|^2
          = (pred_a - target_a)^2 + (pred_a * pred_φ - target_a * target_φ)^2

    This is the correct objective for training complex SSM because:
    - It jointly optimizes amplitude and phase predictions
    - Phase errors are weighted by amplitude (important for low-energy bins)
    - Temporal coherence emerges naturally from the complex state transitions
    """

    def __init__(self, amplitude_weight: float = 1.0, phase_weight: float = 1.0) -> None:
        super().__init__()
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight

    def forward(
        self,
        pred_z: torch.Tensor,  # (B, T, d_model) complex
        target_z: torch.Tensor,  # (B, T, d_model) complex
    ) -> torch.Tensor:
        """
        Compute complex prediction loss.

        Args:
            pred_z: Predicted complex spectra
            target_z: Target complex spectra (from next time step)

        Returns:
            Scalar loss
        """
        # Check for NaN/Inf in inputs
        if not torch.isfinite(pred_z).all() or not torch.isfinite(target_z).all():
            # Return a large but finite loss to signal numerical issues
            return torch.tensor(1e6, device=pred_z.device, requires_grad=True)
        
        # Complex squared error: |pred - target|^2
        # For complex: |a + ib|^2 = a^2 + b^2
        diff = pred_z - target_z
        # Clamp to prevent overflow
        diff_real = diff.real.clamp(-10, 10)
        diff_imag = diff.imag.clamp(-10, 10)
        complex_error = (diff_real ** 2 + diff_imag ** 2).mean()

        # Also compute separate amplitude and phase errors for monitoring
        pred_amp = pred_z.abs().clamp(0, 10)  # Prevent extreme values
        target_amp = target_z.abs().clamp(0, 10)
        amp_error = F.mse_loss(pred_amp, target_amp)

        pred_phase = pred_z.angle()
        target_phase = target_z.angle()
        # Circular phase difference
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(phase_diff.sin(), phase_diff.cos())
        phase_error = (phase_diff ** 2).mean()

        # complex_error = |pred - target|² already penalises both amplitude and phase jointly.
        # Adding amp_error again would double-penalise amplitude and bias gradients toward
        # amplitude at the expense of phase learning. Only the circular phase error is added
        # as a separately weighted term since it captures wrap-around structure.
        total_loss = complex_error + self.phase_weight * phase_error
        
        # Final clamp to ensure finite output
        total_loss = torch.clamp(total_loss, 0, 1e6)

        return total_loss


class PhaseCoherenceMetrics:
    """
    Metrics for verifying phase coherence learning in complex SSM.

    Measures:
    - Diagonal coherence ratio: attention diagonal vs off-diagonal
    - Phase gradient consistency: temporal phase evolution smoothness
    - Complex state correlation: state[t] vs state[t+1] correlation
    """

    @staticmethod
    def diagonal_coherence_ratio(coherence: torch.Tensor) -> float:
        """
        Compute diagonal-to-off-diagonal coherence ratio.

        Args:
            coherence: (B, H, T, T) attention coherence weights (softmax output, all positive)

        Returns:
            Ratio of diagonal mean to off-diagonal mean (should be > 1.0 for coherent phase)
        """
        B, H, T, _ = coherence.shape

        # coherence is already softmax-normalised by ResonanceAttention;
        # re-applying softmax would compress the distribution toward uniform
        # and make the ratio artificially close to 1.0, masking real structure.

        # Extract diagonal (temporal self-coherence)
        diagonal = coherence.diagonal(dim1=-2, dim2=-1)  # (B, H, T)
        diag_mean = diagonal.mean().item()

        # Reshape to (B*H, T, T) so the 2D off-diagonal mask indexes correctly.
        # coherence[..., mask] on a 4D tensor with a 2D mask uses advanced indexing
        # that selects across the wrong dimensions and produces NaN.
        coh_flat = coherence.reshape(B * H, T, T)
        off_diag_mask = ~torch.eye(T, dtype=torch.bool, device=coherence.device)  # (T, T)
        off_diag = coh_flat[:, off_diag_mask]  # (B*H, T*(T-1))
        off_diag_mean = off_diag.mean().item()

        # Ratio: diagonal should be higher than off-diagonal for coherent attention
        ratio = (diag_mean + 1e-8) / (off_diag_mean + 1e-8)
        return ratio

    @staticmethod
    def phase_gradient_smoothness(phase: torch.Tensor) -> float:
        """
        Measure temporal smoothness of phase evolution.

        Args:
            phase: (B, T, D) phase values

        Returns:
            Inverse of mean phase gradient (higher = smoother)
        """
        # Compute phase differences between consecutive frames
        phase_diff = phase[:, 1:, :] - phase[:, :-1, :]  # (B, T-1, D)
        # Wrap to [-π, π]
        phase_diff = torch.atan2(phase_diff.sin(), phase_diff.cos())
        # Mean absolute gradient
        smoothness = 1.0 / (phase_diff.abs().mean().item() + 1e-8)
        return smoothness

    @staticmethod
    def complex_state_correlation(z_t: torch.Tensor, z_t_plus_1: torch.Tensor) -> float:
        """
        Compute correlation between consecutive complex states.

        Args:
            z_t: Complex state at time t
            z_t_plus_1: Complex state at time t+1

        Returns:
            Normalized correlation coefficient
        """
        # Flatten spatial dimensions
        z_t_flat = z_t.reshape(-1)
        z_t1_flat = z_t_plus_1.reshape(-1)

        # Complex correlation: E[z_t * conj(z_{t+1})]
        correlation = (z_t_flat * z_t1_flat.conj()).mean()
        # Normalize
        norm = (z_t_flat.abs().pow(2).mean() * z_t1_flat.abs().pow(2).mean()).sqrt()
        normalized_corr = (correlation.abs() / (norm + 1e-8)).item()

        return normalized_corr


class ComplexBifrostTrainer:
    """
    Trainer for ComplexSpectralDecomposer with proper complex-valued objectives.

    Key improvements over BifrostTrainer:
    1. Complex next-step prediction loss (not independent phase MSE)
    2. Phase coherence metrics during training
    3. Verification that diagonal attention patterns emerge
    """

    def __init__(
        self,
        decomposer: ComplexSpectralDecomposer,
        binding: Optional[SpectralBinding] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        amplitude_weight: float = 1.0,
        phase_weight: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.decomposer = decomposer.to(device)
        self.binding = binding.to(device) if binding else None
        self.device = device
        self.n_freq = decomposer.n_freq
        self.d_model = decomposer.d_model

        # Projection layer: d_model -> n_freq for loss computation
        # This handles the dimension mismatch between decomposer output and target
        if self.d_model != self.n_freq:
            self.output_proj = nn.Linear(self.d_model, self.n_freq).to(device)
        else:
            self.output_proj = None

        # Complex-valued loss
        self.criterion = ComplexNextStepLoss(amplitude_weight, phase_weight)

        # Optimizer (decomposer + projection if needed)
        params = list(decomposer.parameters())
        if self.output_proj:
            params.extend(self.output_proj.parameters())
        if binding:
            params.extend(binding.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.metrics_history: Dict[str, list] = {
            "loss": [],
            "coherence_ratio": [],
            "phase_smoothness": [],
            "complex_correlation": [],
        }

    def train_step(
        self,
        spectral_batch: SpectralTensor,
    ) -> Dict[str, float]:
        """
        Single training step with complex-valued loss.

        Args:
            spectral_batch: Batch of SpectralTensor with temporal structure

        Returns:
            Dictionary of metrics
        """
        self.decomposer.train()
        if self.binding:
            self.binding.train()

        # Forward pass — ComplexSpectralDecomposer returns (SpectralTensor, h_T)
        decomp_out = self.decomposer(spectral_batch)
        decomposed = decomp_out[0] if isinstance(decomp_out, tuple) else decomp_out

        # Get complex representations for loss computation
        # Target: next frame prediction
        z_input = spectral_batch.complex_spectrum()  # (B, n_freq) or (B, T, n_freq)

        if z_input.dim() == 2:
            # Add temporal dimension if missing
            z_input = z_input.unsqueeze(1)  # (B, 1, n_freq)

        # Ensure we have at least 2 frames for prediction
        if z_input.shape[1] < 2:
            # Replicate to create minimum temporal structure
            z_input = z_input.repeat(1, 2, 1)

        # Shift to create prediction target: predict frame t+1 from frame t
        pred_z = z_input[:, :-1, :]  # (B, T-1, n_freq)
        target_z = z_input[:, 1:, :]  # (B, T-1, n_freq)

        # Interpolate to match decomposer output dimension if needed
        if pred_z.shape[-1] != decomposed.amplitude.shape[-1] and pred_z.shape[1] > 0:
            # Complex interpolation: handle real and imag separately
            # F.interpolate needs 3D: (B, C, L) where C = n_freq, L = T
            pred_z_real = F.interpolate(
                pred_z.real.transpose(-2, -1),
                size=decomposed.amplitude.shape[-1],
                mode='linear',
                align_corners=True
            ).transpose(-2, -1)
            pred_z_imag = F.interpolate(
                pred_z.imag.transpose(-2, -1),
                size=decomposed.amplitude.shape[-1],
                mode='linear',
                align_corners=True
            ).transpose(-2, -1)
            pred_z = torch.complex(pred_z_real, pred_z_imag)

            target_z_real = F.interpolate(
                target_z.real.transpose(-2, -1),
                size=decomposed.amplitude.shape[-1],
                mode='linear',
                align_corners=True
            ).transpose(-2, -1)
            target_z_imag = F.interpolate(
                target_z.imag.transpose(-2, -1),
                size=decomposed.amplitude.shape[-1],
                mode='linear',
                align_corners=True
            ).transpose(-2, -1)
            target_z = torch.complex(target_z_real, target_z_imag)

        # Convert decomposed output to complex
        pred_z_complex = torch.complex(
            decomposed.amplitude[:, :-1, :],
            decomposed.phase[:, :-1, :]
        )

        # Project to n_freq if needed (d_model -> n_freq)
        if self.output_proj:
            # Project real and imaginary parts separately
            pred_real = self.output_proj(pred_z_complex.real)
            pred_imag = self.output_proj(pred_z_complex.imag)
            pred_z_complex = torch.complex(pred_real, pred_imag)

        # Complex loss
        loss = self.criterion(pred_z_complex, target_z)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for all parameters
        all_params = list(self.decomposer.parameters())
        if self.output_proj:
            all_params.extend(self.output_proj.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        self.optimizer.step()

        # Compute metrics
        metrics = {
            "loss": loss.item(),
        }

        # Binding coherence metrics if available
        if self.binding:
            with torch.no_grad():
                _, coherence = self.binding(decomposed)
                if coherence.numel() > 0:
                    ratio = PhaseCoherenceMetrics.diagonal_coherence_ratio(coherence)
                    metrics["coherence_ratio"] = ratio

        # Phase smoothness
        smoothness = PhaseCoherenceMetrics.phase_gradient_smoothness(decomposed.phase)
        metrics["phase_smoothness"] = smoothness

        # Complex correlation
        if pred_z_complex.shape[1] > 1:
            corr = PhaseCoherenceMetrics.complex_state_correlation(
                pred_z_complex[:, 0, :],
                pred_z_complex[:, 1, :]
            )
            metrics["complex_correlation"] = corr

        # Update history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        return metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {
            "loss": [],
            "coherence_ratio": [],
            "phase_smoothness": [],
            "complex_correlation": [],
        }

        for batch in dataloader:
            # Move to device and convert to SpectralTensor if needed
            if isinstance(batch, tuple):
                batch = batch[0]

            if isinstance(batch, torch.Tensor):
                # Convert tensor to SpectralTensor
                batch = SpectralTensor(
                    amplitude=batch.abs(),
                    phase=batch.angle(),
                    scale=torch.linspace(0, 1, batch.shape[-1]),
                    uncertainty=torch.ones_like(batch.abs()) * 0.1,
                )

            batch = batch.to(self.device)
            step_metrics = self.train_step(batch)

            for key in epoch_metrics:
                if key in step_metrics:
                    epoch_metrics[key].append(step_metrics[key])

        # Aggregate
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in epoch_metrics.items()
        }

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary of training metrics."""
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in self.metrics_history.items()
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "decomposer": self.decomposer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": self.metrics_history,
        }
        if self.binding:
            checkpoint["binding"] = self.binding.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.decomposer.load_state_dict(checkpoint["decomposer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.metrics_history = checkpoint.get("metrics", self.metrics_history)
        if self.binding and "binding" in checkpoint:
            self.binding.load_state_dict(checkpoint["binding"])


def train_complex_bifrost_simple(
    decomposer: ComplexSpectralDecomposer,
    train_data: DataLoader,
    epochs: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Simple training function for ComplexSpectralDecomposer.

    Example:
        >>> decomp = ComplexSpectralDecomposer(n_fft=512, d_model=128)
        >>> loader = DataLoader(dataset, batch_size=8)
        >>> results = train_complex_fbc_simple(decomp, loader, epochs=20)
    """
    trainer = ComplexBifrostTrainer(
        decomposer=decomposer,
        lr=1e-4,
        device=device,
    )

    print(f"Training ComplexSpectralDecomposer for {epochs} epochs...")

    for epoch in range(epochs):
        metrics = trainer.train_epoch(train_data)
        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Loss={metrics['loss']:.4f}, "
            f"CoherenceRatio={metrics.get('coherence_ratio', 0):.3f}, "
            f"PhaseSmoothness={metrics['phase_smoothness']:.3f}, "
            f"ComplexCorr={metrics.get('complex_correlation', 0):.3f}"
        )

    summary = trainer.get_metrics_summary()
    print(f"\nTraining complete. Final metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

    return {
        "trainer": trainer,
        "metrics": summary,
    }


# Backward compatibility alias
train_complex_fbc_simple = train_complex_bifrost_simple
