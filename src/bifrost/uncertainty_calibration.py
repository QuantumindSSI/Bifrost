"""
Uncertainty Calibration — Temperature Scaling on Held-Out Validation Data

Implements proper uncertainty calibration using:
1. Held-out validation set (never seen during training)
2. Temperature scaling on raw uncertainty logits
3. Expected Calibration Error (ECE) for evaluation

Reference:
- Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
- Temperature scaling: p = sigmoid(z / T) where T is learned on validation set
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Named constants for demo / calibration defaults
_DEMO_N_SAMPLES: int = 1000


@dataclass
class CalibrationMetrics:
    """Metrics for uncertainty calibration quality."""
    expected_calibration_error: float  # ECE in [0, 1], lower is better
    max_calibration_error: float  # Maximum calibration gap
    brier_score: float  # Proper scoring rule, lower is better
    negative_log_likelihood: float  # NLL, lower is better
    
    def is_well_calibrated(self, ece_threshold: float = 0.1) -> bool:
        """Check if model is well-calibrated (ECE < threshold)."""
        return self.expected_calibration_error < ece_threshold


class UncertaintyCalibrator(nn.Module):
    """
    Calibrate uncertainty estimates using held-out validation data.
    
    Uses temperature scaling to map raw network outputs to well-calibrated
    uncertainty probabilities. The temperature parameter is learned on a
    held-out validation set, separate from training data.
    
    The calibration criterion: uncertainty should equal the probability of error.
    If model predicts 0.8 uncertainty, it should be wrong 80% of the time.
    
    Attributes:
        temperature: Learnable temperature parameter (initialized from pre-training)
        bias: Learnable bias parameter
        is_calibrated: Whether calibration has been performed on validation data
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        initial_bias: float = 0.0,
        n_bins: int = 10,  # For ECE computation
    ):
        super().__init__()
        # Initialize from pre-training values
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.bias = nn.Parameter(torch.tensor(initial_bias))
        self.n_bins = n_bins
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_temperature = initial_temperature  # Post-calibration value
        self.calibration_bias = initial_bias
        
        # Store calibration statistics
        self.register_buffer('pre_calibration_ece', torch.tensor(-1.0))
        self.register_buffer('post_calibration_ece', torch.tensor(-1.0))
    
    def forward(self, raw_uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Apply calibrated uncertainty transformation.
        
        Args:
            raw_uncertainty: Raw network outputs (any range)
            
        Returns:
            Calibrated uncertainty in [0, 1]
        """
        # Temperature scaling: p = sigmoid((raw - bias) / temperature)
        # Using softplus to ensure positive temperature
        temp = F.softplus(self.temperature)
        scaled = (raw_uncertainty - self.bias) / (temp + 1e-8)
        calibrated = torch.sigmoid(scaled)
        
        return calibrated
    
    def compute_expected_calibration_error(
        self,
        uncertainties: torch.Tensor,  # Predicted uncertainties
        errors: torch.Tensor,  # Binary: 1 = model was wrong, 0 = correct
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the gap between predicted uncertainty and actual error rate.
        
        Args:
            uncertainties: Predicted uncertainties in [0, 1]
            errors: Binary indicators of prediction errors
            
        Returns:
            ECE value in [0, 1], where 0 = perfectly calibrated
        """
        if len(uncertainties) == 0:
            return 0.0
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(uncertainties)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            bin_size = in_bin.sum().item()
            
            if bin_size > 0:
                # Average predicted uncertainty in bin
                avg_confidence = uncertainties[in_bin].mean().item()
                
                # Actual error rate in bin
                avg_error = errors[in_bin].float().mean().item()
                
                # Calibration gap: |confidence - accuracy|
                # For uncertainty: gap = |uncertainty - error_rate|
                gap = abs(avg_confidence - avg_error)
                
                # Weight by bin size
                ece += (bin_size / total_samples) * gap
        
        return ece
    
    def compute_brier_score(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
    ) -> float:
        """
        Compute Brier score (proper scoring rule).
        
        Brier = E[(uncertainty - error)^2]
        
        Lower is better (0 = perfect).
        """
        brier = ((uncertainties - errors.float()) ** 2).mean().item()
        return brier
    
    def calibrate(
        self,
        validation_uncertainties: torch.Tensor,  # Raw network outputs on validation set
        validation_errors: torch.Tensor,  # Binary error indicators on validation set
        max_iterations: int = 100,
        learning_rate: float = 0.01,
        target_ece: float = 0.05,  # Target ECE for stopping
    ) -> CalibrationMetrics:
        """
        Calibrate uncertainty using held-out validation data.
        
        This method optimizes temperature and bias to minimize ECE on the
        validation set. The validation set must be held-out (not used in training).
        
        Args:
            validation_uncertainties: Raw uncertainty outputs on validation set
            validation_errors: Binary error indicators (1 = wrong, 0 = correct)
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for temperature optimization
            target_ece: Stop if ECE falls below this threshold
            
        Returns:
            CalibrationMetrics before and after calibration
        """
        self._print_calibration_header(
            len(validation_uncertainties), target_ece
        )

        pre_ece, pre_brier = self._compute_pre_metrics(
            validation_uncertainties, validation_errors
        )
        self.pre_calibration_ece = torch.tensor(pre_ece)
        print(f"Pre-calibration ECE: {pre_ece:.4f}")
        print(f"Pre-calibration Brier: {pre_brier:.4f}\n")

        best_temp, best_bias, best_ece = self._optimize_temperature(
            validation_uncertainties,
            validation_errors,
            max_iterations,
            learning_rate,
            target_ece,
            pre_ece,
        )

        self._restore_best_params(best_temp, best_bias)
        post_ece, post_brier, post_nll = self._compute_post_metrics(
            validation_uncertainties, validation_errors
        )
        self.post_calibration_ece = torch.tensor(post_ece)

        self._print_calibration_results(pre_ece, post_ece)
        return self._build_calibration_metrics(
            post_ece, post_brier, post_nll
        )

    def _print_calibration_header(
        self, n_samples: int, target_ece: float
    ) -> None:
        """Print calibration run header."""
        print("=" * 60)
        print("UNCERTAINTY CALIBRATION")
        print("=" * 60)
        print(f"Validation samples: {n_samples}")
        print(f"Target ECE: {target_ece:.4f}\n")

    def _compute_pre_metrics(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute pre-calibration ECE and Brier score."""
        calib = self.forward(uncertainties)
        ece = self.compute_expected_calibration_error(calib, errors)
        brier = self.compute_brier_score(calib, errors)
        return ece, brier

    def _optimize_temperature(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        max_iterations: int,
        learning_rate: float,
        target_ece: float,
        pre_ece: float,
    ) -> Tuple[float, float, float]:
        """Optimize temperature and bias on validation data."""
        optimizer = torch.optim.Adam(
            [self.temperature, self.bias], lr=learning_rate
        )
        best_ece = pre_ece
        best_temp = self.temperature.item()
        best_bias = self.bias.item()

        for iteration in range(max_iterations):
            optimizer.zero_grad()
            calibrated = self.forward(uncertainties)
            nll = F.binary_cross_entropy(calibrated, errors.float())
            temp_reg = 0.01 * F.softplus(self.temperature).abs()
            loss = nll + temp_reg
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                eval_calib = self.forward(uncertainties)
                ece = self.compute_expected_calibration_error(
                    eval_calib, errors
                )
                if ece < best_ece:
                    best_ece = ece
                    best_temp = self.temperature.item()
                    best_bias = self.bias.item()
                if (iteration + 1) % 10 == 0:
                    print(
                        f"  Iter {iteration+1}: ECE={ece:.4f}, "
                        f"T={F.softplus(self.temperature).item():.3f}"
                    )
                if ece < target_ece:
                    print(f"\nTarget ECE reached at iteration {iteration+1}")
                    break

        return best_temp, best_bias, best_ece

    def _restore_best_params(self, best_temp: float, best_bias: float) -> None:
        """Restore best temperature and bias, set calibration state."""
        self.temperature.data = torch.tensor(best_temp)
        self.bias.data = torch.tensor(best_bias)
        self.is_calibrated = True
        self.calibration_temperature = F.softplus(self.temperature).item()
        self.calibration_bias = self.bias.item()

    def _compute_post_metrics(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Compute post-calibration ECE, Brier, and NLL."""
        calib = self.forward(uncertainties)
        ece = self.compute_expected_calibration_error(calib, errors)
        brier = self.compute_brier_score(calib, errors)
        with torch.no_grad():
            nll = F.binary_cross_entropy(
                calib, errors.float()
            ).item()
        return ece, brier, nll

    def _print_calibration_results(
        self, pre_ece: float, post_ece: float
    ) -> None:
        """Print calibration summary."""
        print()
        print("=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)
        print(f"Pre-calibration ECE:  {pre_ece:.4f}")
        print(f"Post-calibration ECE: {post_ece:.4f}")
        print(f"Improvement:          {(pre_ece - post_ece):.4f}\n")
        print(f"Optimal temperature: {self.calibration_temperature:.4f}")
        print(f"Optimal bias:        {self.calibration_bias:.4f}\n")
        if post_ece < pre_ece:
            print("✅ Calibration successful: ECE reduced")
        else:
            print("⚠️  Calibration did not improve ECE")

    def _build_calibration_metrics(
        self,
        post_ece: float,
        post_brier: float,
        post_nll: float,
    ) -> CalibrationMetrics:
        """Build and return CalibrationMetrics dataclass."""
        return CalibrationMetrics(
            expected_calibration_error=post_ece,
            max_calibration_error=0.0,
            brier_score=post_brier,
            negative_log_likelihood=post_nll,
        )
    
    def save_calibration(self, path: str) -> None:
        """Save calibration parameters to file."""
        torch.save({
            'temperature': self.calibration_temperature,
            'bias': self.calibration_bias,
            'is_calibrated': self.is_calibrated,
            'pre_calibration_ece': self.pre_calibration_ece.item(),
            'post_calibration_ece': self.post_calibration_ece.item(),
        }, path)
    
    def load_calibration(self, path: str) -> None:
        """Load calibration parameters from file."""
        checkpoint = torch.load(path)
        self.calibration_temperature = checkpoint['temperature']
        self.calibration_bias = checkpoint['bias']
        self.is_calibrated = checkpoint['is_calibrated']
        
        # Update parameters
        self.temperature.data = torch.tensor(checkpoint['temperature'])
        self.bias.data = torch.tensor(checkpoint['bias'])


def calibrate_model_uncertainty(
    model: nn.Module,
    validation_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> UncertaintyCalibrator:
    """
    High-level function to calibrate model uncertainty.
    
    Args:
        model: Model with uncertainty outputs
        validation_loader: DataLoader for held-out validation set
        device: Device for computation
        
    Returns:
        Calibrated UncertaintyCalibrator
    """
    print("Collecting uncertainty estimates on validation set...")
    
    all_uncertainties = []
    all_errors = []
    
    model.eval()
    with torch.no_grad():
        for batch in validation_loader:
            # Assuming batch format: (inputs, targets)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
            else:
                # Single tensor, can't compute errors
                continue
            
            # Forward pass
            outputs = model(inputs)
            
            # Extract uncertainties (model-specific)
            if hasattr(outputs, 'uncertainty') and outputs.uncertainty is not None:
                uncertainties = outputs.uncertainty
            elif isinstance(outputs, dict) and 'uncertainty' in outputs:
                uncertainties = outputs['uncertainty']
            else:
                continue
            
            # Compute predictions and errors
            if hasattr(outputs, 'logits'):
                predictions = outputs.logits.argmax(dim=-1)
            elif 'logits' in outputs:
                predictions = outputs['logits'].argmax(dim=-1)
            else:
                continue
            
            errors = (predictions != targets).float()
            
            # Aggregate per-sample
            if uncertainties.dim() > 1:
                uncertainties = uncertainties.mean(dim=list(range(1, uncertainties.dim())))
            
            all_uncertainties.append(uncertainties.cpu())
            all_errors.append(errors.cpu())
    
    if len(all_uncertainties) == 0:
        raise ValueError("No uncertainty estimates collected. Check model output format.")
    
    # Concatenate
    uncertainties = torch.cat(all_uncertainties)
    errors = torch.cat(all_errors)
    
    print(f"Collected {len(uncertainties)} uncertainty estimates")
    print(f"Error rate: {errors.mean().item():.4f}")
    print()
    
    # Create calibrator and calibrate
    calibrator = UncertaintyCalibrator()
    metrics = calibrator.calibrate(uncertainties, errors)
    
    return calibrator


def demo_calibration():
    """Demonstrate uncertainty calibration on synthetic data."""
    print("=" * 60)
    print("UNCERTAINTY CALIBRATION DEMO")
    print("=" * 60)
    print()
    
    # Generate synthetic validation data
    # Scenario: model is overconfident (uncertainty < actual error rate)
    n_samples = _DEMO_N_SAMPLES
    
    # Actual error rate is 0.3, but model predicts 0.1 uncertainty (overconfident)
    actual_errors = torch.bernoulli(torch.ones(n_samples) * 0.3)
    
    # Overconfident predictions
    raw_uncertainties = torch.randn(n_samples) * 0.1 + 0.1  # Centered at 0.1
    raw_uncertainties = torch.clamp(raw_uncertainties, 0.01, 0.99)
    
    print("Synthetic scenario:")
    print(f"  Actual error rate: {actual_errors.mean().item():.4f}")
    print(f"  Avg predicted uncertainty (pre-calib): {raw_uncertainties.mean().item():.4f}")
    print()
    
    # Calibrate
    calibrator = UncertaintyCalibrator()
    metrics = calibrator.calibrate(raw_uncertainties, actual_errors, target_ece=0.02)
    
    print()
    print(f"Well-calibrated: {metrics.is_well_calibrated(ece_threshold=0.05)}")


if __name__ == "__main__":
    demo_calibration()
