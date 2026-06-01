#!/usr/bin/env python
"""
Train Uncertainty Calibration for SpectralProjector

Trains the uncertainty temperature and bias parameters to ensure
predicted uncertainty correlates with actual prediction error.

Per Agentic CTO-Persona Policy:
- C-01: Full documentation
- C-02: Explicit error handling
- C-04: Complexity ≤10
- G1-G5: Complete, executable, correct
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from bifrost.llm_adapter import BifrostEnhancedLLM, SpectralProjector


class DifficultyLabeledDataset:
    """
    Dataset with samples labeled by difficulty (expected error).
    
    Difficulty is estimated via:
    - Noise level in signal
    - Length of sequence
    - Out-of-distribution score
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        max_length: int = 128,
        n_classes: int = 2,
        difficulty_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize difficulty-labeled dataset.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        max_length : int
            Maximum sequence length
        n_classes : int
            Number of classes
        difficulty_range : Tuple[float, float]
            Min and max difficulty values
        """
        self.n_samples = n_samples
        self.max_length = max_length
        self.n_classes = n_classes
        self.difficulty_range = difficulty_range
        
        # Generate synthetic data with varying difficulty
        self.samples = []
        self.difficulties = []
        
        for i in range(n_samples):
            # Sample difficulty from uniform distribution
            difficulty = difficulty_range[0] + (difficulty_range[1] - difficulty_range[0]) * (i / n_samples)
            
            # Generate sample with noise proportional to difficulty
            sample = self._generate_sample(difficulty, max_length, n_classes)
            
            self.samples.append(sample)
            self.difficulties.append(difficulty)
    
    def _generate_sample(
        self,
        difficulty: float,
        max_length: int,
        n_classes: int,
    ) -> torch.Tensor:
        """
        Generate a sample with noise proportional to difficulty.
        
        Parameters
        ----------
        difficulty : float
            Difficulty level [0, 1]
        max_length : int
            Maximum sequence length
        n_classes : int
            Number of classes
            
        Returns
        -------
        Tensor
            Sample tensor (max_length, d_model)
        """
        d_model = 128  # Match default d_model
        
        # Base signal (clean)
        base = torch.randn(max_length, d_model) * 0.1
        
        # Add noise proportional to difficulty
        noise = torch.randn(max_length, d_model) * (difficulty * 0.5)
        
        # Length variation (shorter = harder)
        length = int(max_length * (1.0 - 0.5 * difficulty))
        sample = base + noise
        sample[length:] = 0  # Pad with zeros
        
        return sample
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Get sample and its difficulty label.
        
        Returns
        -------
        sample : Tensor
            Sample tensor
        difficulty : float
            Difficulty label [0, 1]
        """
        return self.samples[idx], self.difficulties[idx]


class UncertaintyCalibrationTrainer:
    """
    Trainer for uncertainty calibration.
    
    Trains SpectralProjector uncertainty parameters (temperature, bias)
    to ensure predicted uncertainty correlates with actual error.
    """
    
    def __init__(
        self,
        projector: SpectralProjector,
        device: str = "cpu",
        lr: float = 1e-3,
    ) -> None:
        """
        Initialize trainer.
        
        Parameters
        ----------
        projector : SpectralProjector
            Spectral projector with uncertainty parameters
        device : str
            Device for training
        lr : float
            Learning rate
        """
        self.projector = projector.to(device)
        self.device = device
        
        # Only optimize uncertainty parameters (temperature, bias)
        self.optimizer = optim.Adam(
            [
                projector.uncertainty_temperature,
                projector.uncertainty_bias,
            ],
            lr=lr,
        )
        
        self.history = {
            "ece": [],
            "correlation": [],
            "avg_uncertainty": [],
        }
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Parameters
        ----------
        inputs : Tensor
            Input hidden states (B, T, d_model)
        targets : Tensor
            Target hidden states (B, T, d_model)
            
        Returns
        -------
        Dict
            Training metrics
        """
        self.optimizer.zero_grad()
        
        # Project to spectral space and back (reconstruction)
        spectral, reconstructed = self.projector(inputs)
        
        # Compute uncertainty calibration loss
        calib_loss, metrics = self.projector.compute_uncertainty_calibration_loss(
            predictions=reconstructed,
            targets=targets,
            uncertainties=spectral.uncertainty,
            n_bins=10,
        )
        
        # Backward
        calib_loss.backward()
        self.optimizer.step()
        
        # Update history
        self.history["ece"].append(metrics["ece"])
        self.history["correlation"].append(metrics["correlation"])
        self.history["avg_uncertainty"].append(metrics["avg_uncertainty"])
        
        return metrics
    
    def evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate uncertainty calibration.
        
        Parameters
        ----------
        inputs : Tensor
            Input hidden states (B, T, d_model)
        targets : Tensor
            Target hidden states (B, T, d_model)
            
        Returns
        -------
        Dict
            Evaluation metrics
        """
        self.projector.eval()
        
        with torch.no_grad():
            spectral, reconstructed = self.projector(inputs)
            
            calib_loss, metrics = self.projector.compute_uncertainty_calibration_loss(
                predictions=reconstructed,
                targets=targets,
                uncertainties=spectral.uncertainty,
                n_bins=10,
            )
        
        self.projector.train()
        return metrics
    
    def save_checkpoint(self, path: Path) -> None:
        """
        Save trained uncertainty parameters.
        
        Parameters
        ----------
        path : Path
            Checkpoint save path
        """
        checkpoint = {
            "uncertainty_temperature": self.projector.uncertainty_temperature.item(),
            "uncertainty_bias": self.projector.uncertainty_bias.item(),
            "history": self.history,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load trained uncertainty parameters.
        
        Parameters
        ----------
        path : Path
            Checkpoint load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.projector.uncertainty_temperature.data.fill_(checkpoint["uncertainty_temperature"])
        self.projector.uncertainty_bias.data.fill_(checkpoint["uncertainty_bias"])
        self.history = checkpoint.get("history", self.history)
    
    def generate_reliability_diagram(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10,
        save_path: Path = None,
    ) -> List[Dict]:
        """
        Generate reliability diagram for uncertainty calibration.
        
        Parameters
        ----------
        inputs : Tensor
            Input hidden states (B, T, d_model)
        targets : Tensor
            Target hidden states (B, T, d_model)
        n_bins : int
            Number of bins for reliability diagram
        save_path : Path, optional
            Path to save reliability diagram JSON
            
        Returns
        -------
        List[Dict]
            Bin statistics for reliability diagram
        """
        self.projector.eval()
        
        with torch.no_grad():
            spectral, reconstructed = self.projector(inputs)
            
            errors = (reconstructed - targets).abs()
            error_max = errors.max() + 1e-8
            errors_normalized = errors / error_max
            
            uncertainties = spectral.uncertainty
            
            # Compute bin statistics
            bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=inputs.device)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_stats = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
                bin_count = in_bin.sum().item()
                
                if bin_count > 0:
                    avg_uncertainty = uncertainties[in_bin].mean().item()
                    avg_error = errors_normalized[in_bin].mean().item()
                    
                    bin_stats.append({
                        "bin_lower": bin_lower.item(),
                        "bin_upper": bin_upper.item(),
                        "count": bin_count,
                        "avg_uncertainty": avg_uncertainty,
                        "avg_error": avg_error,
                        "calibration_error": abs(avg_uncertainty - avg_error),
                    })
        
        self.projector.train()
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(bin_stats, f, indent=2)
        
        return bin_stats


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Uncertainty Calibration for SpectralProjector"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/uncertainty_calibration.pt",
        help="Checkpoint save path",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for reliability diagram",
    )
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("Uncertainty Calibration Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"d_model: {args.d_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Samples: {args.n_samples}")
    print()
    
    # Initialize projector
    projector = SpectralProjector(
        d_model=args.d_model,
        spectral_dim=128,
    )
    
    print(f"Initial uncertainty temperature: {projector.uncertainty_temperature.item():.4f}")
    print(f"Initial uncertainty bias: {projector.uncertainty_bias.item():.4f}")
    print()
    
    # Create dataset
    print("Creating difficulty-labeled dataset...")
    dataset = DifficultyLabeledDataset(
        n_samples=args.n_samples,
        max_length=128,
        n_classes=2,
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Trainer
    trainer = UncertaintyCalibrationTrainer(
        projector=projector,
        device=device,
        lr=args.lr,
    )
    
    # Train
    print("=" * 60)
    print("Training")
    print("=" * 60)
    print()
    
    best_ece = float('inf')
    
    for epoch in range(args.epochs):
        epoch_ece = 0.0
        epoch_correlation = 0.0
        n_batches = 0
        
        # Train
        for samples, difficulties in train_loader:
            samples = samples.to(device)
            
            # Add batch dimension if needed (B, T, d_model)
            if samples.dim() == 2:
                samples = samples.unsqueeze(0)  # (1, T, d_model)
            
            # Create targets (reconstruction target = input)
            targets = samples.clone()
            
            # Add noise to targets based on difficulty
            # Higher difficulty = more noise = higher error
            noise = torch.randn_like(samples) * 0.1
            for i in range(samples.shape[0]):
                noise[i] *= difficulties[i].item()
            targets = targets + noise
            
            metrics = trainer.train_step(samples, targets)
            epoch_ece += metrics["ece"]
            epoch_correlation += metrics["correlation"]
            n_batches += 1
        
        avg_ece = epoch_ece / n_batches
        avg_correlation = epoch_correlation / n_batches
        
        # Validate
        val_ece = 0.0
        val_correlation = 0.0
        val_n_batches = 0
        
        for samples, difficulties in val_loader:
            samples = samples.to(device)
            
            # Add batch dimension if needed (B, T, d_model)
            if samples.dim() == 2:
                samples = samples.unsqueeze(0)  # (1, T, d_model)
            
            targets = samples.clone()
            noise = torch.randn_like(samples) * 0.1
            for i in range(samples.shape[0]):
                noise[i] *= difficulties[i].item()
            targets = targets + noise
            
            metrics = trainer.evaluate(samples, targets)
            val_ece += metrics["ece"]
            val_correlation += metrics["correlation"]
            val_n_batches += 1
        
        val_avg_ece = val_ece / val_n_batches
        val_avg_correlation = val_correlation / val_n_batches
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train: ECE={avg_ece:.4f}, Correlation={avg_correlation:.4f}")
        print(f"  Val:   ECE={val_avg_ece:.4f}, Correlation={val_avg_correlation:.4f}")
        
        if val_avg_ece < best_ece:
            best_ece = val_avg_ece
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(save_path)
            print(f"  ✓ Saved best checkpoint (ECE={best_ece:.4f})")
        
        print()
    
    # Final evaluation
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    print()
    
    # Load best checkpoint
    trainer.load_checkpoint(Path(args.save_path))
    
    print(f"Final uncertainty temperature: {projector.uncertainty_temperature.item():.4f}")
    print(f"Final uncertainty bias: {projector.uncertainty_bias.item():.4f}")
    print()
    
    # Generate reliability diagram
    print("Generating reliability diagram...")
    
    # Use validation set for reliability diagram
    all_samples = []
    all_targets = []
    
    for samples, difficulties in val_loader:
        samples = samples.to(device)
        
        # Add batch dimension if needed (B, T, d_model)
        if samples.dim() == 2:
            samples = samples.unsqueeze(0)  # (1, T, d_model)
        
        targets = samples.clone()
        noise = torch.randn_like(samples) * 0.1
        for i in range(samples.shape[0]):
            noise[i] *= difficulties[i].item()
        targets = targets + noise
        
        all_samples.append(samples)
        all_targets.append(targets)
    
    all_samples = torch.cat(all_samples, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    reliability_path = Path(args.save_path).parent / "reliability_diagram.json"
    bin_stats = trainer.generate_reliability_diagram(
        all_samples,
        all_targets,
        n_bins=args.n_bins,
        save_path=reliability_path,
    )
    
    print(f"Reliability diagram saved to: {reliability_path}")
    print()
    print("Bin Statistics:")
    print(f"{'Bin':<12} {'Count':<8} {'Avg Uncertainty':<16} {'Avg Error':<12} {'Calibration Error':<18}")
    print("-" * 70)
    
    for bin_stat in bin_stats:
        print(
            f"{bin_stat['bin_lower']:.2f}-{bin_stat['bin_upper']:.2f}  "
            f"{bin_stat['count']:<8} "
            f"{bin_stat['avg_uncertainty']:<16.4f} "
            f"{bin_stat['avg_error']:<12.4f} "
            f"{bin_stat['calibration_error']:<18.4f}"
        )
    
    print()
    print(f"Best validation ECE: {best_ece:.4f}")
    print(f"Target ECE: < 0.1 (well-calibrated)")
    
    if best_ece < 0.1:
        print("✅ UNCERTAINTY WELL-CALIBRATED")
    elif best_ece < 0.2:
        print("⚠️  UNCERTAINTY PARTIALLY CALIBRATED")
    else:
        print("❌ UNCERTAINTY POORLY CALIBRATED - Need more training")
    
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    
    return 0 if best_ece < 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
