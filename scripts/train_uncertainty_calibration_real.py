#!/usr/bin/env python
"""
Train Uncertainty Calibration on Real-World Data

Uses real audio/text/image datasets instead of synthetic data.
For production-level uncertainty calibration.

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from bifrost.llm_adapter import BifrostEnhancedLLM, SpectralProjector


class RealAudioDataset(Dataset):
    """
    Real audio dataset with natural uncertainty.
    
    Sources: LibriSpeech, Common Voice, AudioSet
    """
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 128,
        d_model: int = 128,
    ):
        """
        Initialize real audio dataset.
        
        Parameters
        ----------
        data_path : str
            Path to audio files directory
        max_length : int
            Maximum sequence length
        d_model : int
            Model dimension
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.d_model = d_model
        
        # Load audio files (recursive search for wav, mp3, flac)
        self.audio_files = (
            list(self.data_path.rglob("*.wav")) +
            list(self.data_path.rglob("*.mp3")) +
            list(self.data_path.rglob("*.flac"))
        )
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_path} (searched recursively for .wav, .mp3, .flac)")
        
        print(f"Loaded {len(self.audio_files)} audio files")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Get audio sample and its inherent difficulty.
        
        Difficulty estimated from:
        - Signal-to-noise ratio (SNR)
        - Duration (shorter = harder)
        - Audio quality (bitrate, sample rate)
        
        Returns
        -------
        sample : Tensor
            Audio sample (max_length, d_model)
        difficulty : float
            Estimated difficulty [0, 1]
        """
        import librosa
        import numpy as np
        
        audio_path = self.audio_files[idx]
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Estimate difficulty
        # 1. SNR estimation (simplified)
        signal_power = np.mean(y ** 2)
        noise_power = np.var(y - np.mean(y))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        snr_difficulty = max(0, min(1, (20 - snr) / 20))  # Lower SNR = higher difficulty
        
        # 2. Duration difficulty
        duration = len(y) / sr
        duration_difficulty = max(0, min(1, (5 - duration) / 5))  # Shorter = harder
        
        # 3. Combined difficulty
        difficulty = 0.7 * snr_difficulty + 0.3 * duration_difficulty
        
        # Convert to tensor
        # Resample or pad to max_length
        if len(y) < self.max_length:
            y = np.pad(y, (0, self.max_length - len(y)))
        else:
            y = y[:self.max_length]
        
        # Create d_model channels (repeat signal)
        sample = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).repeat(1, self.d_model)
        
        return sample, difficulty


class RealTextDataset(Dataset):
    """
    Real text dataset with natural uncertainty.
    
    Sources: C4, The Pile, Wikipedia
    """
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 128,
        d_model: int = 128,
    ):
        """
        Initialize real text dataset.
        
        Parameters
        ----------
        data_path : str
            Path to text files or HuggingFace dataset
        max_length : int
            Maximum sequence length
        d_model : int
            Model dimension
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.d_model = d_model
        
        # Try to load as HuggingFace dataset
        try:
            from datasets import load_from_disk
            self.dataset = load_from_disk(data_path)
            self.use_hf = True
            print(f"Loaded HuggingFace dataset from {data_path}")
        except:
            # Load as text files
            self.text_files = list(self.data_path.glob("*.txt")) + list(self.data_path.glob("*.json"))
            self.use_hf = False
            print(f"Loaded {len(self.text_files)} text files")
    
    def __len__(self) -> int:
        if self.use_hf:
            return len(self.dataset)
        return len(self.text_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Get text sample and its inherent difficulty.
        
        Difficulty estimated from:
        - Text length (shorter = harder)
        - Vocabulary complexity (rare words)
        - Grammar quality (simplified)
        
        Returns
        -------
        sample : Tensor
            Text sample (max_length, d_model)
        difficulty : float
            Estimated difficulty [0, 1]
        """
        if self.use_hf:
            text = self.dataset[idx]['text']
        else:
            with open(self.text_files[idx], 'r') as f:
                text = f.read()
        
        # Estimate difficulty
        # 1. Length difficulty
        length = len(text.split())
        length_difficulty = max(0, min(1, (50 - length) / 50))  # Shorter = harder
        
        # 2. Vocabulary complexity (simplified)
        words = text.lower().split()
        unique_words = set(words)
        vocab_richness = len(unique_words) / (len(words) + 1)
        vocab_difficulty = 1 - vocab_richness  # Lower richness = harder
        
        # 3. Combined difficulty
        difficulty = 0.6 * length_difficulty + 0.4 * vocab_difficulty
        
        # Convert to tensor (simple embedding: random + hash-based)
        # In production, use real embeddings (e.g., BERT)
        import hashlib
        sample = torch.zeros(self.max_length, self.d_model)
        
        for i, word in enumerate(words[:self.max_length]):
            # Deterministic hash-based embedding
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.d_model
            sample[i, word_hash] = 1.0
        
        return sample, difficulty


class UncertaintyCalibrationTrainer:
    """
    Trainer for uncertainty calibration on real data.
    """
    
    def __init__(
        self,
        projector: SpectralProjector,
        device: str = "cpu",
        lr: float = 1e-3,
        reconstruction_weight: float = 1.0,
        correlation_weight: float = 2.0,
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
        reconstruction_weight : float
            Weight for reconstruction loss
        correlation_weight : float
            Weight for correlation loss
        """
        self.projector = projector.to(device)
        self.device = device
        self.reconstruction_weight = reconstruction_weight
        self.correlation_weight = correlation_weight
        
        # Optimize all projector parameters
        self.optimizer = optim.Adam(
            list(projector.parameters()),
            lr=lr,
        )
        
        self.history = {
            "ece": [],
            "correlation": [],
            "avg_uncertainty": [],
            "reconstruction_loss": [],
        }
        self.current_epoch = 0
    
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
        
        # Compute reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, targets)
        
        # Compute uncertainty calibration loss
        calib_loss, metrics = self.projector.compute_uncertainty_calibration_loss(
            predictions=reconstructed,
            targets=targets,
            uncertainties=spectral.uncertainty,
            n_bins=20,
        )
        
        # Recompute correlation with higher weight
        errors = (reconstructed - targets).abs()
        error_max = errors.max() + 1e-8
        errors_normalized = errors / error_max
        
        error_flat = errors_normalized.flatten()
        uncertainty_flat = spectral.uncertainty.flatten()
        
        error_mean = error_flat.mean()
        uncertainty_mean = uncertainty_flat.mean()
        error_std = error_flat.std() + 1e-8
        uncertainty_std = uncertainty_flat.std() + 1e-8
        
        correlation = ((error_flat - error_mean) * (uncertainty_flat - uncertainty_mean)).mean()
        correlation = correlation / (error_std * uncertainty_std)
        correlation_loss = -correlation
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            calib_loss +
            self.correlation_weight * correlation_loss
        )
        
        # Backward
        total_loss.backward()
        self.optimizer.step()
        
        metrics["reconstruction_loss"] = reconstruction_loss.item()
        
        # Update history
        self.history["ece"].append(metrics["ece"])
        self.history["correlation"].append(metrics["correlation"])
        self.history["avg_uncertainty"].append(metrics["avg_uncertainty"])
        self.history["reconstruction_loss"].append(metrics["reconstruction_loss"])
        
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
                n_bins=20,
            )
        
        self.projector.train()
        return metrics
    
    def save_checkpoint(self, path: Path) -> None:
        """
        Save trained projector parameters.
        
        Parameters
        ----------
        path : Path
            Checkpoint save path
        """
        checkpoint = {
            "projector_state_dict": self.projector.state_dict(),
            "uncertainty_temperature": self.projector.uncertainty_temperature.item(),
            "uncertainty_bias": self.projector.uncertainty_bias.item(),
            "history": self.history,
            "current_epoch": self.current_epoch,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load trained projector parameters.
        
        Parameters
        ----------
        path : Path
            Checkpoint load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.projector.load_state_dict(checkpoint["projector_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.current_epoch = checkpoint.get("current_epoch", 0)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Uncertainty Calibration on Real-World Data"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["audio", "text"],
        default="audio",
        help="Type of real data (audio or text)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to real data directory",
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
        "--reconstruction-weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )
    parser.add_argument(
        "--correlation-weight",
        type=float,
        default=2.0,
        help="Weight for correlation loss",
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
        default="checkpoints/uncertainty_calibration_real.pt",
        help="Checkpoint save path",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Starting epoch (overrides checkpoint current_epoch)",
    )
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("Uncertainty Calibration Training (Real Data)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data type: {args.data_type}")
    print(f"Data path: {args.data_path}")
    print(f"d_model: {args.d_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Initialize projector
    projector = SpectralProjector(
        d_model=args.d_model,
        spectral_dim=128,
    )
    
    # Initialize temperature to positive value
    projector.uncertainty_temperature.data.fill_(0.5)
    
    import torch.nn.functional as F
    temp = F.softplus(projector.uncertainty_temperature).item()
    bias = F.softplus(projector.uncertainty_bias).item()
    print(f"Initial uncertainty temperature: {temp:.4f}")
    print(f"Initial uncertainty bias: {bias:.4f}")
    print()
    
    # Create dataset
    print("Loading real-world dataset...")
    if args.data_type == "audio":
        dataset = RealAudioDataset(
            data_path=args.data_path,
            max_length=128,
            d_model=args.d_model,
        )
    else:
        dataset = RealTextDataset(
            data_path=args.data_path,
            max_length=128,
            d_model=args.d_model,
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
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Trainer
    trainer = UncertaintyCalibrationTrainer(
        projector=projector,
        device=device,
        lr=args.lr,
        reconstruction_weight=args.reconstruction_weight,
        correlation_weight=args.correlation_weight,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(Path(args.resume_from))
        if args.start_epoch is not None:
            trainer.current_epoch = args.start_epoch
        print(f"Resuming from epoch {trainer.current_epoch}")
        print()
    
    # Train
    print("=" * 60)
    print("Training")
    print("=" * 60)
    print()
    
    best_ece = float('inf')
    
    for epoch in range(trainer.current_epoch, args.epochs):
        epoch_ece = 0.0
        epoch_correlation = 0.0
        n_batches = 0
        
        # Train
        for samples, difficulties in train_loader:
            samples = samples.to(device)
            
            # Add batch dimension if needed
            if samples.dim() == 2:
                samples = samples.unsqueeze(0)
            
            # Create targets with noise based on difficulty
            targets = samples.clone()
            noise = torch.randn_like(samples) * 0.5
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
            
            if samples.dim() == 2:
                samples = samples.unsqueeze(0)
            
            targets = samples.clone()
            noise = torch.randn_like(samples) * 0.5
            for i in range(samples.shape[0]):
                noise[i] *= difficulties[i].item()
            targets = targets + noise
            
            metrics = trainer.evaluate(samples, targets)
            val_ece += metrics["ece"]
            val_correlation += metrics["correlation"]
            val_n_batches += 1
        
        val_avg_ece = val_ece / val_n_batches
        val_avg_correlation = val_correlation / val_n_batches
        
        avg_recon_loss = sum(trainer.history["reconstruction_loss"][-n_batches:]) / n_batches
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train: ECE={avg_ece:.4f}, Correlation={avg_correlation:.4f}, Recon={avg_recon_loss:.4f}")
        print(f"  Val:   ECE={val_avg_ece:.4f}, Correlation={val_avg_correlation:.4f}")
        
        if val_avg_ece < best_ece:
            best_ece = val_avg_ece
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(save_path)
            print(f"  ✓ Saved best checkpoint (ECE={best_ece:.4f})")
        
        trainer.current_epoch = epoch + 1
        print()
    
    # Final evaluation
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    print()
    
    trainer.load_checkpoint(Path(args.save_path))
    
    temp = F.softplus(projector.uncertainty_temperature).item()
    bias = F.softplus(projector.uncertainty_bias).item()
    print(f"Final uncertainty temperature: {temp:.4f}")
    print(f"Final uncertainty bias: {bias:.4f}")
    print()
    print(f"Best validation ECE: {best_ece:.4f}")
    print(f"Target ECE: < 0.1 (well-calibrated)")
    
    if best_ece < 0.1:
        print("✅ UNCERTAINTY WELL-CALIBRATED")
    elif best_ece < 0.2:
        print("⚠️  UNCERTAINTY PARTIALLY CALIBRATED")
    else:
        print("❌ UNCERTAINTY POORLY CALIBRATED - Need more training or better data")
    
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    
    return 0 if best_ece < 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
