"""
Train S3 Attractor Learning Module with Real Dataset

Trains the attractor dynamics on real audio data to learn
stable phase-lock patterns that correlate with semantic classes.

"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple

from bifrost.pipeline import BifrostPipeline
from bifrost.datasets import load_dataset, auto_detect_dataset, create_data_loader
from bifrost.s3_attractor.attractor_learning import AttractorLearningModule


class AttractorLearningTrainer:
    """
    Trainer for S3 Attractor Learning module.
    
    Trains attractor prototypes to recognize stable phase patterns
    associated with semantic categories.
    """
    
    def __init__(
        self,
        pipeline: BifrostPipeline,
        n_attractors: int = 16,
        n_bands: int = 8,
        device: str = "cpu",
        lr: float = 1e-3,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize trainer.
        
        Parameters
        ----------
        pipeline : BifrostPipeline
            Full Bifrost pipeline (S0-S2 frozen, S3 trained)
        n_attractors : int
            Number of attractor prototypes
        n_bands : int
            Number of frequency bands
        device : str
            Device for training
        lr : float
            Learning rate
        temperature : float
            Softmax temperature for attractor assignment
        """
        self.pipeline = pipeline.to(device)
        self.device = device
        self.temperature = temperature
        
        # Freeze S0-S2, only train S3
        for param in self.pipeline.canonicalizer.parameters():
            param.requires_grad = False
        for param in self.pipeline.decomposer.parameters():
            param.requires_grad = False
        if hasattr(self.pipeline, 'binding'):
            for param in self.pipeline.binding.parameters():
                param.requires_grad = False
        
        # Ensure S3 attractor learner exists
        if not hasattr(self.pipeline, 'attractor_learner') or self.pipeline.attractor_learner is None:
            d_model = getattr(pipeline.decomposer, 'd_model', 128)
            self.pipeline.attractor_learner = AttractorLearningModule(
                d_model=d_model,
                n_bands=n_bands,
                n_attractors=n_attractors,
            ).to(device)
        
        # Optimizer for S3 only
        self.optimizer = optim.Adam(
            self.pipeline.attractor_learner.parameters(),
            lr=lr,
        )
        
        # Loss: stability should correlate with semantic consistency
        self.criterion = nn.CrossEntropyLoss()
        self.history = {"loss": [], "stability": []}
    
    def train_step(
        self,
        signals: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Objective: Attractors assigned to same class should have
        similar stability patterns; different classes → different patterns.
        
        Parameters
        ----------
        signals : List[Tensor]
            Audio signals
        labels : Tensor
            Class labels
        
        Returns
        -------
        Dict
            Training metrics
        """
        self.optimizer.zero_grad()
        
        # Extract attractors for each signal
        all_attractor_stabilities = []
        all_assignments = []
        
        for signal in signals:
            signal = signal.to(self.device)
            
            # Full pipeline forward
            bound, coherence = self.pipeline(signal)
            
            # Get attractors from S3
            attractors, assignment_probs = self.pipeline.attractor_learner(bound)
            
            # Collect stability predictions
            stabilities = torch.stack([a.stability for a in attractors])  # (n_attractors,)
            all_attractor_stabilities.append(stabilities)
            all_assignments.append(assignment_probs)
        
        # Stack: (B, n_attractors)
        stability_matrix = torch.stack(all_attractor_stabilities)
        assignment_matrix = torch.stack(all_assignments)  # (B, n_attractors)
        labels = labels.to(self.device)
        
        # Loss 1: Attractor assignment should predict class
        # Average assignment probabilities → class prediction
        class_logits = assignment_matrix.mean(dim=1)  # (B, n_attractors)
        
        # Project to num_classes (if n_attractors != n_classes)
        if class_logits.shape[1] != labels.max().item() + 1:
            projection = nn.Linear(
                class_logits.shape[1],
                labels.max().item() + 1,
                device=self.device,
            )
            class_logits = projection(class_logits)
        
        cls_loss = self.criterion(class_logits, labels)
        
        # Loss 2: Similar classes → similar stability patterns (contrastive)
        contrastive_loss = 0.0
        n_pairs = 0
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                sim = torch.cosine_similarity(
                    stability_matrix[i].unsqueeze(0),
                    stability_matrix[j].unsqueeze(0),
                )
                if labels[i] == labels[j]:
                    # Same class: stability should be similar
                    contrastive_loss += (1 - sim)
                else:
                    # Different class: stability should differ
                    contrastive_loss += torch.relu(sim - 0.5)
                n_pairs += 1
        
        if n_pairs > 0:
            contrastive_loss = contrastive_loss / n_pairs
        
        # Loss 3: Stability should be calibrated (not all 0.5)
        calibration_loss = ((stability_matrix - 0.5).abs().mean() - 0.2).clamp(min=0)
        
        # Combined
        loss = cls_loss + 0.5 * contrastive_loss + 0.1 * calibration_loss
        
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "contrastive": contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
            "avg_stability": stability_matrix.mean().item(),
        }
    
    def save_checkpoint(self, path: Path) -> None:
        """Save trained attractor module."""
        checkpoint = {
            "attractor_learner": self.pipeline.attractor_learner.state_dict(),
            "history": self.history,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load trained attractor module."""
        checkpoint = torch.load(path, map_location=self.device)
        self.pipeline.attractor_learner.load_state_dict(checkpoint["attractor_learner"])
        self.history = checkpoint.get("history", {})


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train S3 Attractor Learning with Real Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset (folder, CSV, or JSON)",
    )
    parser.add_argument(
        "--n-attractors",
        type=int,
        default=16,
        help="Number of attractor prototypes",
    )
    parser.add_argument(
        "--n-bands",
        type=int,
        default=8,
        help="Number of frequency bands",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
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
        default="checkpoints/attractor_learner.pt",
        help="Checkpoint save path",
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=str,
        default=None,
        help="Path to pretrained spectral encoder checkpoint",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data",
    )
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Training Attractor Learning on: {device}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    if args.dataset and not args.use_synthetic:
        try:
            dataset = auto_detect_dataset(
                args.dataset,
                sample_rate=16000,
                max_duration=5.0,
            )
            n_classes = len(dataset.classes)
            print(f"  Loaded {len(dataset)} samples from {args.dataset}")
            print(f"  Classes: {dataset.classes}")
        except Exception as e:
            print(f"  Error: {e}")
            args.use_synthetic = True
    
    if args.use_synthetic or not args.dataset:
        from bifrost.datasets import SyntheticAudioDataset
        dataset = SyntheticAudioDataset(
            n_samples=300,
            n_classes=3,
            duration=2.0,
        )
        n_classes = 3
        print(f"  Using synthetic: {len(dataset)} samples")
    
    loader = create_data_loader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize pipeline
    print(f"\nInitializing Pipeline...")
    pipeline = BifrostPipeline(
        n_fft_canonical=512,
        d_model=128,
        use_complex_ssm=True,
        use_s3_attractor=True,
        n_bands=args.n_bands,
    )
    
    # Load pretrained encoder if provided
    if args.pretrained_encoder:
        print(f"  Loading pretrained encoder from {args.pretrained_encoder}")
        try:
            checkpoint = torch.load(args.pretrained_encoder, map_location=device)
            pipeline.canonicalizer.load_state_dict(checkpoint["canonicalizer"])
            pipeline.decomposer.load_state_dict(checkpoint["decomposer"])
        except Exception as e:
            print(f"  Warning: Could not load encoder: {e}")
    
    print(f"  n_attractors: {args.n_attractors}")
    print(f"  n_bands: {args.n_bands}")
    
    # Trainer
    trainer = AttractorLearningTrainer(
        pipeline=pipeline,
        n_attractors=args.n_attractors,
        n_bands=args.n_bands,
        device=device,
        lr=args.lr,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training Attractor Learning for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_stability = 0.0
        n_batches = 0
        
        for waveforms, labels, _ in loader:
            signals = [waveforms[i, 0, :].to(device) for i in range(waveforms.shape[0])]
            
            metrics = trainer.train_step(signals, labels)
            epoch_loss += metrics["loss"]
            epoch_stability += metrics["avg_stability"]
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_stability = epoch_stability / n_batches
        
        trainer.history["loss"].append(avg_loss)
        trainer.history["stability"].append(avg_stability)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}, Stability={avg_stability:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(save_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Stability: {avg_stability:.4f}")
    print(f"Model saved: {save_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
