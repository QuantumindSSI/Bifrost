"""
Train Semantic Coherence with Real Dataset

Per Agentic CTO-Persona Policy:
- C-01: Full documentation
- C-02: Explicit error handling  
- C-04: Complexity ≤10
- G1-G5: Complete, executable, correct
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from bifrost.pipeline import BifrostPipeline
from bifrost.semantic_coherence import train_semantic_coherence, SemanticCoherenceTrainer
from bifrost.datasets import load_dataset, auto_detect_dataset, create_data_loader


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Bifrost Semantic Coherence with Real Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset (folder, CSV, or JSON). Uses synthetic if not provided.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=3,
        help="Number of semantic classes",
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
        default="checkpoints/semantic_coherence.pt",
        help="Where to save trained model",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Max audio duration in seconds",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Force synthetic dataset even if --dataset provided",
    )
    parser.add_argument(
        "--n-fft-s0",
        type=int,
        default=512,
        help="FFT size for canonicalization",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension",
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Training on device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    if args.dataset and not args.use_synthetic:
        try:
            dataset = auto_detect_dataset(
                args.dataset,
                sample_rate=args.sample_rate,
                max_duration=args.max_duration,
            )
            print(f"  Detected {len(dataset)} samples")
            print(f"  Classes: {dataset.classes}")
            args.n_classes = len(dataset.classes)
        except Exception as e:
            print(f"  Failed to load dataset: {e}")
            print(f"  Falling back to synthetic data")
            args.use_synthetic = True
    
    if args.use_synthetic or args.dataset is None:
        from bifrost.datasets import SyntheticAudioDataset
        dataset = SyntheticAudioDataset(
            n_samples=200,
            n_classes=args.n_classes,
            duration=2.0,
            sample_rate=args.sample_rate,
        )
        print(f"  Using synthetic dataset: {len(dataset)} samples")
        print(f"  Classes: {dataset.classes}")
    
    # Create data loader
    train_loader = create_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Safe default
    )
    
    # Initialize pipeline
    print(f"\nInitializing Bifrost Pipeline...")
    pipeline = BifrostPipeline(
        n_fft_s0=args.n_fft_s0,
        d_model=args.d_model,
        use_s3_attractor=True,
        use_complex_ssm=True,
    ).to(device)
    print(f"  d_model: {args.d_model}")
    print(f"  n_fft_s0: {args.n_fft_s0}")
    
    # Create trainer
    print(f"\nInitializing Semantic Coherence Trainer...")
    trainer = SemanticCoherenceTrainer(
        pipeline=pipeline,
        num_classes=args.n_classes,
        device=device,
        lr=args.lr,
    )
    print(f"  num_classes: {args.n_classes}")
    print(f"  lr: {args.lr}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, (waveforms, labels, _) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # Bifrost pipeline expects (B, T) tensor, not list
            # waveforms is (B, 1, T), squeeze to (B, T)
            signals = waveforms.squeeze(1)  # (B, T)
            
            # Training step
            metrics = trainer.train_step(signals, labels)
            epoch_loss += metrics['contrastive_loss']
            n_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {metrics['contrastive_loss']:.4f}")
        
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"Final Evaluation")
    print(f"{'='*60}\n")
    
    # Create test signals
    test_signals = []
    test_labels = []
    for waveforms, labels, _ in train_loader:
        for i in range(min(5, waveforms.shape[0])):
            test_signals.append(waveforms[i, 0, :])
            test_labels.append(labels[i].item())
        if len(test_signals) >= 30:
            break
    
    eval_metrics = trainer.evaluate_semantic_coherence(test_signals, test_labels)
    print(f"  Semantic correlation: {eval_metrics.coherence_semantic_correlation:.3f}")
    print(f"  Retrieval recall@5: {eval_metrics.semantic_retrieval_recall:.3f}")
    print(f"  Phase similarity accuracy: {eval_metrics.phase_similarity_accuracy:.3f}")
    
    # Save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Target check
    if eval_metrics.coherence_semantic_correlation > 0.3:
        print(f"\n✅ TARGET ACHIEVED: Correlation > 0.3")
    else:
        print(f"\n⚠️  Below target (0.3). Consider:")
        print(f"   - More training epochs")
        print(f"   - Larger dataset")
        print(f"   - Higher d_model")
        print(f"   - Real labeled data")
    
    return 0 if eval_metrics.coherence_semantic_correlation > 0.3 else 1


if __name__ == "__main__":
    sys.exit(main())
