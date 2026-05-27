#!/usr/bin/env python3
"""
Train FBC pipeline for phase coherence on sample audio.

This script demonstrates training the dual-stream SpectralDecomposer
to learn temporal phase coherence patterns via next-frame prediction.

Usage:
    python scripts/train_phase_coherence.py --epochs 50 --device cuda
"""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from fbc import FBCPipeline, FBCTrainer, train_fbc_simple
from fbc.data.loader import load_sample_audio


def create_dummy_dataloader(batch_size: int = 4, num_batches: int = 10) -> DataLoader:
    """
    Create a dummy dataloader for demonstration.

    In production, replace with actual audio dataset.
    """
    # Load sample audio and create synthetic sequences
    audio, sr = load_sample_audio("speech_synth")

    # Create multiple sequences by adding small variations
    sequences = []
    for i in range(batch_size * num_batches):
        # Add slight noise/variation
        noise = torch.randn_like(audio) * 0.01
        seq = audio + noise
        sequences.append(seq)

    # Stack into batch tensor (B, L)
    data = torch.stack(sequences)

    # Create dataset and dataloader
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    parser = argparse.ArgumentParser(description="Train FBC phase coherence")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--save-path", type=str, default="fbc_checkpoint.pt", help="Checkpoint path")
    args = parser.parse_args()

    print("=" * 60)
    print("FBC Phase Coherence Training")
    print("=" * 60)

    # Create pipeline with dual-stream decomposer
    print(f"\nCreating pipeline (d_model={args.d_model})...")
    pipeline = FBCPipeline(
        n_fft_s0=1024,
        n_fft_s1=512,
        d_model=args.d_model,
        n_heads=4,
        n_bands=8,
        use_mamba=True,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloader
    print(f"\nCreating dataloader (batch_size={args.batch_size})...")
    dataloader = create_dummy_dataloader(batch_size=args.batch_size)

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    trainer = train_fbc_simple(
        pipeline=pipeline,
        dataloader=dataloader,
        epochs=args.epochs,
        device=device,
    )

    # Save checkpoint
    print(f"\nSaving checkpoint to {args.save_path}...")
    trainer.save_checkpoint(args.save_path)

    # Final evaluation
    print("\nFinal evaluation:")
    print("-" * 60)
    batch = next(iter(dataloader))
    signal = batch[0] if isinstance(batch, (list, tuple)) else batch
    stats = trainer.eval_step(signal)

    print(f"Loss: {stats['loss']:.4f}")
    print(f"Coherence mean: {stats['coherence_mean']:.4f}")
    print(f"Coherence std: {stats['coherence_std']:.4f}")
    print("\nPer-head self-attention ratios (higher = more coherent):")
    for h in range(4):
        ratio = stats.get(f"head_{h}_ratio", 0)
        bar = "█" * int(ratio * 10)
        print(f"  Head {h}: {ratio:.3f} {bar}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoint saved to: {args.save_path}")
    print("=" * 60)

    # Expected: After training, head ratios should be > 1.2 (vs ~1.06 untrained)
    avg_ratio = sum(stats.get(f"head_{h}_ratio", 1.0) for h in range(4)) / 4
    if avg_ratio > 1.15:
        print("✅ Phase coherence successfully learned!")
    else:
        print("⚠️  Phase coherence still weak. More training epochs may be needed.")


if __name__ == "__main__":
    main()
