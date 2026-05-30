#!/usr/bin/env python
"""
Multimodal Spectral Encoding Training Script.

Fully implements end-to-end training for audio, text, and image modalities
producing unified SpectralTensor representations.

Usage:
    python scripts/train_multimodal_spectral.py --epochs 10 --batch-size 8
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

from bifrost import BifrostPipeline, BifrostTrainer, SpectralTensor
from bifrost.ingest import IngestPipeline, Modality


def text_to_waveform(tokens: torch.Tensor, d_model: int = 128) -> torch.Tensor:
    """
    Convert text token IDs to pseudo-waveform for spectral processing.
    
    Args:
        tokens: (B, T) token IDs
        d_model: embedding dimension
        
    Returns:
        (B, T * d_model) 1D waveform-like signal
    """
    B, T = tokens.shape
    embed = nn.Embedding(256, d_model)
    embedded = embed(tokens)  # (B, T, d_model)
    # Flatten to 1D per sample
    return embedded.view(B, -1)


def image_to_1d(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images to 1D signal for spectral processing.
    
    Args:
        images: (B, C, H, W) or (B, H, W) image tensor
        
    Returns:
        (B, C*H*W) 1D signal
    """
    B = images.size(0)
    return images.view(B, -1)


def generate_synthetic_audio_batch(batch_size: int, duration: float = 2.0, sr: int = 16000) -> torch.Tensor:
    """Generate synthetic harmonic audio signals."""
    length = int(duration * sr)
    t = torch.linspace(0, duration, length)
    
    batch = []
    for _ in range(batch_size):
        # Harmonic signal: fundamental + overtones
        f0 = torch.rand(1).item() * 200 + 100  # 100-300 Hz fundamental
        signal = torch.sin(2 * torch.pi * f0 * t)
        # Add harmonics
        for h in [2, 3, 4]:
            signal += 0.3 * torch.sin(2 * torch.pi * f0 * h * t + torch.rand(1).item())
        # Add slight noise
        signal += 0.05 * torch.randn(length)
        batch.append(signal)
    
    return torch.stack(batch)  # (B, L)


def generate_synthetic_text_batch(batch_size: int, max_length: int = 256) -> torch.Tensor:
    """Generate synthetic text-like token sequences."""
    lengths = torch.randint(max_length // 2, max_length, (batch_size,))
    max_len = lengths.max().item()
    
    batch = []
    for length in lengths:
        # Generate pseudo-text: alternating patterns like real language
        tokens = torch.zeros(max_len, dtype=torch.long)
        tokens[:length] = torch.randint(0, 256, (length,))
        batch.append(tokens)
    
    return torch.stack(batch)  # (B, T)


def generate_synthetic_image_batch(batch_size: int, size: int = 64) -> torch.Tensor:
    """Generate synthetic images with structure."""
    batch = []
    for _ in range(batch_size):
        # Structured image: gradients + patterns
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Create patterned image
        channel1 = torch.sin(3 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
        channel2 = torch.exp(-(xx**2 + yy**2))
        channel3 = torch.tanh(xx + yy)
        
        image = torch.stack([channel1, channel2, channel3])  # (3, H, W)
        batch.append(image)
    
    return torch.stack(batch)  # (B, 3, H, W)


def validate_multimodal_pipeline(pipeline: BifrostPipeline, device: str = "cuda") -> Dict[str, bool]:
    """
    Validate that pipeline produces SpectralTensor for each modality.
    
    Returns dict with validation results per modality.
    """
    results = {}
    pipeline.eval()
    
    with torch.no_grad():
        # Test Audio
        try:
            audio = generate_synthetic_audio_batch(2).to(device)
            canonical_audio = pipeline.canonicalizer(audio)
            assert isinstance(canonical_audio, SpectralTensor), "Audio: Not SpectralTensor"
            assert canonical_audio.amplitude.shape[0] == 2, "Audio: Batch size mismatch"
            results['audio'] = True
            print(f"  ✓ Audio: {canonical_audio.amplitude.shape}")
        except Exception as e:
            results['audio'] = False
            print(f"  ✗ Audio: {e}")
        
        # Test Text
        try:
            text_tokens = generate_synthetic_text_batch(2)
            text_waveform = text_to_waveform(text_tokens).to(device)
            canonical_text = pipeline.canonicalizer(text_waveform)
            assert isinstance(canonical_text, SpectralTensor), "Text: Not SpectralTensor"
            results['text'] = True
            print(f"  ✓ Text: {canonical_text.amplitude.shape}")
        except Exception as e:
            results['text'] = False
            print(f"  ✗ Text: {e}")
        
        # Test Image
        try:
            images = generate_synthetic_image_batch(2)
            image_1d = image_to_1d(images).to(device)
            canonical_image = pipeline.canonicalizer(image_1d)
            assert isinstance(canonical_image, SpectralTensor), "Image: Not SpectralTensor"
            results['image'] = True
            print(f"  ✓ Image: {canonical_image.amplitude.shape}")
        except Exception as e:
            results['image'] = False
            print(f"  ✗ Image: {e}")
    
    return results


def train_multimodal(
    epochs: int = 10,
    batch_size: int = 8,
    d_model: int = 128,
    lr: float = 0.001,
    device: str = "cuda",
    save_dir: str = "checkpoints",
) -> None:
    """
    Train Bifrost pipeline on multimodal data.
    
    Each epoch trains on audio, text, and image batches with contrastive loss.
    """
    print("=" * 60)
    print("Multimodal Spectral Encoding Training")
    print("=" * 60)
    
    # Initialize
    print(f"\n[Config]")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  d_model: {d_model}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    
    pipeline = BifrostPipeline(
        d_model=d_model,
        use_complex_ssm=True,
    ).to(device)
    
    trainer = BifrostTrainer(
        pipeline,
        lr=lr,
        device=device,
    )
    
    # Validation
    print("\n[Validation] Testing pipeline on all modalities...")
    validation_results = validate_multimodal_pipeline(pipeline, device)
    
    if not all(validation_results.values()):
        failed = [k for k, v in validation_results.items() if not v]
        raise RuntimeError(f"Pipeline validation failed for: {failed}")
    
    print("  All modalities validated ✓")
    
    # Training
    print(f"\n[Training] Starting {epochs} epochs...")
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        epoch_losses = {
            'audio': 0.0,
            'text': 0.0,
            'image': 0.0,
        }
        
        # Train on each modality
        for modality in ['audio', 'text', 'image']:
            # Generate batch
            if modality == 'audio':
                batch = generate_synthetic_audio_batch(batch_size).to(device)
            elif modality == 'text':
                text_tokens = generate_synthetic_text_batch(batch_size)
                batch = text_to_waveform(text_tokens).to(device)
            else:  # image
                images = generate_synthetic_image_batch(batch_size)
                batch = image_to_1d(images).to(device)
            
            # Train step
            try:
                loss = trainer.train_step(batch)
                epoch_losses[modality] = loss
            except Exception as e:
                print(f"  ✗ {modality} training failed: {e}")
                raise
        
        # Report
        avg_loss = sum(epoch_losses.values()) / 3
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"audio={epoch_losses['audio']:.4f}, "
              f"text={epoch_losses['text']:.4f}, "
              f"image={epoch_losses['image']:.4f}, "
              f"avg={avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_path / f"multimodal_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'pipeline_state': pipeline.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'losses': epoch_losses,
            }, checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path}")
    
    # Final validation
    print("\n[Final Validation] Extracting spectral tensors...")
    pipeline.eval()
    with torch.no_grad():
        # Audio
        audio = generate_synthetic_audio_batch(1).to(device)
        audio_canonical = pipeline.canonicalizer(audio)
        print(f"  Audio spectral tensor: {audio_canonical.amplitude.shape}")
        
        # Text
        text_tokens = generate_synthetic_text_batch(1)
        text_waveform = text_to_waveform(text_tokens).to(device)
        text_canonical = pipeline.canonicalizer(text_waveform)
        print(f"  Text spectral tensor: {text_canonical.amplitude.shape}")
        
        # Image
        images = generate_synthetic_image_batch(1)
        image_1d = image_to_1d(images).to(device)
        image_canonical = pipeline.canonicalizer(image_1d)
        print(f"  Image spectral tensor: {image_canonical.amplitude.shape}")
    
    print("\n" + "=" * 60)
    print("Training Complete ✓")
    print("=" * 60)
    print(f"Pipeline can now encode audio/text/image → SpectralTensor")
    print(f"Checkpoints saved to: {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train multimodal spectral encoding")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per modality")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    train_multimodal(
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
