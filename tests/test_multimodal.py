#!/usr/bin/env python3
"""
Test all four modalities (audio, image, text, tensor) through FBC pipeline.

Usage:
    python tests/test_multimodal.py
"""

import torch
import numpy as np

from fbc import create_multimodal_pipeline, Modality


def test_audio():
    """Test audio modality (1D temporal signal)."""
    print("\n" + "=" * 60)
    print("Testing AUDIO modality")
    print("=" * 60)

    # Create synthetic audio (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # Create harmonic signal (440Hz fundamental + 880Hz harmonic)
    audio = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 880 * t)
    audio = audio.unsqueeze(0)  # Add batch dimension

    print(f"Input shape: {audio.shape}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} sec")

    # Create pipeline
    pipeline = create_multimodal_pipeline(
        modality="audio",
        n_fft=1024,
        d_model=128,
        n_heads=4,
        use_mamba=False,  # Use PyTorch fallback for testing
    )

    # Process
    bound, coherence = pipeline(audio, {"sample_rate": sample_rate})

    print(f"\nOutput amplitude shape: {bound.amplitude.shape}")
    print(f"Output phase shape: {bound.phase.shape}")
    print(f"Coherence shape: {coherence.shape}")

    # Check coherence statistics
    print(f"\nCoherence stats:")
    print(f"  Mean: {coherence.mean():.4f}")
    print(f"  Std: {coherence.std():.4f}")
    print(f"  Min: {coherence.min():.4f}")
    print(f"  Max: {coherence.max():.4f}")

    # Diagonal vs off-diagonal for first head
    head0 = coherence[0, 0]  # (T, T)
    diag = head0.diagonal().mean().item()
    offdiag = head0[~torch.eye(head0.shape[0], dtype=torch.bool)].mean().item()
    print(f"\nHead 0 diagonal/off-diag ratio: {diag/offdiag:.3f}")

    print("\n✅ AUDIO: PASSED")
    return True


def test_image():
    """Test image modality (2D spatial data)."""
    print("\n" + "=" * 60)
    print("Testing IMAGE modality")
    print("=" * 60)

    # Create synthetic image (grayscale with horizontal stripes)
    H, W = 256, 256
    image = torch.zeros(1, H, W)
    for i in range(0, H, 16):
        image[0, i:i+8, :] = 1.0  # Horizontal stripes

    print(f"Input shape: {image.shape}")
    print(f"Image size: {H}x{W}")

    # Create pipeline
    pipeline = create_multimodal_pipeline(
        modality="image",
        n_fft=512,
        d_model=128,
        n_heads=4,
        use_2d_fft=True,
        use_mamba=False,
    )

    # Process
    bound, coherence = pipeline(image, {"height": H, "width": W})

    print(f"\nOutput amplitude shape: {bound.amplitude.shape}")
    print(f"Output phase shape: {bound.phase.shape}")
    print(f"Coherence shape: {coherence.shape}")

    print(f"\nCoherence stats:")
    print(f"  Mean: {coherence.mean():.4f}")
    print(f"  Std: {coherence.std():.4f}")

    print("\n✅ IMAGE: PASSED")
    return True


def test_text():
    """Test text modality (token sequences)."""
    print("\n" + "=" * 60)
    print("Testing TEXT modality")
    print("=" * 60)

    # Create synthetic token sequence (batch_size=1, seq_len=128)
    # Tokens are random vocabulary indices
    seq_len = 128
    vocab_size = 50000
    tokens = torch.randint(0, vocab_size, (1, seq_len))

    print(f"Input shape: {tokens.shape}")
    print(f"Sequence length: {seq_len}")
    print(f"Vocabulary size: {vocab_size}")

    # Create pipeline
    pipeline = create_multimodal_pipeline(
        modality="text",
        n_fft=512,
        d_model=128,
        n_heads=4,
        use_mamba=False,
    )

    # Process
    bound, coherence = pipeline(tokens)

    print(f"\nOutput amplitude shape: {bound.amplitude.shape}")
    print(f"Output phase shape: {bound.phase.shape}")
    print(f"Coherence shape: {coherence.shape}")

    print(f"\nCoherence stats:")
    print(f"  Mean: {coherence.mean():.4f}")
    print(f"  Std: {coherence.std():.4f}")

    print("\n✅ TEXT: PASSED")
    return True


def test_tensor():
    """Test tensor modality (arbitrary numeric tensor)."""
    print("\n" + "=" * 60)
    print("Testing TENSOR modality")
    print("=" * 60)

    # Create synthetic tensor (e.g., sensor readings, embeddings, etc.)
    # Shape: (batch=2, features=10, channels=8)
    tensor = torch.randn(2, 10, 8)

    print(f"Input shape: {tensor.shape}")
    print(f"Total elements: {tensor.numel()}")

    # Create pipeline
    pipeline = create_multimodal_pipeline(
        modality="tensor",
        n_fft=1024,
        d_model=128,
        n_heads=4,
        use_mamba=False,
    )

    # Process
    bound, coherence = pipeline(tensor)

    print(f"\nOutput amplitude shape: {bound.amplitude.shape}")
    print(f"Output phase shape: {bound.phase.shape}")
    print(f"Coherence shape: {coherence.shape}")

    print(f"\nCoherence stats:")
    print(f"  Mean: {coherence.mean():.4f}")
    print(f"  Std: {coherence.std():.4f}")

    print("\n✅ TENSOR: PASSED")
    return True


def main():
    """Run all modality tests."""
    print("\n" + "=" * 70)
    print("FBC MULTIMODAL PIPELINE TEST SUITE")
    print("=" * 70)

    results = {}

    try:
        results["audio"] = test_audio()
    except Exception as e:
        print(f"\n❌ AUDIO: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results["audio"] = False

    try:
        results["image"] = test_image()
    except Exception as e:
        print(f"\n❌ IMAGE: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results["image"] = False

    try:
        results["text"] = test_text()
    except Exception as e:
        print(f"\n❌ TEXT: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results["text"] = False

    try:
        results["tensor"] = test_tensor()
    except Exception as e:
        print(f"\n❌ TENSOR: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results["tensor"] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for modality, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {modality.upper():10s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Multimodal FBC is fully operational.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
