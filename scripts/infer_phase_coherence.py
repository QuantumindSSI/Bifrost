#!/usr/bin/env python3
"""
Inference script for phase coherence model.

Loads a trained checkpoint and runs inference on audio files.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bifrost.pipeline import BifrostPipeline


def load_model(checkpoint_path: str, device: str = "cuda") -> BifrostPipeline:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    d_model = checkpoint.get("d_model", 128)
    pipeline = BifrostPipeline(
        d_model=d_model,
        n_fft=1024,
        use_harmonic_binding=False,
        coherence_dim=64,
    )
    pipeline.load_state_dict(checkpoint["model_state_dict"])
    pipeline = pipeline.to(device)
    pipeline.eval()
    
    return pipeline


def run_inference(pipeline: BifrostPipeline, audio_path: str, device: str = "cuda"):
    """Run inference on audio file."""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Move to device
    waveform = waveform.to(device)
    
    # Run through pipeline
    with torch.no_grad():
        bound, coherence = pipeline(waveform)
    
    # Compute metrics
    attn_weights = torch.softmax(coherence / 1.0, dim=-1)
    diag = torch.diagonal(attn_weights, dim1=-2, dim2=-1).mean().item()
    mask = ~torch.eye(attn_weights.shape[-1], dtype=torch.bool, device=attn_weights.device)
    off_diag = attn_weights[..., mask].mean().item()
    diag_ratio = diag / (off_diag + 1e-8)
    
    return {
        "coherence": coherence,
        "bound": bound,
        "diag_ratio": diag_ratio,
        "variance": attn_weights.var().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run phase coherence inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--input", required=True, help="Path to audio file")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    pipeline = load_model(args.checkpoint, args.device)
    
    print(f"Running inference on {args.input}...")
    results = run_inference(pipeline, args.input, args.device)
    
    print("\n" + "="*50)
    print("Inference Results")
    print("="*50)
    print(f"Diagonal ratio: {results['diag_ratio']:.4f}")
    print(f"Coherence variance: {results['variance']:.6f}")
    print(f"Coherence shape: {results['coherence'].shape}")
    
    if results['diag_ratio'] > 1.2:
        print("✓ Phase coherence detected")
    else:
        print("✗ Low phase coherence")
    
    print("="*50)


if __name__ == "__main__":
    main()
