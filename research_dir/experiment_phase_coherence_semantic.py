#!/usr/bin/env python3
"""
Experiment 1: Phase-coherence representation correlates with semantic category.

This is the first real-data validation experiment in the Bifrost research loop.
It tests whether Bifrost spectral embeddings cluster by semantic category on a
real audio dataset (ESC-50) or a synthetic fallback dataset.

Hypothesis (from hypothesis registry):
    Bifrost phase coherence correlates with semantic category similarity on real audio.

Metric:
    Pearson correlation between pairwise embedding similarity and category co-occurrence.

Target:
    r > 0.3, p < 0.05 (weak positive baseline) as a first validation step.

Usage:
    # Requires project installed
    pip install -e ".[dev]"

    # Run with ESC-50 (auto-downloads if torchaudio >= 2.1 is available)
    python research_dir/experiment_phase_coherence_semantic.py --dataset esc50 --root ./data/esc50

    # Run with synthetic fallback (no external data)
    python research_dir/experiment_phase_coherence_semantic.py --dataset synthetic --n_samples 200

    # Run with a local folder of WAV files organized by category
    python research_dir/experiment_phase_coherence_semantic.py --dataset local --root ./my_audio

Output:
    Prints correlation, p-value, and saves results to:
        research_dir/results/phase_coherence_semantic_correlation.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESEARCH_DIR = Path(__file__).parent
RESULTS_DIR = RESEARCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    dataset: str = "synthetic"  # esc50 | synthetic | local
    root: Optional[str] = None
    n_samples: int = 200
    sample_rate: int = 16000
    duration_seconds: float = 1.0
    n_fft: int = 1024
    d_model: int = 128
    n_heads: int = 4
    n_bands: int = 8
    device: str = "cpu"
    max_pairs: int = 5000  # cap for pairwise comparison


# -----------------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------------

def load_esc50(config: ExperimentConfig) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Load ESC-50 via torchaudio. Falls back to synthetic if unavailable.

    Returns:
        signals: list of (1, n_samples) float32 tensors
        labels: list of integer category labels
        classes: list of class names
    """
    try:
        import torchaudio
        from torchaudio.datasets import ESC50
    except ImportError as e:
        raise ImportError(
            "torchaudio is required for ESC-50. Install with: pip install torchaudio"
        ) from e

    root = Path(config.root or "./data/esc50")
    root.mkdir(parents=True, exist_ok=True)

    try:
        dataset = ESC50(root=str(root), download=True, subset="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load ESC-50 from {root}: {e}") from e

    # ESC-50 has 50 classes. Use a subset to keep compute manageable.
    signals: List[torch.Tensor] = []
    labels: List[int] = []

    classes = list(range(50))
    class_names = [f"class_{i}" for i in range(50)]

    for waveform, sample_rate, label, _ in dataset:
        # waveform shape: (channels, time)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sample_rate != config.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, config.sample_rate)
            waveform = resampler(waveform)
        # Trim/pad to target length
        target_len = int(config.duration_seconds * config.sample_rate)
        if waveform.shape[-1] > target_len:
            waveform = waveform[..., :target_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
        signals.append(waveform.squeeze(0))  # (n_samples,)
        labels.append(int(label))

    return signals, labels, class_names


def load_synthetic(config: ExperimentConfig) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Generate synthetic audio with 5 semantic categories defined by frequency structure.

    Categories:
        0: Low-frequency tones
        1: Mid-frequency tones
        2: High-frequency tones
        3: Harmonic stacks
        4: Inharmonic / noisy
    """
    n_samples = config.n_samples
    n_per_class = n_samples // 5
    sample_rate = config.sample_rate
    duration = config.duration_seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))

    signals: List[torch.Tensor] = []
    labels: List[int] = []

    class_defs = [
        ("low_tone", 200.0),
        ("mid_tone", 800.0),
        ("high_tone", 2400.0),
        ("harmonic_stack", None),
        ("noisy", None),
    ]

    for class_idx, (name, base_freq) in enumerate(class_defs):
        for _ in range(n_per_class):
            if name == "harmonic_stack":
                f0 = random.uniform(150.0, 400.0)
                sig = torch.zeros_like(t)
                for k in range(1, 5):
                    sig += (1.0 / k) * torch.sin(2 * math.pi * k * f0 * t + random.uniform(0, 2 * math.pi))
            elif name == "noisy":
                sig = torch.randn_like(t)
            else:
                freq = base_freq * random.uniform(0.9, 1.1)
                sig = torch.sin(2 * math.pi * freq * t + random.uniform(0, 2 * math.pi))
            # Add mild noise
            sig = sig + 0.1 * torch.randn_like(sig)
            sig = sig / sig.abs().max().clamp_min(1e-8)
            signals.append(sig)
            labels.append(class_idx)

    return signals, labels, [name for name, _ in class_defs]


def load_local_folder(config: ExperimentConfig) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Load WAV files from a local folder organized as:
        root/
            class_a/
                file1.wav
                file2.wav
            class_b/
                ...
    """
    import glob
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError("torchaudio is required for local audio loading") from e

    root = Path(config.root or ".")
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {root}")

    signals: List[torch.Tensor] = []
    labels: List[int] = []
    class_names = [d.name for d in class_dirs]

    for class_idx, class_dir in enumerate(class_dirs):
        wav_files = sorted(glob.glob(str(class_dir / "*.wav")))
        for wav_path in wav_files:
            waveform, sr = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
                waveform = resampler(waveform)
            target_len = int(config.duration_seconds * config.sample_rate)
            if waveform.shape[-1] > target_len:
                waveform = waveform[..., :target_len]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
            signals.append(waveform.squeeze(0))
            labels.append(class_idx)

    return signals, labels, class_names


def load_dataset(config: ExperimentConfig) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    if config.dataset == "esc50":
        return load_esc50(config)
    if config.dataset == "synthetic":
        return load_synthetic(config)
    if config.dataset == "local":
        return load_local_folder(config)
    raise ValueError(f"Unknown dataset: {config.dataset}")


# -----------------------------------------------------------------------------
# Bifrost embedding
# -----------------------------------------------------------------------------

def build_bifrost_pipeline(config: ExperimentConfig):
    """Build a lightweight BifrostPipeline for this experiment."""
    import sys
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.bifrost.pipeline import BifrostPipeline

    # Use complex SSM for true phase coherence; disable optional heavy modules
    return BifrostPipeline(
        n_fft_canonical=config.n_fft,
        n_fft_decompose=512,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_bands=config.n_bands,
        dropout=0.0,
        use_complex_ssm=True,
        use_s3_attractor=False,
        use_riemannian_semantic=False,
        preserve_frames=True,
    ).to(config.device).eval()


def extract_embedding(signal: torch.Tensor, pipeline, config: ExperimentConfig) -> torch.Tensor:
    """
    Extract a flat embedding vector from Bifrost for one signal.

    Strategy:
        1. Run signal through BifrostPipeline.
        2. Use the bound spectral amplitude + phase as the representation.
        3. Flatten to a vector.
    """
    with torch.no_grad():
        # Ensure shape (1, n_samples)
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        signal = signal.to(config.device)
        metadata = {"sample_rate": float(config.sample_rate)}
        bound_st, coherence = pipeline(signal, metadata=metadata)

        # Use mean-pooled amplitude and phase as a compact embedding
        amp = bound_st.amplitude
        phase = bound_st.phase

        # Handle possible frame dimension
        if amp.dim() == 3:
            amp = amp.mean(dim=1)
            phase = phase.mean(dim=1)

        # Create a real-valued embedding from amplitude and phase
        emb = torch.cat([amp.flatten(), torch.cos(phase).flatten(), torch.sin(phase).flatten()], dim=0)
        emb = emb.cpu().float()

        # Normalize
        norm = emb.norm().clamp_min(1e-8)
        emb = emb / norm

    return emb


# -----------------------------------------------------------------------------
# Similarity and correlation
# -----------------------------------------------------------------------------

def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Compute Pearson r and two-tailed p-value using scipy if available."""
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x.numpy(), y.numpy())
        return float(r), float(p)
    except ImportError:
        # Fallback: compute r manually; p-value not available
        x = x - x.mean()
        y = y - y.mean()
        num = (x * y).sum()
        den = torch.sqrt((x ** 2).sum() * (y ** 2).sum())
        r = (num / den.clamp_min(1e-8)).item()
        return r, float("nan")


def compute_pairwise_similarities(
    embeddings: List[torch.Tensor],
    labels: List[int],
    max_pairs: int = 5000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample pairs and compute embedding cosine similarity and semantic similarity.

    Returns:
        similarities: (n_pairs,) tensor of embedding cosine similarities
        sem_sims: (n_pairs,) tensor of 1.0 if same class else 0.0
    """
    n = len(embeddings)
    emb_matrix = torch.stack(embeddings)  # (n, d)

    # Cosine similarity matrix
    sim_matrix = emb_matrix @ emb_matrix.T  # (n, n)

    # Build semantic similarity matrix
    label_tensor = torch.tensor(labels)
    sem_matrix = (label_tensor.unsqueeze(0) == label_tensor.unsqueeze(1)).float()

    # Sample pairs from upper triangle
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    if len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]

    similarities = torch.tensor([sim_matrix[i, j].item() for i, j in pairs])
    sem_sims = torch.tensor([sem_matrix[i, j].item() for i, j in pairs])

    return similarities, sem_sims


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> Dict:
    print(f"\nExperiment: phase_coherence_semantic_correlation")
    print(f"Dataset: {config.dataset}")
    print(f"Device: {config.device}")
    print(f"Samples: {config.n_samples}")
    print("-" * 60)

    print("Loading dataset...")
    signals, labels, class_names = load_dataset(config)
    print(f"Loaded {len(signals)} signals across {len(class_names)} classes")

    print("Building Bifrost pipeline...")
    pipeline = build_bifrost_pipeline(config)
    print("Pipeline built.")

    print("Extracting Bifrost embeddings...")
    embeddings = []
    for idx, signal in enumerate(signals):
        emb = extract_embedding(signal, pipeline, config)
        embeddings.append(emb)
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx + 1}/{len(signals)}")

    print("Computing pairwise similarities...")
    similarities, sem_sims = compute_pairwise_similarities(
        embeddings, labels, max_pairs=config.max_pairs
    )

    print(f"Pair count: {similarities.shape[0]}")

    r, p = pearson_correlation(similarities, sem_sims)
    print(f"\nPearson correlation: r = {r:.4f}")
    print(f"p-value: {p:.4e}")

    # Compute per-class mean intra-class similarity
    label_tensor = torch.tensor(labels)
    intra_class_sims = {}
    for c in range(len(class_names)):
        idxs = (label_tensor == c).nonzero(as_tuple=True)[0].tolist()
        if len(idxs) < 2:
            continue
        embs_c = torch.stack([embeddings[i] for i in idxs])
        sim_c = embs_c @ embs_c.T
        mask = torch.triu(torch.ones_like(sim_c), diagonal=1).bool()
        intra_class_sims[class_names[c]] = float(sim_c[mask].mean())

    results = {
        "experiment_id": "phase_coherence_semantic_correlation",
        "dataset": config.dataset,
        "n_samples": len(signals),
        "n_classes": len(class_names),
        "n_pairs": similarities.shape[0],
        "pearson_r": r,
        "p_value": p,
        "target_r": 0.3,
        "target_p": 0.05,
        "verdict": "supported" if r > 0.3 and p < 0.05 else "inconclusive",
        "class_names": class_names,
        "intra_class_similarities": intra_class_sims,
        "config": {
            "sample_rate": config.sample_rate,
            "duration_seconds": config.duration_seconds,
            "n_fft": config.n_fft,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_bands": config.n_bands,
        },
    }

    result_path = RESULTS_DIR / "phase_coherence_semantic_correlation.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate phase-coherence semantic correlation on real audio"
    )
    parser.add_argument(
        "--dataset",
        choices=["esc50", "synthetic", "local"],
        default="synthetic",
        help="Dataset to use",
    )
    parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument("--duration", type=float, default=1.0, help="Audio duration in seconds")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--max_pairs", type=int, default=5000, help="Max pairwise comparisons")

    args = parser.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset,
        root=args.root,
        n_samples=args.n_samples,
        duration_seconds=args.duration,
        n_fft=args.n_fft,
        d_model=args.d_model,
        device=args.device,
        max_pairs=args.max_pairs,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
