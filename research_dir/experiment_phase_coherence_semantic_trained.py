#!/usr/bin/env python3
"""
Experiment 1b: Trained phase-coherence representation correlates with semantic category.

This is an improved version of the first experiment. Instead of using an
untrained Bifrost pipeline, we train the pipeline end-to-end on a synthetic
audio classification task. The question is whether the trained Bifrost
embeddings cluster by semantic category better than the untrained baseline.

Hypothesis (from hypothesis registry):
    Bifrost phase coherence correlates with semantic category similarity on real audio.

This experiment tests the weaker but necessary precondition:
    A Bifrost pipeline trained on synthetic semantic categories learns embeddings
    that correlate with category labels.

Metric:
    Pearson correlation between pairwise embedding similarity and category co-occurrence.
    Also reports classification accuracy.

Target:
    r > 0.3, p < 0.05; accuracy > 0.6

Usage:
    python research_dir/experiment_phase_coherence_semantic_trained.py \
        --n_samples 500 --epochs 50 --lr 1e-3 --batch_size 16

Output:
    Saves results to:
        research_dir/results/phase_coherence_semantic_correlation_trained.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESEARCH_DIR = Path(__file__).parent
RESULTS_DIR = RESEARCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    dataset: str = "synthetic"  # synthetic | esc50
    n_samples: int = 500
    max_per_class: Optional[int] = None
    duration_seconds: float = 1.0
    sample_rate: int = 16000
    n_fft: int = 1024
    d_model: int = 64
    n_heads: int = 4
    n_bands: int = 8
    n_classes: int = 5
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.0
    frozen_pipeline: bool = False
    device: str = "cpu"
    train_frac: float = 0.8


# -----------------------------------------------------------------------------
# Synthetic dataset (same classes as the untrained experiment)
# -----------------------------------------------------------------------------

def generate_synthetic_dataset(config: TrainConfig) -> Tuple[List[torch.Tensor], List[int]]:
    """Generate synthetic audio with 5 semantic categories."""
    n_per_class = config.n_samples // config.n_classes
    duration = config.duration_seconds
    sample_rate = config.sample_rate
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
                    sig += (1.0 / k) * torch.sin(
                        2 * math.pi * k * f0 * t + random.uniform(0, 2 * math.pi)
                    )
            elif name == "noisy":
                sig = torch.randn_like(t)
            else:
                freq = base_freq * random.uniform(0.9, 1.1)
                sig = torch.sin(2 * math.pi * freq * t + random.uniform(0, 2 * math.pi))
            sig = sig + 0.1 * torch.randn_like(sig)
            sig = sig / sig.abs().max().clamp_min(1e-8)
            signals.append(sig)
            labels.append(class_idx)

    return signals, labels


def load_speechcommands_dataset(config: TrainConfig) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Load Google SpeechCommands via torchaudio, restricting to a subset of classes."""
    try:
        import torchaudio
        from torchaudio.datasets import SPEECHCOMMANDS
    except ImportError as e:
        raise ImportError(
            "torchaudio is required for SpeechCommands. Install with: pip install torchaudio"
        ) from e

    root = Path(__file__).parent.parent / "data" / "speechcommands"
    root.mkdir(parents=True, exist_ok=True)

    dataset = SPEECHCOMMANDS(root=str(root), download=True, subset="training")

    # Collect all samples and labels
    all_signals: List[torch.Tensor] = []
    all_labels_str: List[str] = []

    for waveform, sr, label, *_ in dataset:
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
        all_signals.append(waveform.squeeze(0))
        all_labels_str.append(str(label))

    # Select the n most frequent classes
    from collections import Counter
    label_counts = Counter(all_labels_str)
    selected_label_strs = [lab for lab, _ in label_counts.most_common(config.n_classes)]
    selected_label_strs = sorted(selected_label_strs)
    class_names = selected_label_strs
    label_to_idx = {lab: idx for idx, lab in enumerate(class_names)}

    signals = [s for s, y in zip(all_signals, all_labels_str) if y in label_to_idx]
    labels = [label_to_idx[y] for y in all_labels_str if y in label_to_idx]

    # Limit per-class samples if needed
    if config.max_per_class is not None:
        limited_signals: List[torch.Tensor] = []
        limited_labels: List[int] = []
        for c in range(len(class_names)):
            c_signals = [s for s, y in zip(signals, labels) if y == c]
            c_signals = c_signals[: config.max_per_class]
            limited_signals.extend(c_signals)
            limited_labels.extend([c] * len(c_signals))
        signals, labels = limited_signals, limited_labels

    return signals, labels, class_names


# -----------------------------------------------------------------------------
# Bifrost classification model
# -----------------------------------------------------------------------------

def build_pipeline(config: TrainConfig):
    """Build a lightweight BifrostPipeline for this experiment."""
    import sys
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.bifrost.pipeline import BifrostPipeline

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
    ).to(config.device)


class BifrostClassifier(nn.Module):
    """Bifrost pipeline + temporal pooling + linear classifier."""

    def __init__(self, pipeline, d_model: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pipeline = pipeline
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model * 2, n_classes)

    def forward(self, signal: torch.Tensor, sample_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        metadata = {"sample_rate": float(sample_rate)}
        bound_st, _ = self.pipeline(signal, metadata=metadata)
        amp = bound_st.amplitude  # (B, T, d_model)
        # Temporal mean and std pooling
        emb = torch.cat([amp.mean(dim=1), amp.std(dim=1)], dim=-1)  # (B, d_model*2)
        emb = self.dropout(emb)
        logits = self.classifier(emb)
        return logits, emb


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def extract_embedding_flat(bound_st) -> torch.Tensor:
    """Flatten a bound SpectralTensor into a normalized vector."""
    amp = bound_st.amplitude
    phase = bound_st.phase
    if amp.dim() == 3:
        amp = amp.mean(dim=1)
        phase = phase.mean(dim=1)
    emb = torch.cat([amp.flatten(), torch.cos(phase).flatten(), torch.sin(phase).flatten()], dim=0)
    emb = emb / emb.norm().clamp_min(1e-8)
    return emb


def train_model(
    model: BifrostClassifier,
    train_signals: List[torch.Tensor],
    train_labels: List[int],
    config: TrainConfig,
) -> List[float]:
    """Train the classifier on synthetic or real data."""
    if config.frozen_pipeline:
        model.pipeline.eval()
        for param in model.pipeline.parameters():
            param.requires_grad = False
        trainable = model.classifier.parameters()
    else:
        for param in model.pipeline.parameters():
            param.requires_grad = True
        trainable = model.parameters()

    optimizer = torch.optim.Adam(trainable, lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    n = len(train_signals)
    losses = []

    model.train()
    for epoch in range(config.epochs):
        if config.frozen_pipeline:
            model.pipeline.eval()

        indices = list(range(n))
        random.shuffle(indices)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, n, config.batch_size):
            batch_idx = indices[i : i + config.batch_size]
            batch_signals = torch.stack([train_signals[j] for j in batch_idx]).to(config.device)
            batch_labels = torch.tensor([train_labels[j] for j in batch_idx], dtype=torch.long).to(config.device)

            logits, _ = model(batch_signals, config.sample_rate)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{config.epochs}: loss = {avg_loss:.4f}")

    return losses


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate(
    model: BifrostClassifier,
    signals: List[torch.Tensor],
    labels: List[int],
    config: TrainConfig,
) -> Dict:
    """Evaluate classification accuracy and embedding correlation."""
    model.eval()
    embeddings: List[torch.Tensor] = []
    preds: List[int] = []

    with torch.no_grad():
        for sig in signals:
            x = sig.unsqueeze(0).to(config.device)
            logits, emb = model(x, config.sample_rate)
            embeddings.append(emb.squeeze(0).cpu())
            preds.append(int(logits.argmax(dim=-1).item()))

    # Classification accuracy
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    accuracy = correct / len(labels)

    # Pairwise similarity and semantic similarity
    emb_matrix = torch.stack(embeddings)
    sim_matrix = emb_matrix @ emb_matrix.T
    label_tensor = torch.tensor(labels)
    sem_matrix = (label_tensor.unsqueeze(0) == label_tensor.unsqueeze(1)).float()

    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((i, j))

    similarities = torch.tensor([sim_matrix[i, j].item() for i, j in pairs])
    sem_sims = torch.tensor([sem_matrix[i, j].item() for i, j in pairs])

    # Pearson correlation
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(similarities.numpy(), sem_sims.numpy())
    except ImportError:
        x = similarities - similarities.mean()
        y = sem_sims - sem_sims.mean()
        num = (x * y).sum()
        den = torch.sqrt((x ** 2).sum() * (y ** 2).sum())
        r = (num / den.clamp_min(1e-8)).item()
        p = float("nan")

    return {
        "accuracy": accuracy,
        "pearson_r": float(r),
        "p_value": float(p),
        "n_pairs": len(pairs),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_experiment(config: TrainConfig) -> Dict:
    print("\nExperiment: phase_coherence_semantic_correlation (trained)")
    print(f"Dataset: {config.dataset}")
    print(f"Classes: {config.n_classes}, Duration: {config.duration_seconds}s")
    print(f"Train split: {config.train_frac}")
    print(f"Epochs: {config.epochs}, LR: {config.lr}, Batch size: {config.batch_size}")
    print("-" * 60)

    if config.dataset == "synthetic":
        print("Generating synthetic dataset...")
        signals, labels = generate_synthetic_dataset(config)
    elif config.dataset == "speechcommands":
        print("Loading SpeechCommands dataset...")
        signals, labels, _ = load_speechcommands_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    print(f"Loaded {len(signals)} samples across {config.n_classes} classes")

    # Train/test split
    n = len(signals)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * config.train_frac)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_signals = [signals[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_signals = [signals[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"Train: {len(train_signals)}, Test: {len(test_signals)}")

    print("Building model...")
    pipeline = build_pipeline(config)
    model = BifrostClassifier(pipeline, config.d_model, config.n_classes, dropout=config.dropout).to(config.device)
    if config.frozen_pipeline:
        print("Pipeline frozen; only classifier head will be trained.")

    print("Training...")
    train_losses = train_model(model, train_signals, train_labels, config)

    print("\nEvaluating on train set...")
    train_metrics = evaluate(model, train_signals, train_labels, config)
    print(f"  Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Train Pearson r: {train_metrics['pearson_r']:.4f}, p = {train_metrics['p_value']:.4e}")

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_signals, test_labels, config)
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Pearson r: {test_metrics['pearson_r']:.4f}, p = {test_metrics['p_value']:.4e}")

    results = {
        "experiment_id": "phase_coherence_semantic_correlation_trained",
        "dataset": "synthetic",
        "n_samples": config.n_samples,
        "n_classes": config.n_classes,
        "train_split": config.train_frac,
        "epochs": config.epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "train_losses": train_losses,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "target_accuracy": 0.6,
        "target_r": 0.3,
        "verdict": "supported"
        if test_metrics["accuracy"] > 0.6 and test_metrics["pearson_r"] > 0.3
        else "inconclusive",
    }

    suffix = "frozen" if config.frozen_pipeline else "finetuned"
    result_path = RESULTS_DIR / f"phase_coherence_semantic_correlation_trained_{config.dataset}_{suffix}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "speechcommands"], default="synthetic")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--frozen_pipeline", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = TrainConfig(
        dataset=args.dataset,
        n_samples=args.n_samples,
        max_per_class=args.max_per_class,
        duration_seconds=args.duration,
        d_model=args.d_model,
        n_classes=args.n_classes,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        frozen_pipeline=args.frozen_pipeline,
        batch_size=args.batch_size,
        device=args.device,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
