#!/usr/bin/env python3
"""
Pre-registered Experiment 2: Bifrost vs. spectral baselines on SpeechCommands.

Protocol (from EPISTEMIC_AUDIT.md):
    - Dataset: Google SpeechCommands v0.02, 10 core classes, 100 samples per class.
    - Models: Bifrost fine-tuned, Bifrost frozen, STFT baseline, mel baseline.
    - Evaluation: stratified 5-fold cross-validation.
    - Metrics: macro accuracy and F1, mean ± std across folds.
    - Statistics: paired t-test between Bifrost fine-tuned and each baseline.

Primary hypothesis (H1):
    Bifrost fine-tuned exceeds the STFT baseline by ≥ 5 absolute percentage points
    in mean test accuracy, with a statistically significant difference.

This script implements the pre-registered protocol and saves results as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)

RESEARCH_DIR = Path(__file__).parent
RESULTS_DIR = RESEARCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class BaselineConfig:
    n_classes: int = 10
    samples_per_class: int = 100
    duration_seconds: float = 1.0
    sample_rate: int = 16000
    n_fft: int = 1024
    d_model: int = 64
    n_heads: int = 4
    n_bands: int = 8
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.5
    n_folds: int = 5
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_speechcommands(config: BaselineConfig) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS

    root = RESEARCH_DIR.parent / "data" / "speechcommands"
    root.mkdir(parents=True, exist_ok=True)
    dataset = SPEECHCOMMANDS(root=str(root), download=True, subset="training")

    all_signals: List[torch.Tensor] = []
    all_labels: List[str] = []
    for waveform, sr, label, *_ in dataset:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != config.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, config.sample_rate)(waveform)
        target_len = int(config.duration_seconds * config.sample_rate)
        if waveform.shape[-1] > target_len:
            waveform = waveform[..., :target_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
        all_signals.append(waveform.squeeze(0))
        all_labels.append(str(label))

    # Select most frequent classes and cap samples per class
    counts = Counter(all_labels)
    selected = [lab for lab, _ in counts.most_common(config.n_classes)]
    selected = sorted(selected)
    label_to_idx = {lab: i for i, lab in enumerate(selected)}

    signals: List[torch.Tensor] = []
    labels: List[int] = []
    for c in selected:
        c_signals = [s for s, y in zip(all_signals, all_labels) if y == c]
        c_signals = c_signals[: config.samples_per_class]
        signals.extend(c_signals)
        labels.extend([label_to_idx[c]] * len(c_signals))

    return signals, labels, selected


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_bifrost_pipeline(config: BaselineConfig):
    import sys
    repo_root = RESEARCH_DIR.parent
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
    def __init__(self, pipeline, d_model: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pipeline = pipeline
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model * 2, n_classes)

    def forward(self, x, sample_rate):
        st, _ = self.pipeline(x, metadata={"sample_rate": float(sample_rate)})
        amp = st.amplitude
        emb = torch.cat([amp.mean(dim=1), amp.std(dim=1)], dim=-1)
        emb = self.dropout(emb)
        return self.classifier(emb)


class STFTBaseline(nn.Module):
    """Log STFT magnitude → mean-pool → linear classifier."""

    def __init__(self, n_fft: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.n_freq, n_classes)

    def forward(self, x, *_):
        # x: (B, T)
        stft = torch.stft(x, n_fft=self.n_fft, return_complex=True)
        mag = stft.abs()  # (B, n_freq, n_frames)
        mag = torch.log(mag + 1e-8)
        emb = mag.mean(dim=-1)  # (B, n_freq)
        emb = self.dropout(emb)
        return self.classifier(emb)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_x: List[torch.Tensor],
    train_y: List[int],
    config: BaselineConfig,
    model_name: str,
) -> Tuple[nn.Module, List[float]]:
    """Train a model and return the best model by validation loss."""
    # Simple train/val split for early stopping
    n = len(train_x)
    val_n = max(1, int(0.2 * n))
    indices = list(range(n))
    random.shuffle(indices)
    tr_idx = indices[val_n:]
    val_idx = indices[:val_n]

    tr_x = [train_x[i] for i in tr_idx]
    tr_y = [train_y[i] for i in tr_idx]
    val_x = [train_x[i] for i in val_idx]
    val_y = [train_y[i] for i in val_idx]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    model.train()
    for epoch in range(config.epochs):
        # Training
        indices = list(range(len(tr_x)))
        random.shuffle(indices)
        epoch_loss = 0.0
        batches = 0
        for i in range(0, len(tr_x), config.batch_size):
            batch_idx = indices[i : i + config.batch_size]
            batch_x = torch.stack([tr_x[j] for j in batch_idx]).to(config.device)
            batch_y = torch.tensor([tr_y[j] for j in batch_idx], dtype=torch.long).to(config.device)
            optimizer.zero_grad()
            logits = model(batch_x, config.sample_rate)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.stack(val_x).to(config.device), config.sample_rate)
            val_loss = criterion(val_logits, torch.tensor(val_y, dtype=torch.long).to(config.device)).item()
        model.train()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, []


def evaluate_model(model: nn.Module, x: List[torch.Tensor], y: List[int], config: BaselineConfig) -> Dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.stack(x).to(config.device), config.sample_rate)
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_np = np.array(y)
    return {
        "accuracy": float(accuracy_score(y_np, preds)),
        "f1_macro": float(f1_score(y_np, preds, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def run_cross_validation(config: BaselineConfig) -> Dict:
    print("Loading SpeechCommands dataset...")
    signals, labels, class_names = load_speechcommands(config)
    print(f"Loaded {len(signals)} samples across {len(class_names)} classes: {class_names}")

    X = np.array([i for i in range(len(signals))])
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)

    results = {
        "bifrost_finetuned": {"accuracy": [], "f1_macro": []},
        "bifrost_frozen": {"accuracy": [], "f1_macro": []},
        "stft_baseline": {"accuracy": [], "f1_macro": []},
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{config.n_folds} ===")
        train_signals = [signals[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_signals = [signals[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        # Bifrost fine-tuned
        pipeline = build_bifrost_pipeline(config)
        model = BifrostClassifier(pipeline, config.d_model, config.n_classes, dropout=config.dropout).to(config.device)
        for p in model.pipeline.parameters():
            p.requires_grad = True
        model, _ = train_model(model, train_signals, train_labels, config, "bifrost_finetuned")
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["bifrost_finetuned"][k].append(v)
        print(f"  Bifrost fine-tuned: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

        # Bifrost frozen
        pipeline = build_bifrost_pipeline(config)
        model = BifrostClassifier(pipeline, config.d_model, config.n_classes, dropout=config.dropout).to(config.device)
        for p in model.pipeline.parameters():
            p.requires_grad = False
        model, _ = train_model(model, train_signals, train_labels, config, "bifrost_frozen")
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["bifrost_frozen"][k].append(v)
        print(f"  Bifrost frozen:    acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

        # STFT baseline
        model = STFTBaseline(config.n_fft, config.n_classes, dropout=config.dropout).to(config.device)
        model, _ = train_model(model, train_signals, train_labels, config, "stft_baseline")
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["stft_baseline"][k].append(v)
        print(f"  STFT baseline:     acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

    # Summarize
    summary = {}
    for model_name, metrics in results.items():
        summary[model_name] = {}
        for metric_name, values in metrics.items():
            arr = np.array(values)
            summary[model_name][metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": [float(v) for v in values],
            }

    # Paired t-tests
    from scipy.stats import ttest_rel
    bf_acc = np.array(results["bifrost_finetuned"]["accuracy"])
    stft_acc = np.array(results["stft_baseline"]["accuracy"])
    frozen_acc = np.array(results["bifrost_frozen"]["accuracy"])

    t_bf_stft, p_bf_stft = ttest_rel(bf_acc, stft_acc)
    t_bf_frozen, p_bf_frozen = ttest_rel(bf_acc, frozen_acc)

    summary["paired_t_tests"] = {
        "bifrost_finetuned_vs_stft": {"t": float(t_bf_stft), "p": float(p_bf_stft)},
        "bifrost_finetuned_vs_bifrost_frozen": {"t": float(t_bf_frozen), "p": float(p_bf_frozen)},
    }

    summary["config"] = {
        "n_classes": config.n_classes,
        "samples_per_class": config.samples_per_class,
        "n_folds": config.n_folds,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "d_model": config.d_model,
        "n_fft": config.n_fft,
    }
    summary["class_names"] = class_names

    result_path = RESULTS_DIR / "phase_coherence_baseline_comparison.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {result_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--samples_per_class", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = BaselineConfig(
        n_classes=args.n_classes,
        samples_per_class=args.samples_per_class,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=args.d_model,
        device=args.device,
    )
    run_cross_validation(config)


if __name__ == "__main__":
    main()
