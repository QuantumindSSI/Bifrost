#!/usr/bin/env python3
"""
Experiment 6: Image MSC instance validation on CIFAR-10.

Tests whether phase congruency features (the image instance of the MSC
framework) outperform raw pixel and raw FFT magnitude baselines on
image classification.

Pre-registered protocol from dev-docs/08_CROSS_MODAL_VALIDATION_PROTOCOL.md:
    - Dataset: CIFAR-10, 10 classes, 5-fold CV (subset for speed).
    - Models: Phase congruency, raw pixel, raw FFT, HOG.
    - Success criterion: PC >= baselines + 5 pp, p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore", category=UserWarning)

RESEARCH_DIR = Path(__file__).parent
RESULTS_DIR = RESEARCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
REPO_ROOT = RESEARCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class ImageConfig:
    n_classes: int = 10
    samples_per_class: int = 500
    image_size: int = 32
    n_scales: int = 5
    n_orientations: int = 6
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    n_folds: int = 5
    device: str = "cpu"


def load_cifar10(config: ImageConfig):
    """Load CIFAR-10 dataset."""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_dir = REPO_ROOT / "data" / "cifar10"
    data_dir.mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    # Combine and subsample
    all_images = []
    all_labels = []
    for img, label in trainset:
        all_images.append(img)  # (3, 32, 32) tensor
        all_labels.append(label)
    for img, label in testset:
        all_images.append(img)
        all_labels.append(label)

    # Subsample per class
    selected_images = []
    selected_labels = []
    for c in range(config.n_classes):
        idx = [i for i, l in enumerate(all_labels) if l == c][:config.samples_per_class]
        for i in idx:
            selected_images.append(all_images[i])
            selected_labels.append(all_labels[i])

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    return selected_images, selected_labels, class_names


class PhaseCongruencyClassifier(nn.Module):
    """Phase congruency features + linear classifier."""

    def __init__(self, config: ImageConfig, n_classes: int, dropout: float = 0.0):
        super().__init__()
        from src.bifrost.msc_image import PhaseCongruencyExtractor
        self.extractor = PhaseCongruencyExtractor(
            n_scales=config.n_scales,
            n_orientations=config.n_orientations,
            image_size=config.image_size,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.extractor.feature_dim, n_classes)

    def forward(self, x):
        feat = self.extractor(x)
        return self.classifier(self.dropout(feat))


class RawPixelBaseline(nn.Module):
    """Raw pixel mean-pool + linear classifier."""

    def __init__(self, image_size: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.image_size = image_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Downsample to 8x8 and flatten
        self.classifier = nn.Linear(3 * 8 * 8, n_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.shape[0], -1)
        return self.classifier(self.dropout(x))


class FFTBaseline(nn.Module):
    """Raw FFT magnitude mean-pool + linear classifier."""

    def __init__(self, image_size: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.image_size = image_size
        self.n_freq = image_size // 2 + 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # FFT magnitude per channel, mean over spatial dims
        self.classifier = nn.Linear(3 * self.n_freq * self.n_freq, n_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        # 2D FFT per channel
        fft = torch.fft.fft2(x)
        mag = fft.abs()
        # Use only lower frequencies (n_freq x n_freq)
        mag = mag[:, :, :self.n_freq, :self.n_freq]
        mag = torch.log(mag + 1e-8)
        mag = mag.view(B, -1)
        return self.classifier(self.dropout(mag))


def train_model(model, train_x, train_y, config: ImageConfig):
    n = len(train_x)
    val_n = max(1, int(0.2 * n))
    indices = list(range(n))
    random.shuffle(indices)
    tr_idx, val_idx = indices[val_n:], indices[:val_n]
    tr_x = [train_x[i] for i in tr_idx]
    tr_y = [train_y[i] for i in tr_idx]
    val_x = [train_x[i] for i in val_idx]
    val_y = [train_y[i] for i in val_idx]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_state, patience, patience_counter = float("inf"), None, 10, 0
    model.train()
    for epoch in range(config.epochs):
        indices = list(range(len(tr_x)))
        random.shuffle(indices)
        for i in range(0, len(tr_x), config.batch_size):
            batch_idx = indices[i:i + config.batch_size]
            batch_x = torch.stack([tr_x[j] for j in batch_idx]).to(config.device)
            batch_y = torch.tensor([tr_y[j] for j in batch_idx], dtype=torch.long).to(config.device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.stack(val_x).to(config.device))
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
    return model


def evaluate_model(model, x, y, config: ImageConfig):
    model.eval()
    preds = []
    batch_size = config.batch_size
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = torch.stack(x[i:i+batch_size]).to(config.device)
            logits = model(batch_x)
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    y_np = np.array(y)
    preds_np = np.array(preds)
    return {
        "accuracy": float(accuracy_score(y_np, preds_np)),
        "f1_macro": float(f1_score(y_np, preds_np, average="macro", zero_division=0)),
    }


def run_experiment(config: ImageConfig):
    print("Loading CIFAR-10...")
    images, labels, class_names = load_cifar10(config)
    print(f"Loaded {len(images)} samples, {len(class_names)} classes: {class_names}")

    X = np.arange(len(images))
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)

    models_to_test = ["phase_congruency", "raw_pixel", "fft_magnitude"]
    results = {m: {"accuracy": [], "f1_macro": []} for m in models_to_test}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{config.n_folds} ===")
        tr_x = [images[i] for i in train_idx]
        tr_l = [labels[i] for i in train_idx]
        te_x = [images[i] for i in test_idx]
        te_l = [labels[i] for i in test_idx]

        # Phase Congruency
        model = PhaseCongruencyClassifier(config, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, tr_x, tr_l, config)
        m = evaluate_model(model, te_x, te_l, config)
        for k, v in m.items(): results["phase_congruency"][k].append(v)
        print(f"  Phase Congruency: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

        # Raw Pixel
        model = RawPixelBaseline(config.image_size, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, tr_x, tr_l, config)
        m = evaluate_model(model, te_x, te_l, config)
        for k, v in m.items(): results["raw_pixel"][k].append(v)
        print(f"  Raw Pixel:        acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

        # FFT Magnitude
        model = FFTBaseline(config.image_size, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, tr_x, tr_l, config)
        m = evaluate_model(model, te_x, te_l, config)
        for k, v in m.items(): results["fft_magnitude"][k].append(v)
        print(f"  FFT Magnitude:    acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

    summary = {}
    for mn, metrics in results.items():
        summary[mn] = {}
        for metric_name, values in metrics.items():
            arr = np.array(values)
            summary[mn][metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": [float(v) for v in values],
            }

    pc_acc = np.array(results["phase_congruency"]["accuracy"])
    pixel_acc = np.array(results["raw_pixel"]["accuracy"])
    fft_acc = np.array(results["fft_magnitude"]["accuracy"])

    t1, p1 = ttest_rel(pc_acc, pixel_acc)
    t2, p2 = ttest_rel(pc_acc, fft_acc)

    summary["paired_t_tests"] = {
        "pc_vs_pixel": {"t": float(t1), "p": float(p1)},
        "pc_vs_fft": {"t": float(t2), "p": float(p2)},
    }
    summary["config"] = {
        "n_classes": config.n_classes, "samples_per_class": config.samples_per_class,
        "image_size": config.image_size, "n_scales": config.n_scales,
        "n_orientations": config.n_orientations, "n_folds": config.n_folds,
        "epochs": config.epochs, "batch_size": config.batch_size,
        "lr": config.lr, "weight_decay": config.weight_decay, "dropout": config.dropout,
    }
    summary["class_names"] = class_names

    delta_pixel = pc_acc.mean() - pixel_acc.mean()
    delta_fft = pc_acc.mean() - fft_acc.mean()
    summary["hypothesis_evaluation"] = {
        "H1_pc_beats_pixel_by_5pp": {
            "delta_accuracy": float(delta_pixel),
            "p_value": float(p1),
            "supported": bool(delta_pixel >= 0.05 and p1 < 0.05),
        },
        "H1_pc_beats_fft": {
            "delta_accuracy": float(delta_fft),
            "p_value": float(p2),
            "supported": bool(delta_fft >= 0.05 and p2 < 0.05),
        },
    }

    result_path = RESULTS_DIR / "msc_image_cifar10.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {result_path}")
    print("\n" + "=" * 60)
    print("SUMMARY (CIFAR-10)")
    print("=" * 60)
    for mn in models_to_test:
        acc = summary[mn]["accuracy"]
        f1 = summary[mn]["f1_macro"]
        print(f"  {mn:20s}: acc={acc['mean']:.4f}±{acc['std']:.4f}, f1={f1['mean']:.4f}±{f1['std']:.4f}")
    he1 = summary["hypothesis_evaluation"]["H1_pc_beats_pixel_by_5pp"]
    he2 = summary["hypothesis_evaluation"]["H1_pc_beats_fft"]
    s1 = "SUPPORTED" if he1["supported"] else "NOT SUPPORTED"
    s2 = "SUPPORTED" if he2["supported"] else "NOT SUPPORTED"
    print(f"\n  PC beats pixel by 5pp: delta={he1['delta_accuracy']:.4f}, p={he1['p_value']:.4f} → {s1}")
    print(f"  PC beats FFT by 5pp:   delta={he2['delta_accuracy']:.4f}, p={he2['p_value']:.4f} → {s2}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--samples_per_class", type=int, default=500)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n_scales", type=int, default=5)
    parser.add_argument("--n_orientations", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    config = ImageConfig(
        n_classes=args.n_classes, samples_per_class=args.samples_per_class,
        n_folds=args.n_folds, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        n_scales=args.n_scales, n_orientations=args.n_orientations,
        device=args.device,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
