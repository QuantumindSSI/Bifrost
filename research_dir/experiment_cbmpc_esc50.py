#!/usr/bin/env python3
"""
Experiment 5: CBMPC generalization test on ESC-50.

Tests whether CBMPC generalizes beyond speech commands to environmental
sound classification (50 classes, 5-second clips).

Pre-registered protocol from dev-docs/05_ESC50_GENERALIZATION_TEST_PLAN.md:
    - Dataset: ESC-50, 50 classes, 40 clips per class, pre-defined 5-fold CV.
    - Models: CBMPC-STFT, STFT baseline, Mel baseline.
    - Success criterion: CBMPC-STFT >= STFT baseline + 5 pp, p < 0.025.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.request
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

ESC50_URL = "https://github.com/karolpiczak/ESC-50/raw/master/audio/00-dogs.zip"
ESC50_META_URL = "https://github.com/karolpiczak/ESC-50/raw/master/meta/esc50.csv"
ESC50_FULL_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


@dataclass
class ESCConfig:
    n_classes: int = 50
    samples_per_class: int = 40
    duration_seconds: float = 5.0
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 1024
    n_mels: int = 64
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    device: str = "cpu"


def download_esc50(data_dir: Path) -> Path:
    """Download ESC-50 dataset."""
    esc_dir = data_dir / "esc50"
    esc_dir.mkdir(parents=True, exist_ok=True)
    meta_path = esc_dir / "meta" / "esc50.csv"
    audio_dir = esc_dir / "audio"

    if meta_path.exists() and audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 0:
        print(f"ESC-50 already downloaded at {esc_dir}")
        return esc_dir

    print("Downloading ESC-50...")
    zip_path = esc_dir / "esc50_master.zip"
    urllib.request.urlretrieve(ESC50_FULL_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(esc_dir)

    # Move contents from ESC-50-master/ to esc_dir
    master_dir = esc_dir / "ESC-50-master"
    if master_dir.exists():
        for item in master_dir.iterdir():
            target = esc_dir / item.name
            if target.exists():
                if target.is_dir():
                    for sub in item.iterdir():
                        sub_target = target / sub.name
                        if not sub_target.exists():
                            sub.rename(sub_target)
                    item.rmdir()
                else:
                    item.rename(target)
            else:
                item.rename(target)
        master_dir.rmdir()

    zip_path.unlink()
    print(f"ESC-50 downloaded to {esc_dir}")
    return esc_dir


def load_esc50(config: ESCConfig) -> Tuple[List[torch.Tensor], List[int], List[str], List[int]]:
    """Load ESC-50 dataset with pre-defined folds."""
    import torchaudio
    data_dir = REPO_ROOT / "data"
    esc_dir = download_esc50(data_dir)
    meta_path = esc_dir / "meta" / "esc50.csv"
    audio_dir = esc_dir / "audio"

    signals: List[torch.Tensor] = []
    labels: List[int] = []
    folds: List[int] = []
    class_names: List[str] = []

    with open(meta_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Get unique class names sorted by target
    class_map = {}
    for row in rows:
        target = int(row['target'])
        if target not in class_map:
            class_map[target] = row['category']
    class_names = [class_map[i] for i in sorted(class_map.keys())]

    for row in rows:
        filename = row['filename']
        target = int(row['target'])
        fold = int(row['fold'])

        filepath = audio_dir / filename
        if not filepath.exists():
            continue

        waveform, sr = torchaudio.load(str(filepath))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != config.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, config.sample_rate)(waveform)
        target_len = int(config.duration_seconds * config.sample_rate)
        if waveform.shape[-1] > target_len:
            waveform = waveform[..., :target_len]
        else:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))

        signals.append(waveform.squeeze(0))
        labels.append(target)
        folds.append(fold)

    return signals, labels, class_names, folds


class CBMPCSTFTClassifier(nn.Module):
    def __init__(self, config: ESCConfig, n_classes: int, dropout: float = 0.0):
        super().__init__()
        from src.bifrost.cbmpc import CBMPCExtractor
        self.extractor = CBMPCExtractor(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            modulation_freqs=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
            duration_seconds=config.duration_seconds,
            feature_mode="rich",
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.extractor.feature_dim, n_classes)

    def forward(self, x, *_):
        feat = self.extractor(x)
        return self.classifier(self.dropout(feat))


class STFTBaseline(nn.Module):
    def __init__(self, n_fft: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.n_freq, n_classes)

    def forward(self, x, *_):
        stft = torch.stft(x, n_fft=self.n_fft, return_complex=True)
        mag = stft.abs()
        mag = torch.log(mag + 1e-8)
        emb = mag.mean(dim=-1)
        return self.classifier(self.dropout(emb))


class MelBaseline(nn.Module):
    def __init__(self, config: ESCConfig, n_classes: int, dropout: float = 0.0):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(config.n_mels, n_classes)

    def forward(self, x, *_):
        mel = self.mel(x)
        mel = torch.log(mel + 1e-8)
        emb = mel.mean(dim=-1)
        return self.classifier(self.dropout(emb))


def train_model(model, train_x, train_y, config: ESCConfig):
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


def evaluate_model(model, x, y, config: ESCConfig):
    model.eval()
    with torch.no_grad():
        logits = model(torch.stack(x).to(config.device))
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_np = np.array(y)
    return {
        "accuracy": float(accuracy_score(y_np, preds)),
        "f1_macro": float(f1_score(y_np, preds, average="macro", zero_division=0)),
    }


def run_experiment(config: ESCConfig):
    print("Loading ESC-50 dataset...")
    signals, labels, class_names, folds = load_esc50(config)
    print(f"Loaded {len(signals)} samples across {len(class_names)} classes")

    # Use pre-defined folds
    unique_folds = sorted(set(folds))
    models_to_test = ["cbmpc_stft", "stft_baseline", "mel_baseline"]
    results = {m: {"accuracy": [], "f1_macro": []} for m in models_to_test}

    for fold in unique_folds:
        print(f"\n=== Fold {fold} (ESC-50 pre-defined) ===")
        train_idx = [i for i, f in enumerate(folds) if f != fold]
        test_idx = [i for i, f in enumerate(folds) if f == fold]
        tr_s = [signals[i] for i in train_idx]
        tr_l = [labels[i] for i in train_idx]
        te_s = [signals[i] for i in test_idx]
        te_l = [labels[i] for i in test_idx]

        # CBMPC-STFT
        model = CBMPCSTFTClassifier(config, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, tr_s, tr_l, config)
        m = evaluate_model(model, te_s, te_l, config)
        for k, v in m.items(): results["cbmpc_stft"][k].append(v)
        print(f"  CBMPC-STFT:    acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

        # STFT baseline
        model = STFTBaseline(config.n_fft, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, tr_s, tr_l, config)
        m = evaluate_model(model, te_s, te_l, config)
        for k, v in m.items(): results["stft_baseline"][k].append(v)
        print(f"  STFT baseline: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

        # Mel baseline
        model = MelBaseline(config, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, tr_s, tr_l, config)
        m = evaluate_model(model, te_s, te_l, config)
        for k, v in m.items(): results["mel_baseline"][k].append(v)
        print(f"  Mel baseline:  acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

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

    cbmpc_acc = np.array(results["cbmpc_stft"]["accuracy"])
    stft_acc = np.array(results["stft_baseline"]["accuracy"])
    mel_acc = np.array(results["mel_baseline"]["accuracy"])

    t1, p1 = ttest_rel(cbmpc_acc, stft_acc)
    t2, p2 = ttest_rel(cbmpc_acc, mel_acc)

    summary["paired_t_tests"] = {
        "cbmpc_vs_stft": {"t": float(t1), "p": float(p1)},
        "cbmpc_vs_mel": {"t": float(t2), "p": float(p2)},
    }
    summary["config"] = {
        "n_classes": config.n_classes, "samples_per_class": config.samples_per_class,
        "duration_seconds": config.duration_seconds, "sample_rate": config.sample_rate,
        "n_fft": config.n_fft, "hop_length": config.hop_length, "n_mels": config.n_mels,
        "epochs": config.epochs, "batch_size": config.batch_size,
        "lr": config.lr, "weight_decay": config.weight_decay, "dropout": config.dropout,
    }
    summary["class_names"] = class_names
    summary["modulation_freqs"] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

    bonferroni_alpha = 0.025
    delta_stft = cbmpc_acc.mean() - stft_acc.mean()
    summary["hypothesis_evaluation"] = {
        "H1_cbmpc_beats_stft": {
            "delta_accuracy": float(delta_stft),
            "p_value": float(p1),
            "bonferroni_alpha": bonferroni_alpha,
            "supported": bool(delta_stft >= 0.05 and p1 < bonferroni_alpha),
        },
    }

    result_path = RESULTS_DIR / "cbmpc_esc50_comparison.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {result_path}")
    print("\n" + "=" * 60)
    print("SUMMARY (ESC-50)")
    print("=" * 60)
    for mn in models_to_test:
        acc = summary[mn]["accuracy"]
        f1 = summary[mn]["f1_macro"]
        print(f"  {mn:20s}: acc={acc['mean']:.4f}±{acc['std']:.4f}, f1={f1['mean']:.4f}±{f1['std']:.4f}")
    he = summary["hypothesis_evaluation"]["H1_cbmpc_beats_stft"]
    status = "SUPPORTED" if he["supported"] else "NOT SUPPORTED"
    print(f"\n  H1 (CBMPC beats STFT by 5pp): delta={he['delta_accuracy']:.4f}, p={he['p_value']:.4f} → {status}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=50)
    parser.add_argument("--samples_per_class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    config = ESCConfig(
        n_classes=args.n_classes, samples_per_class=args.samples_per_class,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        device=args.device,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
