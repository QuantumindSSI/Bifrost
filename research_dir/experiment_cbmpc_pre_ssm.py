#!/usr/bin/env python3
"""
Experiment 4: CBMPC pre-SSM integration test.

Tests whether integrating CBMPC as a pre-SSM feature extraction layer
(combined with the SSM embedding) beats CBMPC-only and SSM-only.

Models:
    1. CBMPC-only (no SSM) — the validated baseline (0.41 on SpeechCommands)
    2. SSM-only (no CBMPC) — the original Bifrost amplitude embedding
    3. CBMPC + SSM combined — parallel CBMPC and SSM, concatenated

Success criterion: CBMPC + SSM >= CBMPC-only + 3 pp, p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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
class Config:
    n_classes: int = 10
    samples_per_class: int = 200
    duration_seconds: float = 1.0
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_bands: int = 8
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    n_folds: int = 5
    device: str = "cpu"


def load_speechcommands(config: Config):
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
    root = REPO_ROOT / "data" / "speechcommands"
    root.mkdir(parents=True, exist_ok=True)
    dataset = SPEECHCOMMANDS(root=str(root), download=True, subset="training")
    all_signals, all_labels = [], []
    for waveform, sr, label, *_ in dataset:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != config.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, config.sample_rate)(waveform)
        target_len = int(config.duration_seconds * config.sample_rate)
        if waveform.shape[-1] > target_len:
            waveform = waveform[..., :target_len]
        else:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        all_signals.append(waveform.squeeze(0))
        all_labels.append(str(label))
    counts = Counter(all_labels)
    core = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    available = [c for c in core if counts.get(c, 0) >= config.samples_per_class]
    selected = sorted(available[:config.n_classes]) if len(available) >= config.n_classes else sorted([l for l, _ in counts.most_common(config.n_classes)])
    label_to_idx = {lab: i for i, lab in enumerate(selected)}
    signals, labels = [], []
    for c in selected:
        c_sigs = [s for s, y in zip(all_signals, all_labels) if y == c][:config.samples_per_class]
        signals.extend(c_sigs)
        labels.extend([label_to_idx[c]] * len(c_sigs))
    return signals, labels, selected


def build_bifrost_pipeline(config: Config, use_cbmpc: bool = False):
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
        sample_rate=float(config.sample_rate),
        use_cbmpc=use_cbmpc,
        cbmpc_n_mels=config.n_mels,
        cbmpc_modulation_freqs=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        cbmpc_duration_seconds=config.duration_seconds,
    ).to(config.device)


class CBMPCOnlyClassifier(nn.Module):
    """CBMPC features only (no SSM)."""

    def __init__(self, config: Config, n_classes: int, dropout: float = 0.0):
        super().__init__()
        from src.bifrost.cbmpc import CBMPCExtractor
        self.extractor = CBMPCExtractor(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            modulation_freqs=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            duration_seconds=config.duration_seconds,
            feature_mode="rich",
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.extractor.feature_dim, n_classes)

    def forward(self, x, *_):
        feat = self.extractor(x)
        return self.classifier(self.dropout(feat))


class SSMOnlyClassifier(nn.Module):
    """SSM embedding only (no CBMPC)."""

    def __init__(self, pipeline, d_model: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pipeline = pipeline
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model * 2, n_classes)

    def forward(self, x, *_):
        st, _ = self.pipeline(x, metadata={"sample_rate": 16000.0})
        amp = st.amplitude
        if amp.dim() == 2:
            emb = amp
        else:
            emb = torch.cat([amp.mean(dim=1), amp.std(dim=1)], dim=-1)
        return self.classifier(self.dropout(emb))


class CBMPCSSMCombinedClassifier(nn.Module):
    """CBMPC + SSM combined (parallel, concatenated)."""

    def __init__(self, pipeline, cbmpc_feature_dim: int, d_model: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pipeline = pipeline
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(d_model * 2 + cbmpc_feature_dim, n_classes)

    def forward(self, x, *_):
        st, _ = self.pipeline(x, metadata={"sample_rate": 16000.0})
        amp = st.amplitude
        if amp.dim() == 2:
            ssm_emb = amp
        else:
            ssm_emb = torch.cat([amp.mean(dim=1), amp.std(dim=1)], dim=-1)
        cbmpc_emb = st.metadata.get('cbmpc_features')
        if cbmpc_emb is None:
            cbmpc_emb = torch.zeros(x.shape[0], 0, device=x.device)
        emb = torch.cat([ssm_emb, cbmpc_emb], dim=-1)
        return self.classifier(self.dropout(emb))


def train_model(model, train_x, train_y, config: Config):
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


def evaluate_model(model, x, y, config: Config):
    model.eval()
    with torch.no_grad():
        logits = model(torch.stack(x).to(config.device))
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_np = np.array(y)
    return {
        "accuracy": float(accuracy_score(y_np, preds)),
        "f1_macro": float(f1_score(y_np, preds, average="macro", zero_division=0)),
    }


def run_experiment(config: Config):
    print("Loading SpeechCommands...")
    signals, labels, class_names = load_speechcommands(config)
    print(f"Loaded {len(signals)} samples, {len(class_names)} classes: {class_names}")
    X = np.arange(len(signals))
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    models_to_test = ["cbmpc_only", "ssm_only", "cbmpc_ssm_combined"]
    results = {m: {"accuracy": [], "f1_macro": []} for m in models_to_test}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{config.n_folds} ===")
        tr_s = [signals[i] for i in train_idx]
        tr_l = [labels[i] for i in train_idx]
        te_s = [signals[i] for i in test_idx]
        te_l = [labels[i] for i in test_idx]

        # 1. CBMPC-only
        model = CBMPCOnlyClassifier(config, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, tr_s, tr_l, config)
        m = evaluate_model(model, te_s, te_l, config)
        for k, v in m.items(): results["cbmpc_only"][k].append(v)
        print(f"  CBMPC-only:       acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

        # 2. SSM-only
        pipeline = build_bifrost_pipeline(config, use_cbmpc=False)
        model = SSMOnlyClassifier(pipeline, config.d_model, config.n_classes, dropout=config.dropout).to(config.device)
        for p in model.pipeline.parameters(): p.requires_grad = True
        model = train_model(model, tr_s, tr_l, config)
        m = evaluate_model(model, te_s, te_l, config)
        for k, v in m.items(): results["ssm_only"][k].append(v)
        print(f"  SSM-only:         acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

        # 3. CBMPC + SSM combined
        pipeline = build_bifrost_pipeline(config, use_cbmpc=True)
        cbmpc_dim = pipeline.cbmpc_extractor.feature_dim
        model = CBMPCSSMCombinedClassifier(pipeline, cbmpc_dim, config.d_model, config.n_classes, dropout=config.dropout).to(config.device)
        for p in model.pipeline.parameters(): p.requires_grad = True
        model = train_model(model, tr_s, tr_l, config)
        m = evaluate_model(model, te_s, te_l, config)
        for k, v in m.items(): results["cbmpc_ssm_combined"][k].append(v)
        print(f"  CBMPC+SSM:        acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")

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

    cbmpc_acc = np.array(results["cbmpc_only"]["accuracy"])
    ssm_acc = np.array(results["ssm_only"]["accuracy"])
    combined_acc = np.array(results["cbmpc_ssm_combined"]["accuracy"])

    t_combined_vs_cbmpc, p_combined_vs_cbmpc = ttest_rel(combined_acc, cbmpc_acc)
    t_combined_vs_ssm, p_combined_vs_ssm = ttest_rel(combined_acc, ssm_acc)

    summary["paired_t_tests"] = {
        "combined_vs_cbmpc_only": {"t": float(t_combined_vs_cbmpc), "p": float(p_combined_vs_cbmpc)},
        "combined_vs_ssm_only": {"t": float(t_combined_vs_ssm), "p": float(p_combined_vs_ssm)},
    }
    summary["config"] = {
        "n_classes": config.n_classes, "samples_per_class": config.samples_per_class,
        "n_folds": config.n_folds, "epochs": config.epochs, "batch_size": config.batch_size,
        "lr": config.lr, "weight_decay": config.weight_decay, "dropout": config.dropout,
        "d_model": config.d_model, "n_fft": config.n_fft, "n_mels": config.n_mels,
    }
    summary["class_names"] = class_names

    delta = combined_acc.mean() - cbmpc_acc.mean()
    summary["hypothesis_evaluation"] = {
        "combined_beats_cbmpc_only_by_3pp": {
            "delta_accuracy": float(delta),
            "p_value": float(p_combined_vs_cbmpc),
            "supported": bool(delta >= 0.03 and p_combined_vs_cbmpc < 0.05),
        },
    }

    result_path = RESULTS_DIR / "cbmpc_pre_ssm_integration.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {result_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mn in models_to_test:
        acc = summary[mn]["accuracy"]
        f1 = summary[mn]["f1_macro"]
        print(f"  {mn:25s}: acc={acc['mean']:.4f}±{acc['std']:.4f}, f1={f1['mean']:.4f}±{f1['std']:.4f}")
    he = summary["hypothesis_evaluation"]["combined_beats_cbmpc_only_by_3pp"]
    status = "SUPPORTED" if he["supported"] else "NOT SUPPORTED"
    print(f"\n  Combined beats CBMPC-only by 3pp: delta={he['delta_accuracy']:.4f}, p={he['p_value']:.4f} → {status}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    config = Config(
        n_classes=args.n_classes, samples_per_class=args.samples_per_class,
        n_folds=args.n_folds, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        d_model=args.d_model, device=args.device,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
