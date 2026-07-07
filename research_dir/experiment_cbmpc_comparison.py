#!/usr/bin/env python3
"""
Pre-registered Experiment 3: Cross-Band Modulation Phase Coherence (CBMPC)
vs. spectral baselines on SpeechCommands.

Protocol (from CBMPC_TECHNIQUE_PROPOSAL.md):
    - Dataset: Google SpeechCommands v0.02, 10 core classes, 200 samples per class.
    - Models:
        1. CBMPC-Bifrost: Bifrost pipeline → CBMPC extraction → linear classifier
        2. CBMPC-STFT: Raw STFT → CBMPC extraction → linear classifier
        3. STFT magnitude baseline: log STFT mag → mean-pool → linear classifier
        4. Mel baseline: 64-bin mel → mean-pool → linear classifier
        5. Bifrost amplitude-only: Bifrost → amp mean+std → linear classifier
    - Evaluation: stratified 5-fold cross-validation.
    - Metrics: macro accuracy and F1, mean ± std across folds.
    - Statistics: paired t-test between CBMPC models and STFT baseline.

Primary hypothesis (H1):
    CBMPC-Bifrost exceeds STFT magnitude baseline by ≥ 5 absolute percentage points
    in mean test accuracy, with p < 0.05 after Bonferroni correction.

Secondary hypothesis (H2):
    CBMPC-STFT exceeds STFT magnitude baseline by ≥ 5 absolute percentage points,
    demonstrating that modulation phase coherence itself carries semantic structure.
"""

from __future__ import annotations

import argparse
import json
import math
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
class ExperimentConfig:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_speechcommands(config: ExperimentConfig) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS

    root = REPO_ROOT / "data" / "speechcommands"
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
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        all_signals.append(waveform.squeeze(0))
        all_labels.append(str(label))

    counts = Counter(all_labels)
    # Prefer the 10 core command classes
    core_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    available_core = [c for c in core_classes if counts.get(c, 0) >= config.samples_per_class]
    if len(available_core) >= config.n_classes:
        selected = sorted(available_core[:config.n_classes])
    else:
        selected = [lab for lab, _ in counts.most_common(config.n_classes)]
        selected = sorted(selected)

    label_to_idx = {lab: i for i, lab in enumerate(selected)}

    signals: List[torch.Tensor] = []
    labels: List[int] = []
    for c in selected:
        c_signals = [s for s, y in zip(all_signals, all_labels) if y == c]
        c_signals = c_signals[:config.samples_per_class]
        signals.extend(c_signals)
        labels.extend([label_to_idx[c]] * len(c_signals))

    return signals, labels, selected


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_bifrost_pipeline(config: ExperimentConfig):
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


class CBMPCBifrostClassifier(nn.Module):
    """Bifrost pipeline → CBMPC extraction on pipeline output → linear classifier."""

    def __init__(self, pipeline, config: ExperimentConfig, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.n_mod_freqs = 7
        self.modulation_freqs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        # Rich mode: n_mels * n_mod_freqs + 2 * n_mod_freqs
        self.feature_dim = config.n_mels * self.n_mod_freqs + 2 * self.n_mod_freqs
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x, *_):
        st, _ = self.pipeline(x, metadata={"sample_rate": float(self.config.sample_rate)})
        amp = st.amplitude
        if amp.dim() == 2:
            emb = amp
        else:
            emb = self._extract_cbmpc_from_spectrogram(amp)
        emb = self.dropout(emb)
        return self.classifier(emb)

    def _extract_cbmpc_from_spectrogram(self, amp: torch.Tensor) -> torch.Tensor:
        """Extract rich CBMPC features from a spectrogram (B, T, F)."""
        B, T, F_dim = amp.shape
        log_mag = torch.log(amp + 1e-8)

        mod_spec = torch.fft.rfft(log_mag, dim=1)
        mod_amp = mod_spec.abs()
        mod_phase = mod_spec.angle()

        frame_rate = T / self.config.duration_seconds
        n_mod_bins = mod_spec.shape[1]
        mod_freqs_all = torch.fft.rfftfreq(T, d=1.0 / frame_rate)

        target_bins = []
        for target_f in self.modulation_freqs:
            if len(mod_freqs_all) == 0:
                target_bins.append(0)
                continue
            bin_idx = torch.argmin(torch.abs(mod_freqs_all - target_f)).item()
            target_bins.append(min(bin_idx, n_mod_bins - 1))

        plv_values = []
        mean_amp_values = []
        per_band_amp_values = []
        for bin_idx in target_bins:
            phases = mod_phase[:, bin_idx, :]
            plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=1)).real
            plv_values.append(plv)
            amp_val = mod_amp[:, bin_idx, :].mean(dim=1)
            mean_amp_values.append(amp_val)
            per_band_amp_values.append(mod_amp[:, bin_idx, :])

        plv_tensor = torch.stack(plv_values, dim=1)
        amp_tensor = torch.stack(mean_amp_values, dim=1)
        per_band = torch.stack(per_band_amp_values, dim=2)
        per_band_flat = per_band.reshape(B, -1)
        return torch.cat([per_band_flat, plv_tensor, amp_tensor], dim=1).float()


class CBMPCSTFTClassifier(nn.Module):
    """Raw STFT → CBMPC extraction → linear classifier."""

    def __init__(self, config: ExperimentConfig, n_classes: int, dropout: float = 0.0):
        super().__init__()
        from src.bifrost.cbmpc import CBMPCExtractor, CBMPCClassifier
        self.model = CBMPCClassifier(
            extractor=CBMPCExtractor(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                modulation_freqs=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                duration_seconds=config.duration_seconds,
                feature_mode="rich",
            ),
            n_classes=n_classes,
            dropout=dropout,
        )

    def forward(self, x, *_):
        return self.model(x)


class STFTMagnitudeBaseline(nn.Module):
    """Log STFT magnitude → mean-pool → linear classifier."""

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
        emb = self.dropout(emb)
        return self.classifier(emb)


class MelBaseline(nn.Module):
    """Mel-spectrogram → mean-pool → linear classifier."""

    def __init__(self, config: ExperimentConfig, n_classes: int, dropout: float = 0.0):
        super().__init__()
        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(config.n_mels, n_classes)

    def forward(self, x, *_):
        mel = self.mel_transform(x)
        mel = torch.log(mel + 1e-8)
        emb = mel.mean(dim=-1)
        emb = self.dropout(emb)
        return self.classifier(emb)


class BifrostAmplitudeClassifier(nn.Module):
    """Bifrost pipeline → amplitude mean+std → linear classifier (original flawed embedding)."""

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
        emb = self.dropout(emb)
        return self.classifier(emb)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_x, train_y, config: ExperimentConfig) -> nn.Module:
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


def evaluate_model(model, x, y, config: ExperimentConfig) -> Dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.stack(x).to(config.device))
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_np = np.array(y)
    return {
        "accuracy": float(accuracy_score(y_np, preds)),
        "f1_macro": float(f1_score(y_np, preds, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> Dict:
    print("Loading SpeechCommands dataset...")
    signals, labels, class_names = load_speechcommands(config)
    print(f"Loaded {len(signals)} samples across {len(class_names)} classes: {class_names}")

    X = np.arange(len(signals))
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)

    models_to_test = ["cbmpc_stft", "cbmpc_bifrost", "stft_baseline", "mel_baseline", "bifrost_amp"]
    results = {m: {"accuracy": [], "f1_macro": []} for m in models_to_test}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{config.n_folds} ===")
        train_signals = [signals[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_signals = [signals[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        # 1. CBMPC-STFT
        model = CBMPCSTFTClassifier(config, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, train_signals, train_labels, config)
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["cbmpc_stft"][k].append(v)
        print(f"  CBMPC-STFT:       acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

        # 2. CBMPC-Bifrost
        pipeline = build_bifrost_pipeline(config)
        model = CBMPCBifrostClassifier(pipeline, config, config.n_classes, dropout=config.dropout).to(config.device)
        # Freeze pipeline, train only extractor + classifier
        # Actually CBMPC extractor has no learned params, so only classifier trains
        model = train_model(model, train_signals, train_labels, config)
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["cbmpc_bifrost"][k].append(v)
        print(f"  CBMPC-Bifrost:    acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

        # 3. STFT magnitude baseline
        model = STFTMagnitudeBaseline(config.n_fft, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, train_signals, train_labels, config)
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["stft_baseline"][k].append(v)
        print(f"  STFT baseline:    acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

        # 4. Mel baseline
        model = MelBaseline(config, config.n_classes, dropout=config.dropout).to(config.device)
        model = train_model(model, train_signals, train_labels, config)
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["mel_baseline"][k].append(v)
        print(f"  Mel baseline:     acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

        # 5. Bifrost amplitude-only (original flawed embedding)
        pipeline = build_bifrost_pipeline(config)
        model = BifrostAmplitudeClassifier(pipeline, config.d_model, config.n_classes, dropout=config.dropout).to(config.device)
        for p in model.pipeline.parameters():
            p.requires_grad = True
        model = train_model(model, train_signals, train_labels, config)
        metrics = evaluate_model(model, test_signals, test_labels, config)
        for k, v in metrics.items():
            results["bifrost_amp"][k].append(v)
        print(f"  Bifrost amp-only: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

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

    # Paired t-tests (primary comparisons)
    cbmpc_stft_acc = np.array(results["cbmpc_stft"]["accuracy"])
    cbmpc_bifrost_acc = np.array(results["cbmpc_bifrost"]["accuracy"])
    stft_acc = np.array(results["stft_baseline"]["accuracy"])
    mel_acc = np.array(results["mel_baseline"]["accuracy"])
    bifrost_amp_acc = np.array(results["bifrost_amp"]["accuracy"])

    tests = {}
    # H1: CBMPC-Bifrost vs STFT baseline
    t1, p1 = ttest_rel(cbmpc_bifrost_acc, stft_acc)
    tests["cbmpc_bifrost_vs_stft"] = {"t": float(t1), "p": float(p1)}
    # H2: CBMPC-STFT vs STFT baseline
    t2, p2 = ttest_rel(cbmpc_stft_acc, stft_acc)
    tests["cbmpc_stft_vs_stft"] = {"t": float(t2), "p": float(p2)}
    # CBMPC-STFT vs Bifrost amp-only
    t3, p3 = ttest_rel(cbmpc_stft_acc, bifrost_amp_acc)
    tests["cbmpc_stft_vs_bifrost_amp"] = {"t": float(t3), "p": float(p3)}
    # CBMPC-Bifrost vs CBMPC-STFT
    t4, p4 = ttest_rel(cbmpc_bifrost_acc, cbmpc_stft_acc)
    tests["cbmpc_bifrost_vs_cbmpc_stft"] = {"t": float(t4), "p": float(p4)}

    summary["paired_t_tests"] = tests
    summary["config"] = {
        "n_classes": config.n_classes,
        "samples_per_class": config.samples_per_class,
        "n_folds": config.n_folds,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
        "n_mels": config.n_mels,
        "d_model": config.d_model,
    }
    summary["class_names"] = class_names
    summary["modulation_freqs"] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    summary["cbmpc_feature_dim"] = 14  # 7 PLV + 7 mean amplitudes

    # Success criteria evaluation
    bonferroni_alpha = 0.05 / 3  # 3 primary comparisons
    h1_success = (cbmpc_bifrost_acc.mean() - stft_acc.mean()) >= 0.05 and p1 < bonferroni_alpha
    h2_success = (cbmpc_stft_acc.mean() - stft_acc.mean()) >= 0.05 and p2 < bonferroni_alpha
    summary["hypothesis_evaluation"] = {
        "H1_cbmpc_bifrost_beats_stft": {
            "delta_accuracy": float(cbmpc_bifrost_acc.mean() - stft_acc.mean()),
            "p_value": float(p1),
            "bonferroni_alpha": float(bonferroni_alpha),
            "supported": bool(h1_success),
        },
        "H2_cbmpc_stft_beats_stft": {
            "delta_accuracy": float(cbmpc_stft_acc.mean() - stft_acc.mean()),
            "p_value": float(p2),
            "bonferroni_alpha": float(bonferroni_alpha),
            "supported": bool(h2_success),
        },
    }

    result_path = RESULTS_DIR / "cbmpc_baseline_comparison.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {result_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name in models_to_test:
        acc = summary[model_name]["accuracy"]
        f1 = summary[model_name]["f1_macro"]
        print(f"  {model_name:20s}: acc={acc['mean']:.4f}±{acc['std']:.4f}, f1={f1['mean']:.4f}±{f1['std']:.4f}")
    print("\nHypothesis evaluation:")
    for h_name, h_eval in summary["hypothesis_evaluation"].items():
        status = "SUPPORTED" if h_eval["supported"] else "NOT SUPPORTED"
        print(f"  {h_name}: delta={h_eval['delta_accuracy']:.4f}, p={h_eval['p_value']:.4f} → {status}")

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

    config = ExperimentConfig(
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
    run_experiment(config)


if __name__ == "__main__":
    main()
