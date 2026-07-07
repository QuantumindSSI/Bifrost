"""
Shared utilities for Bifrost experiments.

Reduces code duplication across experiment scripts by providing:
- Standardized evaluation (k-fold CV with sklearn)
- Statistical testing (paired t-test, Mann-Whitney U)
- Result formatting and saving
- Deep learning baselines (MLP, simple CNN)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats


def run_kfold_classification(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    classifier: str = "logreg",
    random_state: int = 42,
    **clf_kwargs,
) -> Dict:
    """Run k-fold cross-validation classification.

    Parameters
    ----------
    features : np.ndarray (N, D)
    labels : np.ndarray (N,)
    n_folds : int
    classifier : str
        "logreg" for LogisticRegression, "mlp" for simple MLP
    random_state : int

    Returns
    -------
    dict with mean_accuracy, std_accuracy, accuracies (list), per_fold predictions
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    accuracies = []

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if classifier == "logreg":
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state,
                                     **clf_kwargs)
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
        elif classifier == "mlp":
            clf = MLPClassifier(input_dim=features.shape[1],
                                n_classes=len(np.unique(labels)),
                                **clf_kwargs)
            acc = train_and_eval_mlp(clf, X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        accuracies.append(acc)

    accs = np.array(accuracies)
    return {
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "accuracies": accs.tolist(),
    }


def paired_ttest(
    condition_a: List[float],
    condition_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """Paired t-test between two conditions.

    Returns dict with t_statistic, p_value, delta, significant.
    """
    a = np.array(condition_a)
    b = np.array(condition_b)
    t_stat, p_value = stats.ttest_rel(a, b)
    delta = a.mean() - b.mean()
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta_accuracy": float(delta),
        "significant": bool(p_value < alpha),
    }


def save_results(results: Dict, output_path: str) -> None:
    """Save results to JSON, creating directory if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def print_results_table(
    results: Dict,
    conditions: List[str],
    baseline: str = None,
    title: str = "RESULTS",
) -> None:
    """Print a formatted results table."""
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")

    if baseline:
        baseline_acc = results[baseline]["mean_accuracy"]
        print(f"\n{'Condition':<30} {'Accuracy (mean ± std)':<25} {'Delta':<15}")
    else:
        print(f"\n{'Condition':<30} {'Accuracy (mean ± std)':<25}")
    print("-" * 70)

    for cond in conditions:
        acc = results[cond]["mean_accuracy"]
        std = results[cond]["std_accuracy"]
        if baseline and cond != baseline:
            delta = acc - baseline_acc
            print(f"{cond:<30} {acc:.4f} ± {std:.4f}          {delta:+.4f}")
        else:
            print(f"{cond:<30} {acc:.4f} ± {std:.4f}")


def print_stat_tests(
    stat_tests: Dict,
    title: str = "STATISTICAL TESTS",
) -> None:
    """Print formatted statistical test results."""
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"\n{'Condition':<30} {'Delta':<10} {'t-stat':<10} {'p-value':<10} {'Sig?'}")
    print("-" * 70)

    for cond, test in stat_tests.items():
        p = test["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{cond:<30} {test['delta_accuracy']:+.4f}    "
              f"{test['t_statistic']:+.3f}     {p:.4f}    {sig}")


class MLPClassifier:
    """Simple MLP baseline classifier using PyTorch.

    A competitive nonlinear baseline that can learn complex relationships
    that LogisticRegression cannot.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        lr: float = 1e-3,
        n_epochs: int = 100,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.model = nn.Sequential(*layers).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1)
            acc = (preds == y_t).float().mean().item()
        return acc


def train_and_eval_mlp(
    clf: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train MLP and return test accuracy."""
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


def extract_raw_features(data: torch.Tensor, modality: str) -> np.ndarray:
    """Extract raw flattened features (baseline).

    For audio: flatten waveform
    For images: flatten pixels
    For sensor: flatten channels x time
    """
    return data.reshape(data.shape[0], -1).numpy()


def extract_statistical_features_sensor(signals: torch.Tensor) -> np.ndarray:
    """Extract per-channel statistical features (mean, std, min, max, skew, kurt)."""
    B, C, T = signals.shape
    features = []
    for ch in range(C):
        ch_sig = signals[:, ch, :].numpy()
        features.append(ch_sig.mean(axis=-1))
        features.append(ch_sig.std(axis=-1))
        features.append(ch_sig.min(axis=-1))
        features.append(ch_sig.max(axis=-1))
        # Skewness
        mean = ch_sig.mean(axis=-1, keepdims=True)
        std = ch_sig.std(axis=-1, keepdims=True)
        skew = ((ch_sig - mean) ** 3).mean(axis=-1) / (std.squeeze(-1) ** 3 + 1e-8)
        features.append(skew)
        # Kurtosis
        kurt = ((ch_sig - mean) ** 4).mean(axis=-1) / (std.squeeze(-1) ** 4 + 1e-8)
        features.append(kurt)
    return np.stack(features, axis=-1)


def extract_fft_magnitude_audio(waveforms: torch.Tensor, n_fft: int = 1024) -> np.ndarray:
    """Extract FFT magnitude features from audio (amplitude-only baseline)."""
    B, T = waveforms.shape
    # Use STFT magnitude averaged over time
    stft = torch.stft(waveforms, n_fft=n_fft, hop_length=n_fft // 2,
                      return_complex=True)
    mag = stft.abs()  # (B, n_freq, n_frames)
    # Average over time frames, keep frequency
    features = mag.mean(dim=-1).numpy()  # (B, n_freq)
    return features


def extract_fft_magnitude_image(images: torch.Tensor) -> np.ndarray:
    """Extract 2D FFT magnitude features from images (amplitude-only baseline)."""
    B, C, H, W = images.shape
    if C == 3:
        images_gray = images.mean(dim=1)
    else:
        images_gray = images.squeeze(1)
    fft2d = torch.fft.fft2(images_gray)
    mag = fft2d.abs()
    return mag.reshape(B, -1).numpy()
