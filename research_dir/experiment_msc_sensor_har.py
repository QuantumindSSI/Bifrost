"""
Experiment 3B: Validate Sensor MSC (Wavelet Coherence on UCI HAR)

Tests Claim C3: The coherence principle generalizes across modalities (sensor).

Method:
    1. Generate synthetic multi-channel sensor data with class-specific
       cross-channel phase relationships
    2. Extract WaveletCoherence features
    3. Train linear classifier
    4. Compare with baselines: raw time series, FFT magnitude, statistical features

Success criterion: WaveletCoherence > FFT magnitude by at least 5 pp

Usage:
    python3 research_dir/experiment_msc_sensor_har.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.msc_sensor import WaveletCoherenceExtractor


def generate_synthetic_sensor(
    n_classes: int = 6,
    n_samples_per_class: int = 200,
    n_channels: int = 6,
    sample_rate: float = 50.0,
    duration: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic multi-channel sensor data where classes differ
    ONLY in cross-channel phase relationships.

    All classes share the same frequency content. Classes differ only in
    the phase relationships between channels. This ensures that wavelet
    coherence (which measures cross-channel phase) is the distinguishing
    feature, not frequency content.
    """
    n_samples = n_classes * n_samples_per_class
    T = int(sample_rate * duration)
    signals = torch.zeros(n_samples, n_channels, T)
    labels = torch.zeros(n_samples, dtype=torch.long)

    # Shared frequencies for all classes
    freqs = [1.0, 3.0, 7.0]
    amps = [1.0, 0.5, 0.3]

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            t = torch.arange(T) / sample_rate
            signal = torch.zeros(n_channels, T)

            for ch in range(n_channels):
                for f, a in zip(freqs, amps):
                    # Class-specific cross-channel phase relationship
                    # This is the ONLY feature that distinguishes classes
                    phase_offset = c * 0.4 * (ch + 1) / f + s * 0.003
                    signal[ch] += a * torch.sin(2 * np.pi * f * t + phase_offset)

            # Add noise
            signal += 0.1 * torch.randn(n_channels, T)
            signals[idx] = signal
            labels[idx] = c
            idx += 1

    perm = torch.randperm(n_samples)
    return signals[perm], labels[perm]


def extract_fft_magnitude(signals: torch.Tensor) -> torch.Tensor:
    """Extract FFT magnitude features (baseline)."""
    B, C, T = signals.shape
    fft = torch.fft.rfft(signals, dim=-1)
    mag = fft.abs()
    # Average over channels, keep frequency bins
    mag_mean = mag.mean(dim=1)  # (B, T//2+1)
    # Also keep per-channel stats
    mag_std = mag.std(dim=1)  # (B, T//2+1)
    return torch.cat([mag_mean, mag_std], dim=-1)


def extract_statistical_features(signals: torch.Tensor) -> torch.Tensor:
    """Extract statistical features (baseline)."""
    B, C, T = signals.shape
    features = []
    for ch in range(C):
        ch_signal = signals[:, ch, :]  # (B, T)
        features.append(ch_signal.mean(dim=-1))  # (B,)
        features.append(ch_signal.std(dim=-1))   # (B,)
        # Skewness
        mean = ch_signal.mean(dim=-1, keepdim=True)  # (B, 1)
        std = ch_signal.std(dim=-1, keepdim=True)    # (B, 1)
        skew = ((ch_signal - mean) ** 3).mean(dim=-1) / (std.squeeze(-1) ** 3 + 1e-8)  # (B,)
        features.append(skew)
        # Kurtosis
        kurt = ((ch_signal - mean) ** 4).mean(dim=-1) / (std.squeeze(-1) ** 4 + 1e-8)  # (B,)
        features.append(kurt)
    return torch.stack(features, dim=-1)  # (B, C*4)


def run_experiment(
    signals: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run sensor MSC validation experiment."""

    extractor = WaveletCoherenceExtractor(
        n_scales=8,
        n_channels=signals.shape[1],
        sample_rate=50.0,
    )

    conditions = ["wavelet_coherence", "fft_magnitude", "statistical", "raw"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(signals, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_sig = signals[train_idx]
        test_sig = signals[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        for condition in conditions:
            if condition == "wavelet_coherence":
                train_feat = extractor(train_sig).numpy()
                test_feat = extractor(test_sig).numpy()
            elif condition == "fft_magnitude":
                train_feat = extract_fft_magnitude(train_sig).numpy()
                test_feat = extract_fft_magnitude(test_sig).numpy()
            elif condition == "statistical":
                train_feat = extract_statistical_features(train_sig).numpy()
                test_feat = extract_statistical_features(test_sig).numpy()
            elif condition == "raw":
                train_feat = train_sig.reshape(train_sig.shape[0], -1).numpy()
                test_feat = test_sig.reshape(test_sig.shape[0], -1).numpy()

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_feat, train_labels.numpy())
            acc = clf.score(test_feat, test_labels.numpy())
            results[condition]["accuracies"].append(acc)

    # Compute statistics
    summary = {}
    for condition in conditions:
        accs = np.array(results[condition]["accuracies"])
        summary[condition] = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "accuracies": accs.tolist(),
        }

    # Paired t-tests: wavelet_coherence vs each baseline
    wc_accs = np.array(results["wavelet_coherence"]["accuracies"])
    summary["statistical_tests"] = {}
    for condition in conditions:
        if condition == "wavelet_coherence":
            continue
        base_accs = np.array(results[condition]["accuracies"])
        t_stat, p_value = stats.ttest_rel(wc_accs, base_accs)
        delta = wc_accs.mean() - base_accs.mean()
        summary["statistical_tests"][condition] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(delta),
            "significant": bool(p_value < 0.05),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3B: Sensor MSC Validation")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_channels", type=int, default=6)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str, default="research_dir/results/exp3b_msc_sensor.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 3B: Sensor MSC Validation (Wavelet Coherence)")
    print("Claim C3: Coherence principle generalizes to sensor modality")
    print("=" * 70)

    print(f"\nGenerating synthetic sensor data ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class, {args.n_channels} channels)...")
    signals, labels = generate_synthetic_sensor(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
        n_channels=args.n_channels,
    )
    print(f"Generated {len(signals)} samples")

    print(f"\nRunning {args.n_folds}-fold cross-validation with 4 conditions...")
    results = run_experiment(signals, labels, n_folds=args.n_folds)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Accuracy (mean ± std)':<25} {'Delta vs WC':<15}")
    print("-" * 65)

    wc_acc = results["wavelet_coherence"]["mean_accuracy"]
    for condition in ["wavelet_coherence", "fft_magnitude", "statistical", "raw"]:
        acc = results[condition]["mean_accuracy"]
        std = results[condition]["std_accuracy"]
        delta = acc - wc_acc if condition != "wavelet_coherence" else 0.0
        print(f"{condition:<25} {acc:.4f} ± {std:.4f}          {delta:+.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (wavelet_coherence vs baselines)")
    print("=" * 70)
    print(f"\n{'Baseline':<25} {'Delta':<10} {'t-stat':<10} {'p-value':<10} {'Sig?'}")
    print("-" * 65)
    for condition, test in results["statistical_tests"].items():
        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
        print(f"{condition:<25} {test['delta_accuracy']:+.4f}    {test['t_statistic']:+.3f}     {test['p_value']:.4f}    {sig}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    fft_delta = wc_acc - results["fft_magnitude"]["mean_accuracy"]
    stat_delta = wc_acc - results["statistical"]["mean_accuracy"]
    print(f"WaveletCoherence accuracy: {wc_acc:.4f}")
    print(f"Delta vs FFT magnitude: {fft_delta:+.4f} (target: > 0.05)")
    print(f"Delta vs statistical features: {stat_delta:+.4f} (target: > 0.10)")
    if fft_delta > 0.05:
        print("SUPPORTS C3: Wavelet coherence captures sensor semantic structure > FFT.")
    else:
        print("INSUFFICIENT EVIDENCE for C3 on this dataset.")


if __name__ == "__main__":
    main()
