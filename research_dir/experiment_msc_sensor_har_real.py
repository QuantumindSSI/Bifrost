"""
Experiment 3B-REAL: Sensor MSC on REAL UCI HAR data

Tests Claim C3: Wavelet coherence captures semantic structure in real sensor data.
Uses REAL UCI HAR dataset (6 activities, 6 IMU channels, 50Hz).

Usage:
    python3 research_dir/experiment_msc_sensor_har_real.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.msc_sensor import WaveletCoherenceExtractor
from research_dir.data_loaders import load_uci_har
from research_dir.experiment_utils import (
    save_results, print_results_table, print_stat_tests,
    extract_statistical_features_sensor, MLPClassifier,
)


def extract_fft_magnitude_sensor(signals: torch.Tensor, n_fft: int = 64) -> np.ndarray:
    """Extract FFT magnitude features from sensor signals (amplitude-only)."""
    B, C, T = signals.shape
    fft = torch.fft.rfft(signals, dim=-1, n=n_fft)
    mag = fft.abs()
    # Mean and std across channels
    mag_mean = mag.mean(dim=1).numpy()
    mag_std = mag.std(dim=1).numpy()
    return np.concatenate([mag_mean, mag_std], axis=-1)


def run_experiment(
    signals: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run sensor MSC experiment on real UCI HAR data."""

    extractor = WaveletCoherenceExtractor(
        n_scales=8, n_channels=signals.shape[1], sample_rate=50.0,
    )

    conditions = ["wavelet_coherence", "fft_magnitude", "statistical", "raw"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(signals, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_sig = signals[train_idx]
        test_sig = signals[test_idx]
        train_labels = labels[train_idx].numpy()
        test_labels = labels[test_idx].numpy()

        for condition in conditions:
            try:
                if condition == "wavelet_coherence":
                    train_feat = extractor(train_sig).numpy()
                    test_feat = extractor(test_sig).numpy()
                elif condition == "fft_magnitude":
                    train_feat = extract_fft_magnitude_sensor(train_sig)
                    test_feat = extract_fft_magnitude_sensor(test_sig)
                elif condition == "statistical":
                    train_feat = extract_statistical_features_sensor(train_sig)
                    test_feat = extract_statistical_features_sensor(test_sig)
                elif condition == "raw":
                    train_feat = train_sig.reshape(train_sig.shape[0], -1).numpy()
                    test_feat = test_sig.reshape(test_sig.shape[0], -1).numpy()
                else:
                    continue

                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
                clf.fit(train_feat, train_labels)
                acc = clf.score(test_feat, test_labels)
                results[condition]["accuracies"].append(acc)
            except Exception as e:
                print(f"    {condition}: ERROR - {e}")
                results[condition]["accuracies"].append(0.0)

    summary = {}
    for cond in conditions:
        accs = np.array(results[cond]["accuracies"])
        summary[cond] = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "accuracies": accs.tolist(),
        }

    wc_accs = np.array(results["wavelet_coherence"]["accuracies"])
    summary["statistical_tests"] = {}
    for cond in conditions:
        if cond == "wavelet_coherence":
            continue
        base_accs = np.array(results[cond]["accuracies"])
        t_stat, p_value = stats.ttest_rel(wc_accs, base_accs)
        delta = wc_accs.mean() - base_accs.mean()
        summary["statistical_tests"][cond] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(delta),
            "significant": bool(p_value < 0.05),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3B-REAL: Sensor MSC on Real UCI HAR")
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp3b_real_ucihar.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 3B-REAL: Sensor MSC on REAL UCI HAR")
    print("Claim C3: Wavelet coherence captures real sensor structure")
    print("=" * 70)

    print(f"\nLoading UCI HAR ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class)...")
    signals, labels, class_names = load_uci_har(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    print(f"Loaded {len(signals)} samples, {len(class_names)} classes")
    print(f"Classes: {class_names}")
    print(f"Signal shape: {signals.shape}")

    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    results = run_experiment(signals, labels, n_folds=args.n_folds)

    print_results_table(results, ["wavelet_coherence", "fft_magnitude",
                                   "statistical", "raw"],
                        baseline=None, title="RESULTS (REAL UCI HAR)")
    print_stat_tests(results["statistical_tests"])

    save_results(results, args.output)

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)
    wc_acc = results["wavelet_coherence"]["mean_accuracy"]
    fft_acc = results["fft_magnitude"]["mean_accuracy"]
    stat_acc = results["statistical"]["mean_accuracy"]
    raw_acc = results["raw"]["mean_accuracy"]
    chance = 1.0 / args.n_classes

    print(f"WaveletCoherence:   {wc_acc:.4f}")
    print(f"FFT magnitude:       {fft_acc:.4f}")
    print(f"Statistical:         {stat_acc:.4f}")
    print(f"Raw:                 {raw_acc:.4f}")
    print(f"Chance:              {chance:.4f}")
    print()

    wc_vs_fft = results["statistical_tests"].get("fft_magnitude", {})
    wc_vs_stat = results["statistical_tests"].get("statistical", {})
    wc_vs_raw = results["statistical_tests"].get("raw", {})

    if wc_acc < chance + 0.05:
        print("NEGATIVE RESULT: WaveletCoherence at chance on real data.")
    elif wc_vs_fft.get("delta_accuracy", 0) < 0:
        print("NEGATIVE RESULT: WaveletCoherence WORSE than FFT magnitude.")
        print(f"  Delta: {wc_vs_fft['delta_accuracy']:+.4f}")
    elif wc_vs_fft.get("significant", False) and wc_vs_fft["delta_accuracy"] > 0.05:
        print("SUPPORTS C3: WaveletCoherence > FFT magnitude on real sensor data.")
        if wc_vs_stat.get("delta_accuracy", 0) > 0:
            print(f"  Also beats statistical features by {wc_vs_stat['delta_accuracy']:+.4f}")
        else:
            print(f"  Does NOT beat statistical features ({wc_vs_stat['delta_accuracy']:+.4f})")
    else:
        print("MIXED RESULT: WaveletCoherence does not clearly beat baselines.")


if __name__ == "__main__":
    main()
