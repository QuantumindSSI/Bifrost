"""
Experiment 1A-ESC50: CBMPC on REAL ESC-50 environmental audio

Tests Claim C1 on real environmental sounds (50 classes).
ESC-50 is more challenging than SpeechCommands — longer audio, more classes.

Usage:
    python3 research_dir/experiment_cbmpc_esc50_real.py
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

from bifrost.cbmpc import CBMPCExtractor
from research_dir.data_loaders import load_esc50
from research_dir.experiment_utils import (
    save_results, print_results_table, print_stat_tests,
    extract_fft_magnitude_audio,
)
from research_dir.experiment_phase_ablation_audio_real import (
    extract_cbmpc_with_phase_ablation,
)


def run_experiment(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run CBMPC vs baselines on ESC-50."""

    extractor = CBMPCExtractor(
        sample_rate=16000, n_fft=1024, hop_length=512,
        n_mels=64, feature_mode="compact",
    )

    conditions = ["cbmpc_baseline", "cbmpc_phase_randomize",
                  "fft_magnitude", "raw_waveform"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(waveforms, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_wav = waveforms[train_idx]
        test_wav = waveforms[test_idx]
        train_labels = labels[train_idx].numpy()
        test_labels = labels[test_idx].numpy()

        for condition in conditions:
            try:
                if condition == "cbmpc_baseline":
                    train_feat = extract_cbmpc_with_phase_ablation(
                        train_wav, extractor, "baseline")
                    test_feat = extract_cbmpc_with_phase_ablation(
                        test_wav, extractor, "baseline")
                elif condition == "cbmpc_phase_randomize":
                    train_feat = extract_cbmpc_with_phase_ablation(
                        train_wav, extractor, "phase_randomize")
                    test_feat = extract_cbmpc_with_phase_ablation(
                        test_wav, extractor, "phase_randomize")
                elif condition == "fft_magnitude":
                    train_feat = extract_fft_magnitude_audio(train_wav, n_fft=1024)
                    test_feat = extract_fft_magnitude_audio(test_wav, n_fft=1024)
                elif condition == "raw_waveform":
                    # ESC-50 is 5s at 16kHz = 80000 samples. Downsample heavily.
                    train_feat = train_wav[:, ::40].numpy()  # 2000 features
                    test_feat = test_wav[:, ::40].numpy()
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

    # Statistical tests
    baseline_accs = np.array(results["cbmpc_baseline"]["accuracies"])
    summary["statistical_tests"] = {}

    # Phase ablation test
    phase_accs = np.array(results["cbmpc_phase_randomize"]["accuracies"])
    t_stat, p_value = stats.ttest_rel(baseline_accs, phase_accs)
    summary["statistical_tests"]["phase_randomize"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta_accuracy": float(baseline_accs.mean() - phase_accs.mean()),
        "significant": bool(p_value < 0.05),
    }

    # CBMPC vs baselines
    for baseline in ["fft_magnitude", "raw_waveform"]:
        base_accs = np.array(results[baseline]["accuracies"])
        t_stat, p_value = stats.ttest_rel(baseline_accs, base_accs)
        summary["statistical_tests"][f"cbmpc_vs_{baseline}"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(baseline_accs.mean() - base_accs.mean()),
            "significant": bool(p_value < 0.05),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Experiment: CBMPC on Real ESC-50")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=40)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp1a_real_esc50.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment: CBMPC on REAL ESC-50")
    print("Claim C1: Phase coherence captures structure (real environmental audio)")
    print("=" * 70)

    print(f"\nLoading ESC-50 ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class)...")
    waveforms, labels, class_names = load_esc50(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    print(f"Loaded {len(waveforms)} samples, {len(class_names)} classes")
    print(f"Classes: {class_names}")
    print(f"Waveform shape: {waveforms.shape}")

    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    results = run_experiment(waveforms, labels, n_folds=args.n_folds)

    print_results_table(results, ["cbmpc_baseline", "cbmpc_phase_randomize",
                                   "fft_magnitude", "raw_waveform"],
                        baseline=None, title="RESULTS (REAL ESC-50)")
    print_stat_tests(results["statistical_tests"])

    save_results(results, args.output)

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)
    cbmpc_acc = results["cbmpc_baseline"]["mean_accuracy"]
    fft_acc = results["fft_magnitude"]["mean_accuracy"]
    raw_acc = results["raw_waveform"]["mean_accuracy"]
    chance = 1.0 / args.n_classes

    print(f"CBMPC baseline:     {cbmpc_acc:.4f}")
    print(f"FFT magnitude:      {fft_acc:.4f}")
    print(f"Raw waveform:       {raw_acc:.4f}")
    print(f"Chance:             {chance:.4f}")
    print()

    cbmpc_vs_fft = results["statistical_tests"].get("cbmpc_vs_fft_magnitude", {})
    phase_test = results["statistical_tests"].get("phase_randomize", {})

    if cbmpc_acc < chance + 0.05:
        print("NEGATIVE RESULT: CBMPC at chance on real ESC-50.")
    elif cbmpc_vs_fft.get("delta_accuracy", 0) < 0:
        print(f"NEGATIVE RESULT: CBMPC WORSE than FFT magnitude ({cbmpc_vs_fft['delta_accuracy']:+.4f}).")
    elif phase_test.get("significant", False):
        print("SUPPORTS C1: Phase ablation degrades ESC-50 classification.")
        if cbmpc_vs_fft.get("significant", False) and cbmpc_vs_fft["delta_accuracy"] > 0:
            print("  CBMPC also beats FFT magnitude.")
        else:
            print(f"  But CBMPC doesn't beat FFT magnitude ({cbmpc_vs_fft.get('delta_accuracy', 0):+.4f})")
    else:
        print("MIXED/NEGATIVE: Phase ablation not significant on ESC-50.")


if __name__ == "__main__":
    main()
