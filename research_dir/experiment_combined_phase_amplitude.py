"""
Experiment COMBINED: Phase + Amplitude vs Amplitude Alone

The real-data experiments showed amplitude (FFT magnitude) beats phase (CBMPC).
But the thesis doesn't claim phase REPLACES amplitude — it claims phase
CAPTURES STRUCTURE that amplitude doesn't.

The right test is: does adding phase features to amplitude features
improve classification? If phase captures unique information, the
combination should beat amplitude alone.

Usage:
    python3 research_dir/experiment_combined_phase_amplitude.py
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
from research_dir.data_loaders import load_speechcommands, load_esc50, load_uci_har
from research_dir.experiment_utils import (
    save_results, print_results_table, print_stat_tests,
    extract_fft_magnitude_audio, extract_statistical_features_sensor,
    extract_fft_magnitude_image,
)
from research_dir.experiment_phase_ablation_audio_real import (
    extract_cbmpc_with_phase_ablation,
)


def run_combined_experiment_audio(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Test whether CBMPC phase features add value to FFT amplitude features."""

    extractor = CBMPCExtractor(
        sample_rate=16000, n_fft=512, hop_length=256,
        n_mels=32, feature_mode="compact",
    )

    conditions = ["fft_only", "cbmpc_only", "fft_plus_cbmpc", "fft_plus_phase_randomized"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(waveforms, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_wav = waveforms[train_idx]
        test_wav = waveforms[test_idx]
        train_labels = labels[train_idx].numpy()
        test_labels = labels[test_idx].numpy()

        # Extract features
        fft_train = extract_fft_magnitude_audio(train_wav, n_fft=512)
        fft_test = extract_fft_magnitude_audio(test_wav, n_fft=512)

        cbmpc_train = extract_cbmpc_with_phase_ablation(train_wav, extractor, "baseline")
        cbmpc_test = extract_cbmpc_with_phase_ablation(test_wav, extractor, "baseline")

        cbmpc_pr_train = extract_cbmpc_with_phase_ablation(train_wav, extractor, "phase_randomize")
        cbmpc_pr_test = extract_cbmpc_with_phase_ablation(test_wav, extractor, "phase_randomize")

        feature_sets = {
            "fft_only": (fft_train, fft_test),
            "cbmpc_only": (cbmpc_train, cbmpc_test),
            "fft_plus_cbmpc": (np.concatenate([fft_train, cbmpc_train], axis=-1),
                               np.concatenate([fft_test, cbmpc_test], axis=-1)),
            "fft_plus_phase_randomized": (np.concatenate([fft_train, cbmpc_pr_train], axis=-1),
                                          np.concatenate([fft_test, cbmpc_pr_test], axis=-1)),
        }

        for condition in conditions:
            try:
                train_feat, test_feat = feature_sets[condition]
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

    # Key test: does FFT+CBMPC beat FFT alone?
    fft_accs = np.array(results["fft_only"]["accuracies"])
    combined_accs = np.array(results["fft_plus_cbmpc"]["accuracies"])
    combined_pr_accs = np.array(results["fft_plus_phase_randomized"]["accuracies"])

    summary["statistical_tests"] = {}

    t_stat, p_value = stats.ttest_rel(combined_accs, fft_accs)
    summary["statistical_tests"]["fft_plus_cbmpc_vs_fft_only"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta_accuracy": float(combined_accs.mean() - fft_accs.mean()),
        "significant": bool(p_value < 0.05),
        "interpretation": "Does adding phase features improve amplitude-only?",
    }

    # Does FFT+randomized phase beat FFT alone? (Should NOT if phase matters)
    t_stat, p_value = stats.ttest_rel(combined_pr_accs, fft_accs)
    summary["statistical_tests"]["fft_plus_randomized_vs_fft_only"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta_accuracy": float(combined_pr_accs.mean() - fft_accs.mean()),
        "significant": bool(p_value < 0.05),
        "interpretation": "Does adding RANDOMIZED phase features improve amplitude-only?",
    }

    # Does real phase beat randomized phase when combined with FFT?
    t_stat, p_value = stats.ttest_rel(combined_accs, combined_pr_accs)
    summary["statistical_tests"]["real_vs_randomized_phase_combined"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta_accuracy": float(combined_accs.mean() - combined_pr_accs.mean()),
        "significant": bool(p_value < 0.05),
        "interpretation": "Does REAL phase add more than RANDOMIZED phase?",
    }

    return summary


def run_combined_experiment_sensor(
    signals: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Test whether WaveletCoherence adds value to FFT/statistical on sensor data."""

    from bifrost.msc_sensor import WaveletCoherenceExtractor

    extractor = WaveletCoherenceExtractor(
        n_scales=8, n_channels=signals.shape[1], sample_rate=50.0,
    )

    conditions = ["fft_only", "wc_only", "fft_plus_wc", "stat_plus_wc"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(signals, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_sig = signals[train_idx]
        test_sig = signals[test_idx]
        train_labels = labels[train_idx].numpy()
        test_labels = labels[test_idx].numpy()

        # Extract features
        from research_dir.experiment_msc_sensor_har_real import extract_fft_magnitude_sensor
        fft_train = extract_fft_magnitude_sensor(train_sig)
        fft_test = extract_fft_magnitude_sensor(test_sig)

        stat_train = extract_statistical_features_sensor(train_sig)
        stat_test = extract_statistical_features_sensor(test_sig)

        wc_train = extractor(train_sig).numpy()
        wc_test = extractor(test_sig).numpy()

        feature_sets = {
            "fft_only": (fft_train, fft_test),
            "wc_only": (wc_train, wc_test),
            "fft_plus_wc": (np.concatenate([fft_train, wc_train], axis=-1),
                            np.concatenate([fft_test, wc_test], axis=-1)),
            "stat_plus_wc": (np.concatenate([stat_train, wc_train], axis=-1),
                             np.concatenate([stat_test, wc_test], axis=-1)),
        }

        for condition in conditions:
            try:
                train_feat, test_feat = feature_sets[condition]
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

    fft_accs = np.array(results["fft_only"]["accuracies"])
    combined_accs = np.array(results["fft_plus_wc"]["accuracies"])
    stat_combined_accs = np.array(results["stat_plus_wc"]["accuracies"])

    summary["statistical_tests"] = {}

    t_stat, p_value = stats.ttest_rel(combined_accs, fft_accs)
    summary["statistical_tests"]["fft_plus_wc_vs_fft_only"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta_accuracy": float(combined_accs.mean() - fft_accs.mean()),
        "significant": bool(p_value < 0.05),
        "interpretation": "Does adding wavelet coherence improve FFT?",
    }

    t_stat, p_value = stats.ttest_rel(stat_combined_accs, fft_accs)
    summary["statistical_tests"]["stat_plus_wc_vs_fft_only"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta_accuracy": float(stat_combined_accs.mean() - fft_accs.mean()),
        "significant": bool(p_value < 0.05),
        "interpretation": "Does stat+wavelet coherence beat FFT?",
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Combined Phase+Amplitude vs Amplitude Alone")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp_combined_phase_amplitude.json")
    args = parser.parse_args()

    print("=" * 70)
    print("COMBINED: Phase + Amplitude vs Amplitude Alone")
    print("Does phase add value ON TOP of amplitude?")
    print("=" * 70)

    all_results = {}

    # Test 1: SpeechCommands
    print("\n--- Test 1: SpeechCommands ---")
    wav, labels, _ = load_speechcommands(n_classes=10, n_samples_per_class=200)
    all_results["speechcommands"] = run_combined_experiment_audio(
        wav, labels, args.n_folds)

    # Test 2: ESC-50
    print("\n--- Test 2: ESC-50 ---")
    wav, labels, _ = load_esc50(n_classes=10, n_samples_per_class=40)
    all_results["esc50"] = run_combined_experiment_audio(
        wav, labels, args.n_folds)

    # Test 3: UCI HAR
    print("\n--- Test 3: UCI HAR ---")
    sig, labels, _ = load_uci_har(n_classes=6, n_samples_per_class=200)
    all_results["ucihar"] = run_combined_experiment_sensor(
        sig, labels, args.n_folds)

    # Print summary
    print("\n" + "=" * 70)
    print("COMBINED RESULTS SUMMARY")
    print("=" * 70)

    for dataset, results in all_results.items():
        print(f"\n--- {dataset} ---")
        for cond in results:
            if cond == "statistical_tests":
                continue
            acc = results[cond]["mean_accuracy"]
            std = results[cond]["std_accuracy"]
            print(f"  {cond:<30} {acc:.4f} ± {std:.4f}")

        print(f"\n  Statistical tests:")
        for test_name, test in results["statistical_tests"].items():
            sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
            print(f"    {test_name:<40} delta={test['delta_accuracy']:+.4f} p={test['p_value']:.4f} {sig}")
            print(f"      → {test['interpretation']}")

    save_results(all_results, args.output)

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)

    for dataset, results in all_results.items():
        fft_acc = results.get("fft_only", {}).get("mean_accuracy", 0)
        combined_acc = results.get("fft_plus_cbmpc", results.get("fft_plus_wc", {})).get("mean_accuracy", 0)
        test_key = "fft_plus_cbmpc_vs_fft_only" if "fft_plus_cbmpc_vs_fft_only" in results.get("statistical_tests", {}) else "fft_plus_wc_vs_fft_only"
        test = results.get("statistical_tests", {}).get(test_key, {})

        print(f"\n{dataset}:")
        print(f"  FFT alone:      {fft_acc:.4f}")
        print(f"  FFT + phase:    {combined_acc:.4f}")
        print(f"  Delta:          {test.get('delta_accuracy', 0):+.4f}")
        print(f"  p-value:        {test.get('p_value', 1):.4f}")
        print(f"  Significant:    {test.get('significant', False)}")

        if test.get("significant", False) and test.get("delta_accuracy", 0) > 0:
            print(f"  → Phase ADDS VALUE on top of amplitude.")
        elif test.get("delta_accuracy", 0) > 0:
            print(f"  → Phase helps slightly but NOT significantly.")
        else:
            print(f"  → Phase does NOT add value. Amplitude is sufficient.")


if __name__ == "__main__":
    main()
