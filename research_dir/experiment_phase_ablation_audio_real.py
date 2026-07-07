"""
Experiment 1A-REAL: Phase Ablation on REAL SpeechCommands audio

Tests Claim C1: Phase coherence captures semantic structure.
Uses REAL SpeechCommands data instead of synthetic sinusoids.

Usage:
    python3 research_dir/experiment_phase_ablation_audio_real.py
    python3 research_dir/experiment_phase_ablation_audio_real.py --n_classes 10 --n_samples_per_class 200
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.cbmpc import CBMPCExtractor
from research_dir.data_loaders import load_speechcommands
from research_dir.experiment_utils import (
    save_results, print_results_table, print_stat_tests,
    extract_fft_magnitude_audio,
)


def extract_cbmpc_features(waveforms: torch.Tensor, extractor: CBMPCExtractor) -> np.ndarray:
    """Extract CBMPC features using the extractor's forward method."""
    with torch.no_grad():
        features = extractor(waveforms)
    return features.numpy()


def extract_cbmpc_with_phase_ablation(
    waveforms: torch.Tensor,
    extractor: CBMPCExtractor,
    ablation: str = "baseline",
) -> np.ndarray:
    """Extract CBMPC features with phase ablation applied to the STFT.

    The ablation is applied to the STFT phase BEFORE computing mel spectrogram
    and modulation spectrum. This tests whether phase information matters.
    """
    n_fft = extractor.n_fft
    hop_length = extractor.hop_length
    n_mels = extractor.n_mels
    sample_rate = extractor.sample_rate

    # Step 1: STFT
    stft = torch.stft(waveforms, n_fft=n_fft, hop_length=hop_length,
                      return_complex=True)  # (B, n_freq, T)
    amplitude = stft.abs()
    phase = stft.angle()

    # Step 2: Apply ablation to STFT phase
    if ablation == "baseline":
        pass
    elif ablation == "phase_zero":
        phase = torch.zeros_like(phase)
    elif ablation == "phase_randomize":
        phase = torch.rand_like(phase) * 2 * np.pi - np.pi
    elif ablation == "phase_noise":
        phase = phase + torch.randn_like(phase) * 0.5
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
    elif ablation == "phase_noise_severe":
        phase = phase + torch.randn_like(phase) * 2.0
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
    elif ablation == "cross_band_scramble":
        # Per-sample random offsets per frequency bin
        B, n_freq, T = phase.shape
        offsets = torch.rand(B, n_freq, 1) * 2 * np.pi - np.pi
        phase = phase + offsets
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    # Step 3: Reconstruct STFT with ablated phase
    ablated_stft = amplitude * torch.exp(1j * phase)

    # Step 4: Inverse STFT to get ablated waveform
    ablated_wav = torch.istft(ablated_stft, n_fft=n_fft, hop_length=hop_length,
                              length=waveforms.shape[-1])

    # Step 5: Extract CBMPC features from ablated waveform
    with torch.no_grad():
        features = extractor(ablated_wav)
    return features.numpy()


def run_experiment(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run phase ablation experiment on real audio."""

    extractor = CBMPCExtractor(
        sample_rate=16000, n_fft=512, hop_length=256,
        n_mels=32, feature_mode="compact",
    )

    ablations = [
        "baseline",
        "phase_zero",
        "phase_randomize",
        "phase_noise",
        "phase_noise_severe",
        "cross_band_scramble",
    ]

    baselines = ["fft_magnitude", "raw_waveform"]
    all_conditions = ablations + baselines

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(waveforms, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_wav = waveforms[train_idx]
        test_wav = waveforms[test_idx]  # FIX: was train_idx
        train_labels = labels[train_idx].numpy()
        test_labels = labels[test_idx].numpy()

        for condition in all_conditions:
            try:
                if condition in ablations:
                    train_feat = extract_cbmpc_with_phase_ablation(
                        train_wav, extractor, condition)
                    test_feat = extract_cbmpc_with_phase_ablation(
                        test_wav, extractor, condition)
                elif condition == "fft_magnitude":
                    train_feat = extract_fft_magnitude_audio(train_wav, n_fft=512)
                    test_feat = extract_fft_magnitude_audio(test_wav, n_fft=512)
                elif condition == "raw_waveform":
                    # Downsample raw waveform for manageable feature count
                    # Use every 8th sample (2000 features from 16000)
                    train_feat = train_wav[:, ::8].numpy()
                    test_feat = test_wav[:, ::8].numpy()
                else:
                    continue

                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
                clf.fit(train_feat, train_labels)
                acc = clf.score(test_feat, test_labels)
                results[condition]["accuracies"].append(acc)
            except Exception as e:
                print(f"    {condition}: ERROR - {e}")
                results[condition]["accuracies"].append(0.0)

    # Compute statistics
    summary = {}
    for cond in all_conditions:
        accs = np.array(results[cond]["accuracies"])
        summary[cond] = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "accuracies": accs.tolist(),
        }

    # Statistical tests: baseline vs each ablation
    baseline_accs = np.array(results["baseline"]["accuracies"])
    summary["statistical_tests"] = {}
    for ablation in ablations:
        if ablation == "baseline":
            continue
        ablated_accs = np.array(results[ablation]["accuracies"])
        t_stat, p_value = stats.ttest_rel(baseline_accs, ablated_accs)
        delta = baseline_accs.mean() - ablated_accs.mean()
        summary["statistical_tests"][ablation] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(delta),
            "significant": bool(p_value < 0.05),
        }

    # Also test CBMPC vs baselines
    for baseline in baselines:
        base_accs = np.array(results[baseline]["accuracies"])
        t_stat, p_value = stats.ttest_rel(baseline_accs, base_accs)
        delta = baseline_accs.mean() - base_accs.mean()
        summary["statistical_tests"][f"cbmpc_vs_{baseline}"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(delta),
            "significant": bool(p_value < 0.05),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1A-REAL: Phase Ablation on Real SpeechCommands")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp1a_real_speechcommands.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 1A-REAL: Phase Ablation on REAL SpeechCommands")
    print("Claim C1: Phase coherence captures semantic structure (real audio)")
    print("=" * 70)

    print(f"\nLoading SpeechCommands ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class)...")
    waveforms, labels, class_names = load_speechcommands(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    print(f"Loaded {len(waveforms)} samples, {len(class_names)} classes")
    print(f"Classes: {class_names}")
    print(f"Waveform shape: {waveforms.shape}")

    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    results = run_experiment(waveforms, labels, n_folds=args.n_folds)

    all_conds = ["baseline", "phase_zero", "phase_randomize", "phase_noise",
                 "phase_noise_severe", "cross_band_scramble",
                 "fft_magnitude", "raw_waveform"]
    print_results_table(results, all_conds, baseline="baseline",
                        title="RESULTS (REAL SpeechCommands)")
    print_stat_tests(results["statistical_tests"])

    save_results(results, args.output)

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)
    baseline_acc = results["baseline"]["mean_accuracy"]
    fft_acc = results["fft_magnitude"]["mean_accuracy"]
    raw_acc = results["raw_waveform"]["mean_accuracy"]
    n_sig = sum(1 for k, t in results["statistical_tests"].items()
                if t["significant"] and "cbmpc_vs" not in k)

    print(f"CBMPC baseline accuracy:     {baseline_acc:.4f}")
    print(f"FFT magnitude (amp-only):    {fft_acc:.4f}")
    print(f"Raw waveform:                {raw_acc:.4f}")
    print(f"Chance level:                {1.0/args.n_classes:.4f}")
    print(f"Significant phase ablations: {n_sig}/5")
    print()

    cbmpc_vs_fft = results["statistical_tests"].get("cbmpc_vs_fft_magnitude", {})
    cbmpc_vs_raw = results["statistical_tests"].get("cbmpc_vs_raw_waveform", {})

    if baseline_acc < 1.0 / args.n_classes + 0.05:
        print("NEGATIVE RESULT: CBMPC performs at chance level on real data.")
        print("  Phase coherence does NOT capture semantic structure in real audio.")
    elif cbmpc_vs_fft.get("delta_accuracy", 0) < 0:
        print("NEGATIVE RESULT: CBMPC performs WORSE than FFT magnitude on real data.")
        print(f"  Delta: {cbmpc_vs_fft['delta_accuracy']:+.4f} (p={cbmpc_vs_fft['p_value']:.4f})")
        print("  Phase coherence does not help on real audio.")
    elif n_sig >= 3:
        print("SUPPORTS C1: Phase ablation significantly degrades real audio classification.")
        if cbmpc_vs_fft.get("significant", False) and cbmpc_vs_fft["delta_accuracy"] > 0:
            print("  CBMPC also significantly beats FFT magnitude (amplitude-only).")
        else:
            print(f"  BUT: CBMPC does not significantly beat FFT magnitude "
                  f"(delta={cbmpc_vs_fft.get('delta_accuracy', 0):+.4f}, "
                  f"p={cbmpc_vs_fft.get('p_value', 1):.4f})")
    else:
        print("MIXED RESULT: Some ablations significant, but evidence is weak.")
        print("  Phase may matter but the effect is not robust on real data.")


if __name__ == "__main__":
    main()
