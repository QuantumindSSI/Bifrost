"""
Experiment 2A: Cross-Scale Coherence on Audio

Tests Claim C2: Multi-scale coherence is necessary.

Method:
    1. Generate multi-scale wavelet decomposition of audio
    2. Compute cross-scale coherence features
    3. Compare: full cross-scale vs single-scale vs cross-scale-destroyed
    4. Train linear classifier on each condition

Success criterion: Cross-scale > single-scale and cross-scale-destroyed (p < 0.05)

Usage:
    python3 research_dir/experiment_cross_scale_audio.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.cross_scale_coherence import CrossScaleCoherence
from bifrost.validation.scale_ablation import ScaleAblationHarness


def generate_synthetic_audio_multiscale(
    n_classes: int = 10,
    n_samples_per_class: int = 200,
    sample_rate: int = 16000,
    duration: float = 1.0,
    n_scales: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic audio with class-specific multi-scale structure.

    Each class has different cross-scale phase relationships — the phase
    at fine scales is coherently related to phase at coarse scales in
    a class-specific way.
    """
    n_samples = n_classes * n_samples_per_class
    T = int(sample_rate * duration)
    signals = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            t = torch.arange(T) / sample_rate
            signal = torch.zeros(T)

            # Multi-scale structure: each class has coherent phase across scales
            for scale_idx in range(n_scales):
                freq = 50 * (2 ** scale_idx)  # dyadic frequency progression
                amp = 1.0 / (scale_idx + 1)

                # Class-specific cross-scale phase relationship
                # Phase at scale s is c * 0.1 * scale_idx + s * 0.01
                phase = c * 0.1 * scale_idx + s * 0.01
                signal += amp * torch.sin(2 * np.pi * freq * t + phase)

            # Add noise
            signal += 0.05 * torch.randn(T)
            signals[idx] = signal
            labels[idx] = c
            idx += 1

    perm = torch.randperm(n_samples)
    return signals[perm], labels[perm]


def compute_multiscale_phases(
    waveforms: torch.Tensor,
    n_scales: int = 6,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute multi-scale STFT phases and amplitudes.

    Uses dyadic frequency band grouping to create scales.
    """
    B = waveforms.shape[0]

    # Compute STFT
    stft = torch.stft(
        waveforms,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    )  # (B, n_freq, T_frames)

    n_freq = stft.shape[1]

    # Group frequency bins into dyadic scales
    phases = []
    amplitudes = []

    for s in range(n_scales):
        # Dyadic frequency bands
        f_start = (n_freq * (2 ** s)) // (2 ** n_scales)
        f_end = (n_freq * (2 ** (s + 1))) // (2 ** n_scales)
        if s == n_scales - 1:
            f_end = n_freq

        if f_end <= f_start:
            f_end = f_start + 1

        # Extract band
        band_stft = stft[:, f_start:f_end, :]  # (B, band_width, T)

        # Average phase across band (circular mean)
        band_phase = band_stft.angle()
        band_amp = band_stft.abs()

        # Circular mean of phase
        mean_sin = torch.sin(band_phase).mean(dim=1)
        mean_cos = torch.cos(band_phase).mean(dim=1)
        mean_phase = torch.atan2(mean_sin, mean_cos)  # (B, T)

        # Mean amplitude
        mean_amp = band_amp.mean(dim=1)  # (B, T)

        phases.append(mean_phase)
        amplitudes.append(mean_amp)

    return phases, amplitudes


def extract_cross_scale_features(
    waveforms: torch.Tensor,
    cross_scale: CrossScaleCoherence,
    n_scales: int = 6,
) -> torch.Tensor:
    """Extract cross-scale coherence features from waveforms."""
    phases, amplitudes = compute_multiscale_phases(waveforms, n_scales=n_scales)
    features = cross_scale(phases, amplitudes)
    return features


def extract_ablated_features(
    waveforms: torch.Tensor,
    cross_scale: CrossScaleCoherence,
    ablation_harness: ScaleAblationHarness,
    ablation: str,
    n_scales: int = 6,
) -> torch.Tensor:
    """Extract cross-scale features with scale ablation applied."""
    phases, amplitudes = compute_multiscale_phases(waveforms, n_scales=n_scales)

    if ablation == "baseline":
        return cross_scale(phases, amplitudes)
    elif ablation.startswith("single_scale_"):
        scale_idx = int(ablation.split("_")[-1])
        p, a = ablation_harness.single_scale(phases, amplitudes, scale_idx)
        # For single scale, compute simple phase statistics as features
        # (can't compute cross-scale PLV with only 1 scale)
        B = p[0].shape[0]
        phase_flat = p[0].reshape(B, -1)
        amp_flat = a[0].reshape(B, -1)
        # Features: mean phase coherence, phase entropy, amplitude stats
        R = torch.abs(torch.mean(torch.exp(1j * phase_flat), dim=-1)).real  # (B,)
        amp_mean = amp_flat.mean(dim=-1)  # (B,)
        amp_std = amp_flat.std(dim=-1)  # (B,)
        return torch.stack([R, amp_mean, amp_std], dim=-1)  # (B, 3)
    elif ablation == "scale_subset_half":
        p, a = ablation_harness.scale_subset(phases, amplitudes, k=n_scales // 2)
        cs = CrossScaleCoherence(n_scales=len(p), dyadic=True)
        return cs(p, a)
    elif ablation == "scale_shuffle":
        p, a = ablation_harness.scale_shuffle(phases, amplitudes)
        return cross_scale(p, a)
    elif ablation == "cross_scale_destroy":
        p, a = ablation_harness.cross_scale_destroy(phases, amplitudes)
        return cross_scale(p, a)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def run_experiment(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    n_scales: int = 6,
    n_folds: int = 5,
) -> Dict:
    """Run cross-scale ablation experiment."""

    cross_scale = CrossScaleCoherence(n_scales=n_scales, dyadic=True)
    ablation_harness = ScaleAblationHarness(n_scales=n_scales)

    ablations = [
        "baseline",
        "single_scale_0",
        "single_scale_2",
        "single_scale_4",
        "scale_subset_half",
        "cross_scale_destroy",
    ]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(waveforms, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_wav = waveforms[train_idx]
        test_wav = waveforms[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        for ablation in ablations:
            try:
                train_feat = extract_ablated_features(
                    train_wav, cross_scale, ablation_harness, ablation, n_scales
                ).numpy()
                test_feat = extract_ablated_features(
                    test_wav, cross_scale, ablation_harness, ablation, n_scales
                ).numpy()

                # Ensure 2D
                if train_feat.ndim == 1:
                    train_feat = train_feat.reshape(1, -1)
                if test_feat.ndim == 1:
                    test_feat = test_feat.reshape(1, -1)

                # Ensure same feature dim (pad with zeros if needed)
                if train_feat.shape[1] != test_feat.shape[1]:
                    dim = max(train_feat.shape[1], test_feat.shape[1])
                    train_feat = np.pad(train_feat, ((0, 0), (0, dim - train_feat.shape[1])))
                    test_feat = np.pad(test_feat, ((0, 0), (0, dim - test_feat.shape[1])))

                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
                clf.fit(train_feat, train_labels.numpy())
                acc = clf.score(test_feat, test_labels.numpy())
                results[ablation]["accuracies"].append(acc)
            except Exception as e:
                print(f"    {ablation}: ERROR - {e}")
                results[ablation]["accuracies"].append(0.0)

    # Compute statistics
    summary = {}
    for ablation in ablations:
        accs = np.array(results[ablation]["accuracies"])
        summary[ablation] = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "accuracies": accs.tolist(),
        }

    # Paired t-tests
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

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 2A: Cross-Scale Coherence on Audio")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_scales", type=int, default=6)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str, default="research_dir/results/exp2a_cross_scale_audio.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 2A: Cross-Scale Coherence on Audio")
    print("Claim C2: Multi-scale coherence is necessary")
    print("=" * 70)

    print(f"\nGenerating synthetic audio ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class, {args.n_scales} scales)...")
    waveforms, labels = generate_synthetic_audio_multiscale(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
        n_scales=args.n_scales,
    )
    print(f"Generated {len(waveforms)} samples")

    print(f"\nRunning {args.n_folds}-fold cross-validation with 6 scale conditions...")
    results = run_experiment(waveforms, labels, n_scales=args.n_scales, n_folds=args.n_folds)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Accuracy (mean ± std)':<25} {'Delta vs baseline':<15}")
    print("-" * 65)

    baseline_acc = results["baseline"]["mean_accuracy"]
    for ablation in ["baseline", "single_scale_0", "single_scale_2", "single_scale_4",
                      "scale_subset_half", "cross_scale_destroy"]:
        acc = results[ablation]["mean_accuracy"]
        std = results[ablation]["std_accuracy"]
        delta = acc - baseline_acc if ablation != "baseline" else 0.0
        print(f"{ablation:<25} {acc:.4f} ± {std:.4f}          {delta:+.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)
    print(f"\n{'Ablation':<25} {'Delta':<10} {'t-stat':<10} {'p-value':<10} {'Sig?'}")
    print("-" * 65)
    for ablation, test in results["statistical_tests"].items():
        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
        print(f"{ablation:<25} {test['delta_accuracy']:+.4f}    {test['t_statistic']:+.3f}     {test['p_value']:.4f}    {sig}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    n_significant = sum(1 for t in results["statistical_tests"].values() if t["significant"])
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Significant ablations (p < 0.05): {n_significant}/{len(results['statistical_tests'])}")
    if n_significant >= 3:
        print("SUPPORTS C2: Multi-scale coherence is necessary for audio semantic structure.")
    else:
        print("INSUFFICIENT EVIDENCE for C2 on this dataset.")
    if n_significant < len(results['statistical_tests']):
        print("NOTE: Some ablations were not significant — see details above.")


if __name__ == "__main__":
    main()
