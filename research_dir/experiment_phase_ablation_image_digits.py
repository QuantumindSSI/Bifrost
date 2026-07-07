"""
Experiment 1B-REAL: Phase Ablation on REAL handwritten digits (sklearn)

Tests Claim C1: Phase coherence captures semantic structure in real images.
Uses sklearn digits dataset (1797 samples, 10 classes, 8x8 grayscale).

This is real image data (handwritten digits from MNIST-like source).
Not synthetic sinusoids.

Usage:
    python3 research_dir/experiment_phase_ablation_image_digits.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.msc_image import PhaseCongruencyExtractor
from research_dir.experiment_utils import (
    save_results, print_results_table, print_stat_tests,
)


def extract_pc_with_ablation(
    images: torch.Tensor,
    extractor: PhaseCongruencyExtractor,
    ablation: str = "baseline",
) -> np.ndarray:
    """Extract PhaseCongruency features with optional phase ablation."""
    B, C, H, W = images.shape

    # Convert to grayscale if needed
    if C == 3:
        gray = images.mean(dim=1)
    else:
        gray = images.squeeze(1)

    # 2D FFT
    fft2d = torch.fft.fft2(gray)
    amplitude = fft2d.abs()
    phase = fft2d.angle()

    # Apply ablation
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
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    # Reconstruct
    ablated_fft = amplitude * torch.exp(1j * phase)
    ablated_img = torch.fft.ifft2(ablated_fft).real
    ablated_img = (ablated_img - ablated_img.amin(dim=(1, 2), keepdim=True)) / \
                  (ablated_img.amax(dim=(1, 2), keepdim=True) -
                   ablated_img.amin(dim=(1, 2), keepdim=True) + 1e-8)

    ablated_img_4d = ablated_img.unsqueeze(1)
    features = extractor(ablated_img_4d)
    return features.numpy()


def extract_fft_magnitude_digits(images: torch.Tensor) -> np.ndarray:
    """Extract 2D FFT magnitude features (amplitude-only baseline)."""
    B, C, H, W = images.shape
    if C == 3:
        gray = images.mean(dim=1)
    else:
        gray = images.squeeze(1)
    fft2d = torch.fft.fft2(gray)
    mag = fft2d.abs()
    return mag.reshape(B, -1).numpy()


def run_experiment(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run phase ablation experiment on real digits."""

    extractor = PhaseCongruencyExtractor(
        n_scales=3, n_orientations=4, image_size=8,
    )

    ablations = ["baseline", "phase_zero", "phase_randomize",
                 "phase_noise", "phase_noise_severe"]
    baselines = ["fft_magnitude", "raw_pixels"]
    all_conditions = ablations + baselines

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_img = images[train_idx]
        test_img = images[test_idx]
        train_labels = labels[train_idx].numpy()
        test_labels = labels[test_idx].numpy()

        for condition in all_conditions:
            try:
                if condition in ablations:
                    train_feat = extract_pc_with_ablation(
                        train_img, extractor, condition)
                    test_feat = extract_pc_with_ablation(
                        test_img, extractor, condition)
                elif condition == "fft_magnitude":
                    train_feat = extract_fft_magnitude_digits(train_img)
                    test_feat = extract_fft_magnitude_digits(test_img)
                elif condition == "raw_pixels":
                    train_feat = train_img.reshape(train_img.shape[0], -1).numpy()
                    test_feat = test_img.reshape(test_img.shape[0], -1).numpy()
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
    for cond in all_conditions:
        accs = np.array(results[cond]["accuracies"])
        summary[cond] = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "accuracies": accs.tolist(),
        }

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

    for baseline in baselines:
        base_accs = np.array(results[baseline]["accuracies"])
        t_stat, p_value = stats.ttest_rel(baseline_accs, base_accs)
        delta = baseline_accs.mean() - base_accs.mean()
        summary["statistical_tests"][f"pc_vs_{baseline}"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(delta),
            "significant": bool(p_value < 0.05),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1B-REAL: Phase Ablation on Real Digits")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp1b_real_digits.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 1B-REAL: Phase Ablation on REAL handwritten digits")
    print("Claim C1: Phase coherence captures structure (real images)")
    print("=" * 70)

    print("\nLoading sklearn digits dataset...")
    digits = load_digits()
    images = torch.tensor(digits.data).float().reshape(-1, 1, 8, 8) / 16.0  # normalize to [0,1]
    labels = torch.tensor(digits.target).long()
    print(f"Loaded {len(images)} samples, 10 classes")
    print(f"Image shape: {images.shape}")

    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    results = run_experiment(images, labels, n_folds=args.n_folds)

    all_conds = ["baseline", "phase_zero", "phase_randomize", "phase_noise",
                 "phase_noise_severe", "fft_magnitude", "raw_pixels"]
    print_results_table(results, all_conds, baseline="baseline",
                        title="RESULTS (REAL handwritten digits)")
    print_stat_tests(results["statistical_tests"])

    save_results(results, args.output)

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)
    baseline_acc = results["baseline"]["mean_accuracy"]
    fft_acc = results["fft_magnitude"]["mean_accuracy"]
    raw_acc = results["raw_pixels"]["mean_accuracy"]
    n_sig = sum(1 for k, t in results["statistical_tests"].items()
                if t["significant"] and "pc_vs" not in k)

    print(f"PhaseCongruency baseline:    {baseline_acc:.4f}")
    print(f"FFT magnitude (amp-only):    {fft_acc:.4f}")
    print(f"Raw pixels:                  {raw_acc:.4f}")
    print(f"Chance level:                0.1000")
    print(f"Significant phase ablations: {n_sig}/4")
    print()

    pc_vs_fft = results["statistical_tests"].get("pc_vs_fft_magnitude", {})
    pc_vs_raw = results["statistical_tests"].get("pc_vs_raw_pixels", {})

    if baseline_acc < 0.15:
        print("NEGATIVE RESULT: PhaseCongruency at chance on real digits.")
    elif pc_vs_fft.get("delta_accuracy", 0) < 0:
        print(f"NEGATIVE RESULT: PhaseCongruency WORSE than FFT magnitude ({pc_vs_fft['delta_accuracy']:+.4f}).")
    elif n_sig >= 3:
        print("SUPPORTS C1: Phase ablation degrades real digit classification.")
    else:
        print("MIXED RESULT: Evidence is weak on real digits.")


if __name__ == "__main__":
    main()
