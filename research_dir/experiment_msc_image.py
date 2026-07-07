"""
Experiment 3A: Validate Image MSC (PhaseCongruency vs baselines)

Tests Claim C3: The coherence principle captures semantic structure in images.

Method:
    1. Generate synthetic images with class-specific phase structure
       (same amplitude spectrum across classes, different phase only)
    2. Extract PhaseCongruency features
    3. Compare with baselines: raw pixels, FFT magnitude, FFT phase, HOG-like
    4. Train linear classifier on each condition

Success criterion: PhaseCongruency > raw pixels and FFT magnitude (p < 0.05)

Usage:
    python3 research_dir/experiment_msc_image.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.msc_image import PhaseCongruencyExtractor


def generate_synthetic_images(n_classes: int = 10, n_samples_per_class: int = 200,
                               image_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic images where classes differ ONLY in phase structure.

    All classes share the same spatial frequency content. Classes differ only
    in the phase relationships between frequency components. This ensures
    that phase is the only distinguishing feature.
    """
    n_samples = n_classes * n_samples_per_class
    images = torch.zeros(n_samples, 1, image_size, image_size)
    labels = torch.zeros(n_samples, dtype=torch.long)

    freqs_x = [2, 4, 6, 8, 10]
    freqs_y = [3, 5, 7, 9, 11]
    amps = [1.0, 0.5, 0.33, 0.25, 0.2]

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, image_size),
        torch.linspace(0, 2 * np.pi, image_size),
    )

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            img = torch.zeros(image_size, image_size)
            for i, (fx, fy, amp) in enumerate(zip(freqs_x, freqs_y, amps)):
                phase_x = c * 0.3 * (i + 1) + s * 0.005 * i
                phase_y = c * 0.2 * (i + 1) + s * 0.003 * i
                img += amp * torch.sin(fx * x_grid + phase_x) * \
                       torch.cos(fy * y_grid + phase_y)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img += 0.05 * torch.randn_like(img)
            img = torch.clamp(img, 0, 1)
            images[idx, 0] = img
            labels[idx] = c
            idx += 1

    perm = torch.randperm(n_samples)
    return images[perm], labels[perm]


def extract_fft_magnitude(images: torch.Tensor) -> torch.Tensor:
    """FFT magnitude features (amplitude-only baseline)."""
    B, C, H, W = images.shape
    fft = torch.fft.fft2(images.squeeze(1))  # (B, H, W)
    mag = fft.abs()
    # Use magnitude of low-frequency components (flatten)
    return mag.reshape(B, -1).float()


def extract_fft_phase(images: torch.Tensor) -> torch.Tensor:
    """FFT phase features (phase-only baseline)."""
    B, C, H, W = images.shape
    fft = torch.fft.fft2(images.squeeze(1))
    phase = fft.angle()
    # Circular statistics: sin and cos of phase
    return torch.cat([torch.sin(phase).reshape(B, -1),
                      torch.cos(phase).reshape(B, -1)], dim=-1).float()


def extract_gradient_histogram(images: torch.Tensor, n_bins: int = 9) -> torch.Tensor:
    """HOG-like gradient histogram features (baseline).

    Computes gradient magnitude and orientation histograms. This is a
    standard image feature that does NOT use phase information explicitly
    (though gradients implicitly encode local phase).
    """
    B, C, H, W = images.shape
    img = images.squeeze(1)  # (B, H, W)

    # Compute gradients
    gx = img[:, :, 1:] - img[:, :, :-1]  # (B, H, W-1)
    gy = img[:, 1:, :] - img[:, :-1, :]  # (B, H-1, W)

    # Pad to same size
    gx = F.pad(gx, (0, 1))  # (B, H, W)
    gy = F.pad(gy, (0, 0, 0, 1))  # (B, H, W)

    mag = torch.sqrt(gx ** 2 + gy ** 2)
    angle = torch.atan2(gy, gx)  # [-pi, pi]

    # Quantize angle into n_bins
    bin_size = 2 * np.pi / n_bins
    bin_idx = ((angle + np.pi) / bin_size).long() % n_bins

    # Histogram per image
    features = []
    for b in range(B):
        hist = torch.zeros(n_bins)
        for bin_i in range(n_bins):
            mask = bin_idx[b] == bin_i
            hist[bin_i] = mag[b][mask].sum()
        features.append(hist)
    return torch.stack(features, dim=0).float()


def run_experiment(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run image MSC validation experiment."""

    extractor = PhaseCongruencyExtractor(
        n_scales=4, n_orientations=4, image_size=32,
    )

    conditions = ["phase_congruency", "fft_magnitude", "fft_phase",
                   "gradient_histogram", "raw_pixels"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_img = images[train_idx]
        test_img = images[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        for condition in conditions:
            if condition == "phase_congruency":
                train_feat = extractor(train_img).numpy()
                test_feat = extractor(test_img).numpy()
            elif condition == "fft_magnitude":
                train_feat = extract_fft_magnitude(train_img).numpy()
                test_feat = extract_fft_magnitude(test_img).numpy()
            elif condition == "fft_phase":
                train_feat = extract_fft_phase(train_img).numpy()
                test_feat = extract_fft_phase(test_img).numpy()
            elif condition == "gradient_histogram":
                train_feat = extract_gradient_histogram(train_img).numpy()
                test_feat = extract_gradient_histogram(test_img).numpy()
            elif condition == "raw_pixels":
                train_feat = train_img.reshape(train_img.shape[0], -1).numpy()
                test_feat = test_img.reshape(test_img.shape[0], -1).numpy()

            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
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

    # Paired t-tests: phase_congruency vs each baseline
    pc_accs = np.array(results["phase_congruency"]["accuracies"])
    summary["statistical_tests"] = {}
    for condition in conditions:
        if condition == "phase_congruency":
            continue
        base_accs = np.array(results[condition]["accuracies"])
        t_stat, p_value = stats.ttest_rel(pc_accs, base_accs)
        delta = pc_accs.mean() - base_accs.mean()
        summary["statistical_tests"][condition] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(delta),
            "significant": bool(p_value < 0.05),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3A: Validate Image MSC")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp3a_msc_image.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 3A: Validate Image MSC (PhaseCongruency)")
    print("Claim C3: Coherence principle captures image semantic structure")
    print("=" * 70)

    print(f"\nGenerating synthetic images ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class)...")
    images, labels = generate_synthetic_images(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    print(f"Generated {len(images)} samples")

    print(f"\nRunning {args.n_folds}-fold cross-validation with 5 conditions...")
    results = run_experiment(images, labels, n_folds=args.n_folds)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Accuracy (mean ± std)':<25} {'Delta vs PC':<15}")
    print("-" * 65)

    pc_acc = results["phase_congruency"]["mean_accuracy"]
    for condition in ["phase_congruency", "fft_magnitude", "fft_phase",
                       "gradient_histogram", "raw_pixels"]:
        acc = results[condition]["mean_accuracy"]
        std = results[condition]["std_accuracy"]
        delta = acc - pc_acc if condition != "phase_congruency" else 0.0
        print(f"{condition:<25} {acc:.4f} ± {std:.4f}          {delta:+.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (phase_congruency vs baselines)")
    print("=" * 70)
    print(f"\n{'Baseline':<25} {'Delta':<10} {'t-stat':<10} {'p-value':<10} {'Sig?'}")
    print("-" * 65)
    for condition, test in results["statistical_tests"].items():
        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
        print(f"{condition:<25} {test['delta_accuracy']:+.4f}    {test['t_statistic']:+.3f}     {test['p_value']:.4f}    {sig}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    fft_mag_delta = pc_acc - results["fft_magnitude"]["mean_accuracy"]
    raw_delta = pc_acc - results["raw_pixels"]["mean_accuracy"]
    print(f"PhaseCongruency accuracy: {pc_acc:.4f}")
    print(f"Delta vs FFT magnitude:   {fft_mag_delta:+.4f} (amplitude-only)")
    print(f"Delta vs raw pixels:      {raw_delta:+.4f}")
    if fft_mag_delta > 0.05:
        print("SUPPORTS C3: PhaseCongruency > FFT magnitude (amplitude-only).")
        print("  Coherence captures image structure that amplitude cannot.")
    else:
        print("INSUFFICIENT EVIDENCE: PhaseCongruency does not beat FFT magnitude.")


if __name__ == "__main__":
    main()
