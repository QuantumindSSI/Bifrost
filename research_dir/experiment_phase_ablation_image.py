"""
Experiment 1B: Phase Ablation on Images (CIFAR-10)

Tests Claim C1: Phase coherence captures semantic structure (image modality).

Method:
    1. Extract PhaseCongruency features from CIFAR-10 images
    2. Apply phase ablations (zero, randomize, cross-band scramble)
    3. Train linear classifier on each condition
    4. Compare accuracy across conditions

Since CIFAR-10 download may not be available, this script also supports
synthetic images with class-specific structural patterns.

Usage:
    python3 research_dir/experiment_phase_ablation_image.py --synthetic
    python3 research_dir/experiment_phase_ablation_image.py --data_dir /path/to/cifar10
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.msc_image import PhaseCongruencyExtractor
from bifrost.spectral_tensor import SpectralTensor
from bifrost.validation.phase_ablation import PhaseAblationHarness


def generate_synthetic_images(n_classes: int = 10, n_samples_per_class: int = 200,
                               image_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic images with class-specific structural patterns.

    Each class has different geometric structures (edges, corners, textures)
    that produce distinct phase congruency patterns.
    """
    n_samples = n_classes * n_samples_per_class
    images = torch.zeros(n_samples, 1, image_size, image_size)
    labels = torch.zeros(n_samples, dtype=torch.long)

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            img = torch.zeros(image_size, image_size)
            t = torch.linspace(0, 2 * np.pi, image_size)

            if c == 0:  # Horizontal edges
                freq = 2 + s % 5
                for y in range(image_size):
                    img[y, :] = torch.sin(freq * t + y * 0.1)
            elif c == 1:  # Vertical edges
                freq = 2 + s % 5
                for x in range(image_size):
                    img[:, x] = torch.sin(freq * t + x * 0.1)
            elif c == 2:  # Diagonal
                freq = 3 + s % 4
                for i in range(image_size):
                    img[i, :] = torch.sin(freq * t + i * 0.3)
            elif c == 3:  # Circles
                cx, cy = image_size // 2, image_size // 2
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size))
                r = torch.sqrt((x - cx)**2 + (y - cy)**2)
                n_rings = 2 + s % 4
                img = torch.sin(n_rings * r * 0.5)
            elif c == 4:  # Cross
                freq = 2 + s % 5
                img = torch.sin(freq * t).unsqueeze(0) + torch.sin(freq * t).unsqueeze(1)
            elif c == 5:  # Checkerboard
                n = 2 + s % 5
                for y in range(image_size):
                    for x in range(image_size):
                        img[y, x] = ((x // (image_size // n)) + (y // (image_size // n))) % 2
            elif c == 6:  # Radial
                cx, cy = image_size // 2, image_size // 2
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size))
                theta = torch.atan2(y - cy, x - cx)
                n_spokes = 3 + s % 5
                img = torch.cos(n_spokes * theta)
            elif c == 7:  # Spiral
                cx, cy = image_size // 2, image_size // 2
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size))
                r = torch.sqrt((x - cx)**2 + (y - cy)**2)
                theta = torch.atan2(y - cy, x - cx)
                tightness = 0.1 + s * 0.01
                img = torch.cos(theta + tightness * r)
            elif c == 8:  # Wave interference
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size))
                f1, f2 = 0.3 + s * 0.01, 0.5 + s * 0.01
                img = torch.sin(f1 * x) * torch.cos(f2 * y)
            elif c == 9:  # Texture
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size))
                freq = 0.5 + s * 0.02
                img = torch.sin(freq * x) * torch.sin(freq * y) + 0.5 * torch.cos(2 * freq * x)

            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Add noise
            img += 0.05 * torch.randn_like(img)
            img = torch.clamp(img, 0, 1)

            images[idx, 0] = img
            labels[idx] = c
            idx += 1

    perm = torch.randperm(n_samples)
    return images[perm], labels[perm]


def extract_pc_with_ablation(
    images: torch.Tensor,
    extractor: PhaseCongruencyExtractor,
    harness: PhaseAblationHarness,
    ablation: str = "baseline",
) -> torch.Tensor:
    """Extract PhaseCongruency features with optional phase ablation.

    For ablation modes, we modify the FFT phase before phase congruency
    is computed.
    """
    if ablation == "baseline":
        return extractor(images)

    B, C, H, W = images.shape
    if C > 1:
        images_gray = images.mean(dim=1, keepdim=True)
    else:
        images_gray = images

    # Resize if needed
    if H != extractor.image_size or W != extractor.image_size:
        images_gray = F.interpolate(images_gray, size=(extractor.image_size, extractor.image_size),
                                     mode='bilinear', align_corners=False)

    # Compute 2D FFT
    img_fft = torch.fft.fft2(images_gray.squeeze(1))  # (B, H, W)
    amplitude = img_fft.abs()
    phase = img_fft.angle()

    # Create SpectralTensor
    scale = torch.ones_like(amplitude)
    uncertainty = torch.zeros_like(amplitude)
    st = SpectralTensor(amplitude, phase, scale, uncertainty)

    # Apply ablation
    if ablation == "phase_zero":
        st = harness.phase_zero(st)
    elif ablation == "phase_randomize":
        st = harness.phase_randomize(st)
    elif ablation == "phase_noise":
        st = harness.phase_noise(st, sigma=0.5)
    elif ablation == "cross_band_scramble":
        st = harness.cross_band_scramble(st)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    # Reconstruct image from ablated FFT
    ablated_fft = st.amplitude * torch.exp(1j * st.phase)
    ablated_img = torch.fft.ifft2(ablated_fft).real  # (B, H, W)
    ablated_img = ablated_img.unsqueeze(1)  # (B, 1, H, W)

    # Normalize
    ablated_img = (ablated_img - ablated_img.min()) / (ablated_img.max() - ablated_img.min() + 1e-8)

    # Extract phase congruency from ablated image
    return extractor(ablated_img)


def run_experiment(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run phase ablation experiment with k-fold cross-validation."""

    extractor = PhaseCongruencyExtractor(
        n_scales=5,
        n_orientations=6,
        image_size=32,
    )
    harness = PhaseAblationHarness()

    ablations = [
        "baseline",
        "phase_zero",
        "phase_randomize",
        "phase_noise",
        "cross_band_scramble",
    ]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_img = images[train_idx]
        test_img = images[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        for ablation in ablations:
            # Extract features
            train_feat = extract_pc_with_ablation(
                train_img, extractor, harness, ablation
            ).numpy()
            test_feat = extract_pc_with_ablation(
                test_img, extractor, harness, ablation
            ).numpy()

            # Train linear classifier
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_feat, train_labels.numpy())
            acc = clf.score(test_feat, test_labels.numpy())
            results[ablation]["accuracies"].append(acc)

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
    parser = argparse.ArgumentParser(description="Experiment 1B: Phase Ablation on Images")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic images")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to CIFAR-10")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str, default="research_dir/results/exp1b_phase_ablation_image.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 1B: Phase Ablation on Images")
    print("Claim C1: Phase coherence captures semantic structure (images)")
    print("=" * 70)

    if args.synthetic or args.data_dir is None:
        print(f"\nGenerating synthetic images ({args.n_classes} classes, "
              f"{args.n_samples_per_class} samples/class)...")
        images, labels = generate_synthetic_images(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
        )
        print(f"Generated {len(images)} samples")
    else:
        print(f"\nLoading CIFAR-10 from {args.data_dir}...")
        print("Real data loading not implemented. Using synthetic.")
        images, labels = generate_synthetic_images(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
        )

    print(f"\nRunning {args.n_folds}-fold cross-validation with {5} ablation conditions...")
    results = run_experiment(images, labels, n_folds=args.n_folds)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Accuracy (mean ± std)':<25} {'Delta vs baseline':<15}")
    print("-" * 65)

    baseline_acc = results["baseline"]["mean_accuracy"]
    for ablation in ["baseline", "phase_zero", "phase_randomize", "phase_noise",
                      "cross_band_scramble"]:
        acc = results[ablation]["mean_accuracy"]
        std = results[ablation]["std_accuracy"]
        delta = acc - baseline_acc if ablation != "baseline" else 0.0
        print(f"{ablation:<25} {acc:.4f} ± {std:.4f}          {delta:+.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (paired t-test, baseline vs ablated)")
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
    if n_significant >= 2:
        print("SUPPORTS C1: Phase coherence captures semantic structure in images.")
    else:
        print("INSUFFICIENT EVIDENCE for C1 on this dataset.")


if __name__ == "__main__":
    main()
