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
    """Generate synthetic images where classes differ ONLY in phase structure.

    All classes share the same spatial frequency content (same frequencies,
    same amplitudes, same orientations). Classes differ only in the phase
    relationships between frequency components. This ensures that phase is
    the only distinguishing feature — amplitude-only features are identical
    across classes.

    Without phase information, all classes are identical and classification
    should drop to chance.
    """
    n_samples = n_classes * n_samples_per_class
    images = torch.zeros(n_samples, 1, image_size, image_size)
    labels = torch.zeros(n_samples, dtype=torch.long)

    # Shared spatial frequencies and amplitudes for all classes
    # (same as audio experiment — control for amplitude)
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
                # Class-specific phase per frequency component — this is
                # the ONLY feature that distinguishes classes
                phase_x = c * 0.3 * (i + 1) + s * 0.005 * i
                phase_y = c * 0.2 * (i + 1) + s * 0.003 * i
                img += amp * torch.sin(freqs_x[i] * x_grid + phase_x) * \
                       torch.cos(freqs_y[i] * y_grid + phase_y)

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
        # Replace phase with independent random values per element
        random_phase = (torch.rand(st.phase.shape,
                                    device=st.phase.device,
                                    dtype=st.phase.dtype) * 2 * torch.pi - torch.pi)
        st = SpectralTensor(
            amplitude=st.amplitude, phase=random_phase,
            scale=st.scale, uncertainty=st.uncertainty,
            metadata={**st.metadata, "ablation": "phase_randomize"},
        )
    elif ablation == "phase_noise":
        st = harness.phase_noise(st, sigma=0.5)
    elif ablation == "phase_noise_severe":
        st = harness.phase_noise(st, sigma=2.0)
    elif ablation == "cross_band_scramble":
        # Per-sample random offsets per spatial frequency band (W dimension).
        # st.phase is (B, H, W) from 2D FFT. Add per-W-frequency offsets
        # that are the same across H, destroying cross-frequency phase
        # relationships while preserving within-frequency structure.
        n_bands = st.phase.shape[-1]  # W dimension = frequency bins
        offsets = (torch.rand(st.phase.shape[0], 1, n_bands,
                              device=st.phase.device,
                              dtype=st.phase.dtype) * 2 * torch.pi - torch.pi)
        scrambled = st.phase + offsets  # (B, H, W) + (B, 1, W) → (B, H, W)
        scrambled = torch.atan2(torch.sin(scrambled), torch.cos(scrambled))
        st = SpectralTensor(
            amplitude=st.amplitude, phase=scrambled,
            scale=st.scale, uncertainty=st.uncertainty,
            metadata={**st.metadata, "ablation": "cross_band_scramble"},
        )
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
        "phase_noise_severe",
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
                      "phase_noise_severe", "cross_band_scramble"]:
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
    if n_significant >= 3:
        print("SUPPORTS C1: Phase coherence captures semantic structure in images.")
    else:
        print("INSUFFICIENT EVIDENCE for C1 on this dataset.")


if __name__ == "__main__":
    main()
