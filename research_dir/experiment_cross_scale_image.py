"""
Experiment 2B: Cross-Scale Coherence on Images

Tests Claim C2: Multi-scale coherence is necessary (image modality).

Method:
    1. Generate multi-scale phase decomposition of images (2D FFT bands)
    2. Compute cross-scale coherence features
    3. Compare: full cross-scale vs single-scale vs cross-scale-destroyed
    4. Train linear classifier on each condition

Success criterion: Cross-scale > single-scale and cross-scale-destroyed (p < 0.05)

Usage:
    python3 research_dir/experiment_cross_scale_image.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.cross_scale_coherence import CrossScaleCoherence
from bifrost.validation.scale_ablation import ScaleAblationHarness


def generate_synthetic_images_multiscale(
    n_classes: int = 10,
    n_samples_per_class: int = 200,
    image_size: int = 32,
    n_scales: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic images with class-specific multi-scale phase structure.

    All classes share the same spatial frequency content. Classes differ
    only in the cross-scale phase relationships. Each scale band has
    class-specific phase that is coherently related across scales.
    """
    n_samples = n_classes * n_samples_per_class
    images = torch.zeros(n_samples, 1, image_size, image_size)
    labels = torch.zeros(n_samples, dtype=torch.long)

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, image_size),
        torch.linspace(0, 2 * np.pi, image_size),
    )

    # Shared frequencies per scale (dyadic progression)
    scale_freqs = []
    for s in range(n_scales):
        fx = 2 ** (s + 1)
        fy = 2 ** (s + 1) + 1
        scale_freqs.append((fx, fy))

    amps = [1.0 / (s + 1) for s in range(n_scales)]

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            img = torch.zeros(image_size, image_size)
            for scale_idx, ((fx, fy), amp) in enumerate(zip(scale_freqs, amps)):
                # Class-specific cross-scale phase relationship
                phase = c * 0.15 * (scale_idx + 1) + s * 0.005 * scale_idx
                img += amp * torch.sin(fx * x_grid + phase) * \
                       torch.cos(fy * y_grid + phase * 0.7)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img += 0.05 * torch.randn_like(img)
            img = torch.clamp(img, 0, 1)
            images[idx, 0] = img
            labels[idx] = c
            idx += 1

    perm = torch.randperm(n_samples)
    return images[perm], labels[perm]


def compute_multiscale_phases_image(
    images: torch.Tensor,
    n_scales: int = 4,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute multi-scale phases from 2D FFT of images.

    Groups spatial frequency bins into dyadic scale bands and computes
    circular mean phase per band.
    """
    B, C, H, W = images.shape
    img_gray = images.squeeze(1) if C == 1 else images.mean(dim=1)

    # 2D FFT
    fft2d = torch.fft.fft2(img_gray)  # (B, H, W)
    amplitude = fft2d.abs()
    phase = fft2d.angle()

    # Compute radial frequency distance from center
    cy, cx = H // 2, W // 2
    y_coords = torch.arange(H).float().unsqueeze(1).expand(H, W)
    x_coords = torch.arange(W).float().unsqueeze(0).expand(H, W)
    r = torch.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    max_r = r.max().item()

    phases = []
    amplitudes = []

    for s in range(n_scales):
        r_start = max_r * s / n_scales
        r_end = max_r * (s + 1) / n_scales
        mask = (r >= r_start) & (r < r_end)

        if mask.sum() == 0:
            phases.append(torch.zeros(B))
            amplitudes.append(torch.zeros(B))
            continue

        # Extract phases and amplitudes at this scale band
        band_phase = phase[:, mask]  # (B, N_band)
        band_amp = amplitude[:, mask]  # (B, N_band)

        # Circular mean of phase
        mean_sin = torch.sin(band_phase).mean(dim=-1)
        mean_cos = torch.cos(band_phase).mean(dim=-1)
        mean_phase = torch.atan2(mean_sin, mean_cos)  # (B,)

        # Mean amplitude
        mean_amp = band_amp.mean(dim=-1)  # (B,)

        phases.append(mean_phase)
        amplitudes.append(mean_amp)

    return phases, amplitudes


def extract_cross_scale_features(
    images: torch.Tensor,
    cross_scale: CrossScaleCoherence,
    n_scales: int = 4,
) -> torch.Tensor:
    """Extract cross-scale coherence features from images.

    Combines cross-scale PLV features with per-scale phase/amplitude
    statistics for richer representation.
    """
    phases, amplitudes = compute_multiscale_phases_image(images, n_scales)
    phases_2d = [p.unsqueeze(-1) for p in phases]  # (B, 1)
    amplitudes_2d = [a.unsqueeze(-1) for a in amplitudes]

    # Cross-scale coherence features
    cs_features = cross_scale(phases_2d, amplitudes_2d)  # (B, n_pairs * 2)

    # Per-scale features: phase (sin, cos) and amplitude
    per_scale = []
    for p, a in zip(phases_2d, amplitudes_2d):
        per_scale.append(torch.sin(p))  # (B, 1)
        per_scale.append(torch.cos(p))  # (B, 1)
        per_scale.append(a)             # (B, 1)

    per_scale_feat = torch.cat(per_scale, dim=-1)  # (B, n_scales * 3)

    return torch.cat([cs_features, per_scale_feat], dim=-1)


def extract_ablated_features(
    images: torch.Tensor,
    cross_scale: CrossScaleCoherence,
    ablation_harness: ScaleAblationHarness,
    ablation: str,
    n_scales: int = 4,
) -> torch.Tensor:
    """Extract cross-scale features with scale ablation applied."""
    phases, amplitudes = compute_multiscale_phases_image(images, n_scales)
    phases_2d = [p.unsqueeze(-1) for p in phases]
    amplitudes_2d = [a.unsqueeze(-1) for a in amplitudes]

    if ablation == "baseline":
        return extract_cross_scale_features(images, cross_scale, n_scales)
    elif ablation.startswith("single_scale_"):
        scale_idx = int(ablation.split("_")[-1])
        p, a = ablation_harness.single_scale(phases_2d, amplitudes_2d, scale_idx)
        # Single scale: use phase (sin, cos) and amplitude directly
        # (no cross-scale coherence possible with 1 scale)
        B = p[0].shape[0]
        feat = torch.cat([
            torch.sin(p[0]), torch.cos(p[0]), a[0]
        ], dim=-1)  # (B, 3)
        # Pad to match baseline feature dim for fair comparison
        return feat
    elif ablation == "scale_subset_half":
        p, a = ablation_harness.scale_subset(phases_2d, amplitudes_2d, k=n_scales // 2)
        # Use cross-scale coherence on the subset
        cs = CrossScaleCoherence(n_scales=len(p), dyadic=True)
        cs_feat = cs(p, a)
        per_scale = []
        for pp, aa in zip(p, a):
            per_scale.append(torch.sin(pp))
            per_scale.append(torch.cos(pp))
            per_scale.append(aa)
        per_scale_feat = torch.cat(per_scale, dim=-1)
        return torch.cat([cs_feat, per_scale_feat], dim=-1)
    elif ablation == "cross_scale_destroy":
        p, a = ablation_harness.cross_scale_destroy(phases_2d, amplitudes_2d)
        cs_feat = cross_scale(p, a)
        per_scale = []
        for pp, aa in zip(p, a):
            per_scale.append(torch.sin(pp))
            per_scale.append(torch.cos(pp))
            per_scale.append(aa)
        per_scale_feat = torch.cat(per_scale, dim=-1)
        return torch.cat([cs_feat, per_scale_feat], dim=-1)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def run_experiment(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_scales: int = 4,
    n_folds: int = 5,
) -> Dict:
    cross_scale = CrossScaleCoherence(n_scales=n_scales, dyadic=True)
    ablation_harness = ScaleAblationHarness(n_scales=n_scales)

    ablations = [
        "baseline",
        "single_scale_0",
        "single_scale_1",
        "scale_subset_half",
        "cross_scale_destroy",
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
            try:
                train_feat = extract_ablated_features(
                    train_img, cross_scale, ablation_harness, ablation, n_scales
                ).numpy()
                test_feat = extract_ablated_features(
                    test_img, cross_scale, ablation_harness, ablation, n_scales
                ).numpy()

                if train_feat.ndim == 1:
                    train_feat = train_feat.reshape(1, -1)
                if test_feat.ndim == 1:
                    test_feat = test_feat.reshape(1, -1)

                if train_feat.shape[1] != test_feat.shape[1]:
                    dim = max(train_feat.shape[1], test_feat.shape[1])
                    train_feat = np.pad(train_feat, ((0, 0), (0, dim - train_feat.shape[1])))
                    test_feat = np.pad(test_feat, ((0, 0), (0, dim - test_feat.shape[1])))

                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
                clf.fit(train_feat, train_labels.numpy())
                acc = clf.score(test_feat, test_labels.numpy())
                results[ablation]["accuracies"].append(acc)
            except Exception as e:
                print(f"    {ablation}: ERROR - {e}")
                results[ablation]["accuracies"].append(0.0)

    summary = {}
    for ablation in ablations:
        accs = np.array(results[ablation]["accuracies"])
        summary[ablation] = {
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

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 2B: Cross-Scale Coherence on Images")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_scales", type=int, default=4)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp2b_cross_scale_image.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 2B: Cross-Scale Coherence on Images")
    print("Claim C2: Multi-scale coherence is necessary (images)")
    print("=" * 70)

    print(f"\nGenerating synthetic images ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class, {args.n_scales} scales)...")
    images, labels = generate_synthetic_images_multiscale(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
        n_scales=args.n_scales,
    )
    print(f"Generated {len(images)} samples")

    print(f"\nRunning {args.n_folds}-fold cross-validation with 5 scale conditions...")
    results = run_experiment(images, labels, n_scales=args.n_scales, n_folds=args.n_folds)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Accuracy (mean ± std)':<25} {'Delta vs baseline':<15}")
    print("-" * 65)

    baseline_acc = results["baseline"]["mean_accuracy"]
    for ablation in ["baseline", "single_scale_0", "single_scale_1",
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    n_significant = sum(1 for t in results["statistical_tests"].values() if t["significant"])
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Significant ablations (p < 0.05): {n_significant}/{len(results['statistical_tests'])}")
    if n_significant >= 3:
        print("SUPPORTS C2: Multi-scale coherence is necessary for image semantic structure.")
    else:
        print("INSUFFICIENT EVIDENCE for C2 on this dataset.")


if __name__ == "__main__":
    main()
