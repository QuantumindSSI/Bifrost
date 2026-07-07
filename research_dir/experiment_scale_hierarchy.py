"""
Experiment 2C: Scale Hierarchy Analysis

Tests Claim C2 (stronger): Cross-scale coherence profiles are semantic —
samples from the same class have more similar cross-scale coherence profiles
than samples from different classes.

Method:
    1. Generate synthetic audio with class-specific multi-scale structure
    2. Compute cross-scale coherence features for each sample
    3. Measure: do same-class samples have more similar coherence profiles
       than different-class samples?
    4. Use silhouette score and Mann-Whitney U test

Success criterion: Same-class coherence profile similarity > different-class
similarity (p < 0.05, Mann-Whitney U).

Usage:
    python3 research_dir/experiment_scale_hierarchy.py --modality audio
    python3 research_dir/experiment_scale_hierarchy.py --modality image
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import silhouette_score

from bifrost.cross_scale_coherence import CrossScaleCoherence


def generate_synthetic_audio_multiscale(
    n_classes: int = 10, n_samples_per_class: int = 200,
    sample_rate: int = 16000, duration: float = 1.0, n_scales: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as experiment 2A."""
    n_samples = n_classes * n_samples_per_class
    T = int(sample_rate * duration)
    signals = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            t = torch.arange(T) / sample_rate
            signal = torch.zeros(T)
            for scale_idx in range(n_scales):
                freq = 50 * (2 ** scale_idx)
                amp = 1.0 / (scale_idx + 1)
                phase = c * 0.1 * scale_idx + s * 0.01
                signal += amp * torch.sin(2 * np.pi * freq * t + phase)
            signal += 0.05 * torch.randn(T)
            signals[idx] = signal
            labels[idx] = c
            idx += 1

    perm = torch.randperm(n_samples)
    return signals[perm], labels[perm]


def generate_synthetic_images_multiscale(
    n_classes: int = 10, n_samples_per_class: int = 200,
    image_size: int = 32, n_scales: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as experiment 2B."""
    n_samples = n_classes * n_samples_per_class
    images = torch.zeros(n_samples, 1, image_size, image_size)
    labels = torch.zeros(n_samples, dtype=torch.long)

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, image_size),
        torch.linspace(0, 2 * np.pi, image_size),
    )

    scale_freqs = [(2 ** (s + 1), 2 ** (s + 1) + 1) for s in range(n_scales)]
    amps = [1.0 / (s + 1) for s in range(n_scales)]

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            img = torch.zeros(image_size, image_size)
            for scale_idx, ((fx, fy), amp) in enumerate(zip(scale_freqs, amps)):
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


def compute_audio_cross_scale_features(
    waveforms: torch.Tensor, n_scales: int = 6,
) -> torch.Tensor:
    """Compute cross-scale coherence features for audio."""
    cross_scale = CrossScaleCoherence(n_scales=n_scales, dyadic=True)

    B = waveforms.shape[0]
    stft = torch.stft(waveforms, n_fft=1024, hop_length=512,
                      return_complex=True)
    n_freq = stft.shape[1]

    phases = []
    amplitudes = []
    for s in range(n_scales):
        f_start = (n_freq * (2 ** s)) // (2 ** n_scales)
        f_end = (n_freq * (2 ** (s + 1))) // (2 ** n_scales)
        if s == n_scales - 1:
            f_end = n_freq
        if f_end <= f_start:
            f_end = f_start + 1

        band_stft = stft[:, f_start:f_end, :]
        band_phase = band_stft.angle()
        band_amp = band_stft.abs()

        mean_sin = torch.sin(band_phase).mean(dim=1)
        mean_cos = torch.cos(band_phase).mean(dim=1)
        mean_phase = torch.atan2(mean_sin, mean_cos)
        mean_amp = band_amp.mean(dim=1)

        phases.append(mean_phase)
        amplitudes.append(mean_amp)

    features = cross_scale(phases, amplitudes)
    return features


def compute_image_cross_scale_features(
    images: torch.Tensor, n_scales: int = 4,
) -> torch.Tensor:
    """Compute cross-scale coherence features for images."""
    cross_scale = CrossScaleCoherence(n_scales=n_scales, dyadic=True)

    B, C, H, W = images.shape
    img_gray = images.squeeze(1) if C == 1 else images.mean(dim=1)
    fft2d = torch.fft.fft2(img_gray)
    amplitude = fft2d.abs()
    phase = fft2d.angle()

    cy, cx = H // 2, W // 2
    y_c = torch.arange(H).float().unsqueeze(1).expand(H, W)
    x_c = torch.arange(W).float().unsqueeze(0).expand(H, W)
    r = torch.sqrt((y_c - cy) ** 2 + (x_c - cx) ** 2)
    max_r = r.max().item()

    phases = []
    amplitudes = []
    for s in range(n_scales):
        r_start = max_r * s / n_scales
        r_end = max_r * (s + 1) / n_scales
        mask = (r >= r_start) & (r < r_end)
        if mask.sum() == 0:
            phases.append(torch.zeros(B, 1))
            amplitudes.append(torch.zeros(B, 1))
            continue
        band_phase = phase[:, mask]
        band_amp = amplitude[:, mask]
        mean_sin = torch.sin(band_phase).mean(dim=-1)
        mean_cos = torch.cos(band_phase).mean(dim=-1)
        mean_phase = torch.atan2(mean_sin, mean_cos)
        mean_amp = band_amp.mean(dim=-1)
        phases.append(mean_phase.unsqueeze(-1))
        amplitudes.append(mean_amp.unsqueeze(-1))

    features = cross_scale(phases, amplitudes)
    return features


def run_experiment(features: np.ndarray, labels: np.ndarray) -> Dict:
    """Analyze whether cross-scale coherence profiles are semantic."""
    n_samples = len(labels)

    # 1. Silhouette score: how well do coherence profiles cluster by class?
    # Sample if too large
    if n_samples > 2000:
        idx = np.random.RandomState(42).choice(n_samples, 2000, replace=False)
        feat_sub = features[idx]
        labels_sub = labels[idx]
    else:
        feat_sub = features
        labels_sub = labels

    silhouette = silhouette_score(feat_sub, labels_sub, sample_size=min(1000, len(labels_sub)))

    # 2. Same-class vs different-class profile similarity
    # Compute cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(features)

    # Extract same-class and different-class similarities
    same_class_sims = []
    diff_class_sims = []
    for i in range(min(n_samples, 1000)):
        for j in range(i + 1, min(n_samples, 1000)):
            if labels[i] == labels[j]:
                same_class_sims.append(sim_matrix[i, j])
            else:
                diff_class_sims.append(sim_matrix[i, j])

    same_class_sims = np.array(same_class_sims)
    diff_class_sims = np.array(diff_class_sims)

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(same_class_sims, diff_class_sims,
                                          alternative="greater")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((same_class_sims.std() ** 2 + diff_class_sims.std() ** 2) / 2)
    cohens_d = (same_class_sims.mean() - diff_class_sims.mean()) / (pooled_std + 1e-8)

    return {
        "silhouette_score": float(silhouette),
        "same_class_similarity": {
            "mean": float(same_class_sims.mean()),
            "std": float(same_class_sims.std()),
            "n": len(same_class_sims),
        },
        "diff_class_similarity": {
            "mean": float(diff_class_sims.mean()),
            "std": float(diff_class_sims.std()),
            "n": len(diff_class_sims),
        },
        "mann_whitney_u": {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        },
        "cohens_d": float(cohens_d),
        "delta_similarity": float(same_class_sims.mean() - diff_class_sims.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 2C: Scale Hierarchy Analysis")
    parser.add_argument("--modality", choices=["audio", "image", "both"], default="both")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_scales", type=int, default=6)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp2c_scale_hierarchy.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 2C: Scale Hierarchy Analysis")
    print("Claim C2 (stronger): Cross-scale coherence profiles are semantic")
    print("=" * 70)

    all_results = {}

    if args.modality in ("audio", "both"):
        print(f"\n--- AUDIO ---")
        print(f"Generating synthetic audio ({args.n_classes} classes, "
              f"{args.n_samples_per_class} samples/class)...")
        waveforms, labels = generate_synthetic_audio_multiscale(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
            n_scales=args.n_scales,
        )
        print(f"Generated {len(waveforms)} samples")

        print("Computing cross-scale coherence features...")
        features = compute_audio_cross_scale_features(waveforms, n_scales=args.n_scales)
        print(f"Features: {features.shape}")

        print("Analyzing profile similarity...")
        audio_results = run_experiment(features.numpy(), labels.numpy())
        all_results["audio"] = audio_results

    if args.modality in ("image", "both"):
        print(f"\n--- IMAGE ---")
        n_scales_img = min(args.n_scales, 4)
        print(f"Generating synthetic images ({args.n_classes} classes, "
              f"{args.n_samples_per_class} samples/class, {n_scales_img} scales)...")
        images, labels = generate_synthetic_images_multiscale(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
            n_scales=n_scales_img,
        )
        print(f"Generated {len(images)} samples")

        print("Computing cross-scale coherence features...")
        features = compute_image_cross_scale_features(images, n_scales=n_scales_img)
        print(f"Features: {features.shape}")

        print("Analyzing profile similarity...")
        image_results = run_experiment(features.numpy(), labels.numpy())
        all_results["image"] = image_results

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for modality, mod_results in all_results.items():
        print(f"\n--- {modality.upper()} ---")
        print(f"Silhouette score:        {mod_results['silhouette_score']:.4f}")
        print(f"Same-class similarity:   {mod_results['same_class_similarity']['mean']:.4f} "
              f"± {mod_results['same_class_similarity']['std']:.4f}")
        print(f"Diff-class similarity:   {mod_results['diff_class_similarity']['mean']:.4f} "
              f"± {mod_results['diff_class_similarity']['std']:.4f}")
        print(f"Delta:                   {mod_results['delta_similarity']:+.4f}")
        print(f"Cohen's d:               {mod_results['cohens_d']:.4f}")
        mw = mod_results["mann_whitney_u"]
        sig = "***" if mw["p_value"] < 0.001 else "**" if mw["p_value"] < 0.01 else "*" if mw["p_value"] < 0.05 else "ns"
        print(f"Mann-Whitney U:          U={mw['u_statistic']:.1f}, p={mw['p_value']:.6f} {sig}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    n_supported = 0
    for modality, mod_results in all_results.items():
        mw = mod_results["mann_whitney_u"]
        if mw["significant"] and mod_results["delta_similarity"] > 0:
            n_supported += 1
            print(f"{modality}: SUPPORTS C2 — same-class profiles more similar (p={mw['p_value']:.6f})")
        else:
            print(f"{modality}: INSUFFICIENT EVIDENCE (p={mw['p_value']:.6f})")

    if n_supported >= 2:
        print("\nSUPPORTS C2 (stronger): Cross-scale coherence profiles are semantic.")
    elif n_supported >= 1:
        print("\nPARTIAL SUPPORT: Cross-scale coherence profiles are semantic in one modality.")
    else:
        print("\nINSUFFICIENT EVIDENCE for C2 scale hierarchy claim.")


if __name__ == "__main__":
    main()
