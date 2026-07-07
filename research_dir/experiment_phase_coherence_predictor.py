"""
Experiment 1C: Phase Coherence as a Predictor of Classification Confidence

Tests Claim C1 (stronger): Phase coherence is not just useful for
classification — it PREDICTS classification confidence.

Method:
    1. Generate synthetic audio with class-specific phase structure
    2. Extract CBMPC features and train a classifier
    3. Compute phase coherence metrics (PLV, phase entropy, phase congruency)
       for each sample
    4. Correlate phase coherence metrics with classification confidence
       (max softmax probability)

Success criterion: Phase coherence metrics correlate with classification
confidence (r > 0.3, p < 0.05).

Usage:
    python3 research_dir/experiment_phase_coherence_predictor.py --modality audio
    python3 research_dir/experiment_phase_coherence_predictor.py --modality image
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.cbmpc import CBMPCExtractor
from bifrost.msc_image import PhaseCongruencyExtractor
from bifrost.validation.phase_metrics import PhaseCoherenceSignalMetrics


def generate_synthetic_audio(n_classes: int = 10, n_samples_per_class: int = 200,
                              sample_rate: int = 16000, duration: float = 1.0):
    """Same controlled-for-amplitude audio as experiment 1A."""
    n_samples = n_classes * n_samples_per_class
    T = int(sample_rate * duration)
    signals = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)

    base_freqs = [200, 400, 600, 800, 1000]
    base_amps = [1.0, 0.5, 0.33, 0.25, 0.2]

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            t = torch.arange(T) / sample_rate
            signal = torch.zeros(T)
            for i, (freq, amp) in enumerate(zip(base_freqs, base_amps)):
                phase_offset = c * 0.3 * (i + 1) + s * 0.005 * i
                signal += amp * torch.sin(2 * np.pi * freq * t + phase_offset)
            mod_freq = 4.0
            envelope = 1 + 0.2 * torch.sin(2 * np.pi * mod_freq * t)
            signal = signal * envelope
            signal += 0.02 * torch.randn(T)
            signals[idx] = signal
            labels[idx] = c
            idx += 1

    perm = torch.randperm(n_samples)
    return signals[perm], labels[perm]


def generate_synthetic_images(n_classes: int = 10, n_samples_per_class: int = 200,
                               image_size: int = 32):
    """Same controlled-for-amplitude images as experiment 1B."""
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


def compute_audio_coherence_metrics(
    waveforms: torch.Tensor,
    metrics_calc: PhaseCoherenceSignalMetrics,
) -> Dict[str, np.ndarray]:
    """Compute phase coherence metrics for each audio sample."""
    B = waveforms.shape[0]
    plv_values = []
    entropy_values = []
    stability_values = []

    for i in range(B):
        wav = waveforms[i:i + 1]
        stft = torch.stft(wav, n_fft=1024, hop_length=512,
                          return_complex=True)
        phase = stft.angle()  # (1, n_freq, T)

        # PLV between adjacent frequency bands
        plv = metrics_calc.phase_locking_value(
            phase[:, :-1, :], phase[:, 1:, :], dim=-1
        ).mean().item()

        # Phase entropy — returns tensor, take mean
        entropy = metrics_calc.phase_entropy(phase, dim=-1).mean().item()

        # Phase stability (temporal) — returns (1, n_freq), take mean
        stability = metrics_calc.phase_stability(phase, time_dim=-1).mean().item()

        plv_values.append(plv)
        entropy_values.append(entropy)
        stability_values.append(stability)

    return {
        "plv": np.array(plv_values),
        "phase_entropy": np.array(entropy_values),
        "phase_stability": np.array(stability_values),
    }


def compute_image_coherence_metrics(
    images: torch.Tensor,
    metrics_calc: PhaseCoherenceSignalMetrics,
) -> Dict[str, np.ndarray]:
    """Compute phase coherence metrics for each image sample."""
    B = images.shape[0]
    pc_values = []
    entropy_values = []
    plv_values = []

    for i in range(B):
        img = images[i, 0]  # (H, W)
        fft2d = torch.fft.fft2(img.unsqueeze(0))  # (1, H, W)
        phase = fft2d.angle()

        # Phase congruency across spatial frequency bands
        H, W = img.shape
        cy, cx = H // 2, W // 2
        y_c = torch.arange(H).float().unsqueeze(1).expand(H, W)
        x_c = torch.arange(W).float().unsqueeze(0).expand(H, W)
        r = torch.sqrt((y_c - cy) ** 2 + (x_c - cx) ** 2)
        max_r = r.max().item()

        band_phases = []
        band_amps = []
        for b in range(4):
            r_start = max_r * b / 4
            r_end = max_r * (b + 1) / 4
            mask = (r >= r_start) & (r < r_end)
            if mask.sum() > 0:
                band_phases.append(phase[0][mask])
                band_amps.append(fft2d.abs()[0][mask])

        if len(band_phases) >= 2:
            # Phase congruency: stack phases and amplitudes
            # Need same length per band — use circular mean per band
            mean_sins = [torch.sin(bp).mean() for bp in band_phases]
            mean_coss = [torch.cos(bp).mean() for bp in band_phases]
            mean_phases = torch.stack([torch.atan2(s, c) for s, c in zip(mean_sins, mean_coss)])
            mean_amps = torch.stack([ba.mean() for ba in band_amps])
            # phase_congruency expects (..., n_scales, ...) with scale_dim=0
            pc = metrics_calc.phase_congruency(
                mean_phases.unsqueeze(0),  # (1, n_scales)
                mean_amps.unsqueeze(0),
                scale_dim=0,
            ).mean().item()

            # PLV between first two bands
            if band_phases[0].shape[0] > 0 and band_phases[1].shape[0] > 0:
                min_len = min(band_phases[0].shape[0], band_phases[1].shape[0])
                plv = metrics_calc.phase_locking_value(
                    band_phases[0][:min_len].unsqueeze(0),
                    band_phases[1][:min_len].unsqueeze(0),
                    dim=-1
                ).item()
            else:
                plv = 0.0
        else:
            pc = 0.0
            plv = 0.0

        entropy = metrics_calc.phase_entropy(phase, dim=-1).mean().item()

        pc_values.append(pc)
        plv_values.append(plv)
        entropy_values.append(entropy)

    return {
        "phase_congruency": np.array(pc_values),
        "cross_band_plv": np.array(plv_values),
        "phase_entropy": np.array(entropy_values),
    }


def run_experiment_audio(waveforms, labels, n_folds=5):
    """Run phase coherence predictor experiment on audio."""
    extractor = CBMPCExtractor(
        sample_rate=16000, n_fft=1024, hop_length=512,
        n_mels=64, feature_mode="compact",
    )
    metrics_calc = PhaseCoherenceSignalMetrics()

    print("  Extracting CBMPC features...")
    features = extractor(waveforms).numpy()

    print("  Computing phase coherence metrics...")
    coherence_metrics = compute_audio_coherence_metrics(waveforms, metrics_calc)

    all_confidences = []
    all_metrics = {k: [] for k in coherence_metrics}
    all_correct = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(features[train_idx], labels[train_idx].numpy())

        # Get confidence (max probability)
        probs = clf.predict_proba(features[test_idx])
        confidence = probs.max(axis=1)
        predictions = clf.predict(features[test_idx])
        correct = (predictions == labels[test_idx].numpy())

        all_confidences.extend(confidence)
        all_correct.extend(correct)
        for k in coherence_metrics:
            all_metrics[k].extend(coherence_metrics[k][test_idx])

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    # Correlate coherence metrics with confidence
    results = {"modality": "audio"}
    results["overall_accuracy"] = float(all_correct.mean())
    results["mean_confidence"] = float(all_confidences.mean())

    results["correlations"] = {}
    for metric_name, metric_values in all_metrics.items():
        metric_values = np.array(metric_values)
        r, p = stats.pearsonr(metric_values, all_confidences)
        results["correlations"][metric_name] = {
            "pearson_r": float(r),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "n_samples": len(metric_values),
        }

    # Also correlate with correctness (point-biserial)
    results["correctness_correlations"] = {}
    for metric_name, metric_values in all_metrics.items():
        metric_values = np.array(metric_values)
        r, p = stats.pointbiserialr(all_correct.astype(float), metric_values)
        results["correctness_correlations"][metric_name] = {
            "point_biserial_r": float(r),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        }

    return results


def run_experiment_image(images, labels, n_folds=5):
    """Run phase coherence predictor experiment on images."""
    extractor = PhaseCongruencyExtractor(
        n_scales=4, n_orientations=4, image_size=32,
    )
    metrics_calc = PhaseCoherenceSignalMetrics()

    print("  Extracting PhaseCongruency features...")
    features = extractor(images).numpy()

    print("  Computing phase coherence metrics...")
    coherence_metrics = compute_image_coherence_metrics(images, metrics_calc)

    all_confidences = []
    all_metrics = {k: [] for k in coherence_metrics}
    all_correct = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(features[train_idx], labels[train_idx].numpy())

        probs = clf.predict_proba(features[test_idx])
        confidence = probs.max(axis=1)
        predictions = clf.predict(features[test_idx])
        correct = (predictions == labels[test_idx].numpy())

        all_confidences.extend(confidence)
        all_correct.extend(correct)
        for k in coherence_metrics:
            all_metrics[k].extend(coherence_metrics[k][test_idx])

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    results = {"modality": "image"}
    results["overall_accuracy"] = float(all_correct.mean())
    results["mean_confidence"] = float(all_confidences.mean())

    results["correlations"] = {}
    for metric_name, metric_values in all_metrics.items():
        metric_values = np.array(metric_values)
        r, p = stats.pearsonr(metric_values, all_confidences)
        results["correlations"][metric_name] = {
            "pearson_r": float(r),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "n_samples": len(metric_values),
        }

    results["correctness_correlations"] = {}
    for metric_name, metric_values in all_metrics.items():
        metric_values = np.array(metric_values)
        r, p = stats.pointbiserialr(all_correct.astype(float), metric_values)
        results["correctness_correlations"][metric_name] = {
            "point_biserial_r": float(r),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 1C: Phase Coherence Predictor")
    parser.add_argument("--modality", choices=["audio", "image", "both"], default="both")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp1c_phase_coherence_predictor.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 1C: Phase Coherence as Predictor of Classification Confidence")
    print("Claim C1 (stronger): Phase coherence PREDICTS semantic structure")
    print("=" * 70)

    all_results = {}

    if args.modality in ("audio", "both"):
        print(f"\n--- AUDIO ---")
        print(f"Generating synthetic audio ({args.n_classes} classes, "
              f"{args.n_samples_per_class} samples/class)...")
        waveforms, labels = generate_synthetic_audio(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
        )
        print(f"Generated {len(waveforms)} samples")
        audio_results = run_experiment_audio(waveforms, labels, n_folds=args.n_folds)
        all_results["audio"] = audio_results

    if args.modality in ("image", "both"):
        print(f"\n--- IMAGE ---")
        print(f"Generating synthetic images ({args.n_classes} classes, "
              f"{args.n_samples_per_class} samples/class)...")
        images, labels = generate_synthetic_images(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
        )
        print(f"Generated {len(images)} samples")
        image_results = run_experiment_image(images, labels, n_folds=args.n_folds)
        all_results["image"] = image_results

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for modality, mod_results in all_results.items():
        print(f"\n--- {modality.upper()} ---")
        print(f"Overall accuracy: {mod_results['overall_accuracy']:.4f}")
        print(f"Mean confidence:  {mod_results['mean_confidence']:.4f}")

        print(f"\n  Correlation with classification confidence:")
        print(f"  {'Metric':<25} {'Pearson r':<12} {'p-value':<12} {'Sig?'}")
        print(f"  {'-' * 55}")
        for metric, corr in mod_results["correlations"].items():
            sig = "***" if corr["p_value"] < 0.001 else "**" if corr["p_value"] < 0.01 else "*" if corr["p_value"] < 0.05 else "ns"
            print(f"  {metric:<25} {corr['pearson_r']:+.4f}      {corr['p_value']:.4f}      {sig}")

        print(f"\n  Correlation with correctness (point-biserial):")
        print(f"  {'Metric':<25} {'r':<12} {'p-value':<12} {'Sig?'}")
        print(f"  {'-' * 55}")
        for metric, corr in mod_results["correctness_correlations"].items():
            sig = "***" if corr["p_value"] < 0.001 else "**" if corr["p_value"] < 0.01 else "*" if corr["p_value"] < 0.05 else "ns"
            print(f"  {metric:<25} {corr['point_biserial_r']:+.4f}      {corr['p_value']:.4f}      {sig}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    n_sig_confidence = 0
    n_sig_correctness = 0
    total_metrics = 0
    for mod_results in all_results.values():
        for corr in mod_results["correlations"].values():
            total_metrics += 1
            if corr["significant"] and abs(corr["pearson_r"]) > 0.3:
                n_sig_confidence += 1
        for corr in mod_results["correctness_correlations"].values():
            if corr["significant"]:
                n_sig_correctness += 1

    print(f"Metrics with r > 0.3 and p < 0.05 (confidence): {n_sig_confidence}/{total_metrics}")
    print(f"Metrics with p < 0.05 (correctness): {n_sig_correctness}/{total_metrics}")
    if n_sig_confidence >= 2:
        print("SUPPORTS C1 (stronger): Phase coherence PREDICTS classification confidence.")
    elif n_sig_correctness >= 2:
        print("PARTIAL SUPPORT: Phase coherence correlates with correctness,")
        print("  but not strongly enough with confidence (r > 0.3).")
    else:
        print("INSUFFICIENT EVIDENCE for C1 predictor claim.")


if __name__ == "__main__":
    main()
