"""
Experiment 1A: Phase Ablation on Audio (SpeechCommands)

Tests Claim C1: Phase coherence captures semantic structure.

Method:
    1. Extract CBMPC features from SpeechCommands audio
    2. Apply phase ablations (zero, randomize, noise, quantize, cross-band scramble)
    3. Train linear classifier on each condition
    4. Compare accuracy across conditions

Success criterion: Full phase > ablated conditions (p < 0.05)

Since SpeechCommands download may not be available, this script also
supports a synthetic audio dataset that generates structured signals
(different frequencies/amplitudes per class) to validate the pipeline.

Usage:
    python3 research_dir/experiment_phase_ablation_audio.py --synthetic
    python3 research_dir/experiment_phase_ablation_audio.py --data_dir /path/to/speech_commands
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.cbmpc import CBMPCExtractor, CBMPCClassifier
from bifrost.spectral_tensor import SpectralTensor
from bifrost.validation.phase_ablation import PhaseAblationHarness
from bifrost.validation.phase_metrics import PhaseCoherenceSignalMetrics


def generate_synthetic_audio(n_classes: int = 10, n_samples_per_class: int = 200,
                              sample_rate: int = 16000, duration: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic audio where classes differ ONLY in phase structure.

    All classes share the same amplitude spectrum (same frequencies, same
    amplitudes). Classes differ only in the phase relationships between
    harmonics. This ensures that phase is the only distinguishing feature.

    Without phase information, all classes are identical and classification
    should drop to chance.
    """
    n_samples = n_classes * n_samples_per_class
    T = int(sample_rate * duration)
    signals = torch.zeros(n_samples, T)
    labels = torch.zeros(n_samples, dtype=torch.long)

    # Shared amplitude spectrum for all classes
    base_freqs = [200, 400, 600, 800, 1000]  # same for all classes
    base_amps = [1.0, 0.5, 0.33, 0.25, 0.2]  # same for all classes

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            t = torch.arange(T) / sample_rate
            signal = torch.zeros(T)

            for i, (freq, amp) in enumerate(zip(base_freqs, base_amps)):
                # Class-specific phase per harmonic — this is the ONLY
                # feature that distinguishes classes
                phase_offset = c * 0.3 * (i + 1) + s * 0.005 * i
                signal += amp * torch.sin(2 * np.pi * freq * t + phase_offset)

            # Shared amplitude modulation (same for all classes)
            mod_freq = 4.0
            envelope = 1 + 0.2 * torch.sin(2 * np.pi * mod_freq * t)
            signal = signal * envelope

            # Add noise
            signal += 0.02 * torch.randn(T)

            signals[idx] = signal
            labels[idx] = c
            idx += 1

    # Shuffle
    perm = torch.randperm(n_samples)
    return signals[perm], labels[perm]


def extract_cbmpc_with_ablation(
    waveforms: torch.Tensor,
    extractor: CBMPCExtractor,
    harness: PhaseAblationHarness,
    ablation: str = "baseline",
) -> torch.Tensor:
    """Extract CBMPC features with optional phase ablation.

    Ablation is applied to the MODULATION SPECTRUM phase — the phase that
    CBMPC uses for PLV computation. This is the correct level of ablation
    because CBMPC's key feature (PLV) is computed from modulation phase.

    For signal-level ablation (phase_zero, phase_randomize), we also
    reconstruct the signal from the ablated STFT to affect the magnitude
    spectrogram through temporal interference patterns.
    """
    if ablation == "baseline":
        return extractor(waveforms)

    # Step 1: Compute STFT
    stft = torch.stft(
        waveforms,
        n_fft=extractor.n_fft,
        hop_length=extractor.hop_length,
        return_complex=True,
    )  # (B, n_freq, T_frames)

    # Step 2: Mel projection on magnitude
    mel_mag = torch.matmul(extractor.mel_fb, stft.abs())  # (B, n_mels, T)

    # Step 3: Log compression
    log_mag = torch.log(mel_mag + 1e-8)

    # Step 4: Temporal FFT (modulation spectrum)
    mod_spectrum = torch.fft.rfft(log_mag, dim=-1)
    mod_amp = mod_spectrum.abs()
    mod_phase = mod_spectrum.angle()

    # Step 5: Apply ablation to modulation spectrum phase
    # Create SpectralTensor from modulation spectrum
    # Shape: (B, n_mels, n_mod_bins)
    scale = torch.ones_like(mod_amp)
    uncertainty = torch.zeros_like(mod_amp)
    mod_st = SpectralTensor(mod_amp, mod_phase, scale, uncertainty)

    if ablation == "phase_zero":
        mod_st = harness.phase_zero(mod_st)
    elif ablation == "phase_randomize":
        mod_st = harness.phase_randomize(mod_st)
    elif ablation == "phase_noise":
        mod_st = harness.phase_noise(mod_st, sigma=0.5)
    elif ablation == "phase_noise_severe":
        mod_st = harness.phase_noise(mod_st, sigma=2.0)
    elif ablation == "phase_quantize":
        mod_st = harness.phase_quantize(mod_st, n_levels=4)
    elif ablation == "cross_band_scramble":
        # Add independent random phase offset to each mel band — this
        # directly destroys the cross-band phase coherence that PLV measures.
        # PLV = |mean_f exp(i*phase_f)| requires phases to be aligned across
        # bands. Adding independent offsets makes them incoherent.
        g = torch.Generator(device=mod_st.phase.device).manual_seed(42)
        n_mels = mod_st.phase.shape[1]
        # Random offset per mel band: (1, n_mels, 1)
        offsets = (torch.rand(1, n_mels, 1, generator=g,
                              device=mod_st.phase.device,
                              dtype=mod_st.phase.dtype) * 2 * torch.pi - torch.pi)
        scrambled_phase = mod_st.phase + offsets
        scrambled_phase = torch.atan2(
            torch.sin(scrambled_phase), torch.cos(scrambled_phase)
        )
        mod_st = SpectralTensor(
            amplitude=mod_st.amplitude,
            phase=scrambled_phase,
            scale=mod_st.scale,
            uncertainty=mod_st.uncertainty,
            metadata={**mod_st.metadata, "ablation": "cross_band_scramble"},
        )
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    # Step 6: Reconstruct modulation spectrum from ablated phase
    ablated_mod = mod_st.amplitude * torch.exp(1j * mod_st.phase)

    # Step 7: Extract features at target modulation frequencies
    n_frames = log_mag.shape[-1]
    frame_rate = extractor.sample_rate / extractor.hop_length
    mod_freqs_all = torch.fft.rfftfreq(n_frames, d=1.0 / frame_rate)

    target_bins = []
    for target_f in extractor.modulation_freqs:
        if len(mod_freqs_all) == 0:
            target_bins.append(0)
            continue
        bin_idx = torch.argmin(torch.abs(mod_freqs_all - target_f)).item()
        target_bins.append(bin_idx)

    plv_values = []
    mean_amp_values = []
    per_band_amp_values = []

    for bin_idx in target_bins:
        phases = ablated_mod[:, :, bin_idx].angle()  # use ablated phase
        plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=1)).real
        plv_values.append(plv)
        amp = ablated_mod[:, :, bin_idx].abs().mean(dim=1)
        mean_amp_values.append(amp)
        per_band_amp_values.append(ablated_mod[:, :, bin_idx].abs())

    plv_tensor = torch.stack(plv_values, dim=1)
    amp_tensor = torch.stack(mean_amp_values, dim=1)

    if extractor.feature_mode == "compact":
        features = torch.cat([plv_tensor, amp_tensor], dim=1)
    else:
        per_band = torch.stack(per_band_amp_values, dim=2)
        per_band_flat = per_band.reshape(per_band.shape[0], -1)
        features = torch.cat([per_band_flat, plv_tensor, amp_tensor], dim=1)

    return features.float()


def run_experiment(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
) -> Dict:
    """Run phase ablation experiment with k-fold cross-validation."""

    extractor = CBMPCExtractor(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        feature_mode="compact",
    )
    harness = PhaseAblationHarness()

    ablations = [
        "baseline",
        "phase_zero",
        "phase_randomize",
        "phase_noise",
        "phase_noise_severe",
        "phase_quantize",
        "cross_band_scramble",
    ]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = defaultdict(lambda: defaultdict(list))
    metrics_calculator = PhaseCoherenceSignalMetrics()

    for fold, (train_idx, test_idx) in enumerate(skf.split(waveforms, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_wav = waveforms[train_idx]
        test_wav = waveforms[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        for ablation in ablations:
            # Extract features
            train_feat = extract_cbmpc_with_ablation(
                train_wav, extractor, harness, ablation
            ).numpy()
            test_feat = extract_cbmpc_with_ablation(
                test_wav, extractor, harness, ablation
            ).numpy()

            # Train linear classifier
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_feat, train_labels.numpy())
            acc = clf.score(test_feat, test_labels.numpy())
            results[ablation]["accuracies"].append(acc)

            # Compute phase coherence metrics on test set
            if ablation == "baseline":
                stft = torch.stft(test_wav, n_fft=1024, hop_length=512,
                                  return_complex=True)
                phase = stft.angle()
                plv = metrics_calculator.phase_locking_value(
                    phase[:, :-1, :], phase[:, 1:, :], dim=-1
                ).mean().item()
                results[ablation]["mean_plv"].append(plv)

    # Compute statistics
    summary = {}
    for ablation in ablations:
        accs = np.array(results[ablation]["accuracies"])
        summary[ablation] = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "accuracies": accs.tolist(),
        }

    # Paired t-tests: baseline vs each ablation
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
    parser = argparse.ArgumentParser(description="Experiment 1A: Phase Ablation on Audio")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic audio")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SpeechCommands")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str, default="research_dir/results/exp1a_phase_ablation_audio.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 1A: Phase Ablation on Audio")
    print("Claim C1: Phase coherence captures semantic structure")
    print("=" * 70)

    if args.synthetic or args.data_dir is None:
        print(f"\nGenerating synthetic audio ({args.n_classes} classes, "
              f"{args.n_samples_per_class} samples/class)...")
        waveforms, labels = generate_synthetic_audio(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
        )
        print(f"Generated {len(waveforms)} samples")
    else:
        print(f"\nLoading SpeechCommands from {args.data_dir}...")
        # TODO: implement real data loading
        print("Real data loading not implemented. Using synthetic.")
        waveforms, labels = generate_synthetic_audio(
            n_classes=args.n_classes,
            n_samples_per_class=args.n_samples_per_class,
        )

    print(f"\nRunning {args.n_folds}-fold cross-validation with 7 ablation conditions...")
    results = run_experiment(waveforms, labels, n_folds=args.n_folds)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Accuracy (mean ± std)':<25} {'Delta vs baseline':<15}")
    print("-" * 65)

    baseline_acc = results["baseline"]["mean_accuracy"]
    for ablation in ["baseline", "phase_zero", "phase_randomize", "phase_noise",
                      "phase_noise_severe", "phase_quantize", "cross_band_scramble"]:
        acc = results[ablation]["mean_accuracy"]
        std = results[ablation]["std_accuracy"]
        delta = acc - baseline_acc if ablation != "baseline" else 0.0
        print(f"{ablation:<25} {acc:.4f} ± {std:.4f}          {delta:+.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (paired t-test, baseline vs ablated)")
    print("=" * 70)
    print(f"\n{'Ablation':<25} {'Delta':<10} {'t-stat':<10} {'p-value':<10} {'Significant?'}")
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
        print("SUPPORTS C1: Phase coherence captures semantic structure in audio.")
    else:
        print("INSUFFICIENT EVIDENCE for C1 on this dataset.")


if __name__ == "__main__":
    main()
