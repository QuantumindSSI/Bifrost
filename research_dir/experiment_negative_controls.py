"""
Experiment NEG: Negative Controls for Phase Coherence

Tests whether phase coherence effects are SPECIFIC to structured data
or appear universally (including data where they shouldn't matter).

A negative control tests the null hypothesis: if phase coherence "works"
on random noise data, then the method is not measuring anything meaningful.

Conditions:
1. Random noise audio (no structure) — phase ablation should NOT matter
2. Shuffled SpeechCommands (structure destroyed) — phase ablation should NOT matter
3. Constant signal (no information) — all methods should be at chance
4. Pure tones (amplitude-only structure) — phase coherence should NOT help

If phase ablation still "matters" on these controls, our positive results
on synthetic data were artifacts, not evidence for the thesis.

Usage:
    python3 research_dir/experiment_negative_controls.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from bifrost.cbmpc import CBMPCExtractor
from research_dir.data_loaders import load_speechcommands
from research_dir.experiment_utils import (
    save_results, print_results_table, print_stat_tests,
    extract_fft_magnitude_audio,
)
from research_dir.experiment_phase_ablation_audio_real import (
    extract_cbmpc_with_phase_ablation,
)


def generate_random_noise_audio(
    n_classes: int = 5,
    n_samples_per_class: int = 200,
    duration: float = 1.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random noise audio with random class labels.

    There is NO structure in this data. Phase coherence should not matter.
    """
    rng = np.random.RandomState(seed)
    target_len = int(sample_rate * duration)
    n_total = n_classes * n_samples_per_class
    waveforms = torch.from_numpy(rng.randn(n_total, target_len)).float()
    labels = torch.from_numpy(rng.randint(0, n_classes, n_total)).long()
    return waveforms, labels


def generate_pure_tones_audio(
    n_classes: int = 5,
    n_samples_per_class: int = 200,
    duration: float = 1.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> torch.Tensor:
    """Generate pure tones where classes differ ONLY in frequency (amplitude structure).

    Phase is random/uniform — there is no phase structure to exploit.
    Phase coherence should NOT help; amplitude features should work.
    """
    rng = np.random.RandomState(seed)
    target_len = int(sample_rate * duration)
    t = np.linspace(0, duration, target_len)

    class_freqs = [200, 400, 600, 800, 1000][:n_classes]

    waveforms = []
    labels = []
    for cls_idx, freq in enumerate(class_freqs):
        for _ in range(n_samples_per_class):
            # Random phase offset (no shared phase structure)
            phase_offset = rng.uniform(0, 2 * np.pi)
            # Random amplitude variation
            amp = rng.uniform(0.5, 1.0)
            wav = amp * np.sin(2 * np.pi * freq * t + phase_offset)
            # Add small noise
            wav += 0.01 * rng.randn(target_len)
            waveforms.append(wav)
            labels.append(cls_idx)

    waveforms = torch.tensor(np.array(waveforms)).float()
    labels = torch.tensor(labels).long()

    # Shuffle
    perm = rng.permutation(len(labels))
    return waveforms[perm], labels[perm]


def generate_shuffled_speechcommands(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """Shuffle the time dimension of each waveform independently.

    This destroys temporal structure while preserving frequency content.
    Phase relationships across time are destroyed.
    """
    rng = np.random.RandomState(seed)
    shuffled = waveforms.clone()
    for i in range(len(shuffled)):
        perm = rng.permutation(shuffled.shape[-1])
        shuffled[i] = shuffled[i, perm]
    return shuffled, labels


def generate_constant_signal(
    n_classes: int = 5,
    n_samples_per_class: int = 200,
    duration: float = 1.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> torch.Tensor:
    """Generate constant signals (no information at all).

    All methods should perform at chance level.
    """
    target_len = int(sample_rate * duration)
    n_total = n_classes * n_samples_per_class
    waveforms = torch.ones(n_total, target_len) * 0.5
    rng = np.random.RandomState(seed)
    labels = torch.from_numpy(rng.randint(0, n_classes, n_total)).long()
    return waveforms, labels


def run_control_experiment(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
    control_name: str = "",
) -> Dict:
    """Run phase ablation on control data."""

    extractor = CBMPCExtractor(
        sample_rate=16000, n_fft=512, hop_length=256,
        n_mels=32, feature_mode="compact",
    )

    ablations = ["baseline", "phase_randomize", "phase_zero"]
    baselines = ["fft_magnitude"]
    all_conditions = ablations + baselines

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(waveforms, labels)):
        print(f"  [{control_name}] Fold {fold + 1}/{n_folds}...")

        train_wav = waveforms[train_idx]
        test_wav = waveforms[test_idx]
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
        summary["statistical_tests"][f"cbmpc_vs_{baseline}"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_accuracy": float(delta),
            "significant": bool(p_value < 0.05),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Negative Controls for Phase Coherence")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--n_samples_per_class", type=int, default=200)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp_negative_controls.json")
    args = parser.parse_args()

    print("=" * 70)
    print("NEGATIVE CONTROLS: Testing specificity of phase coherence effects")
    print("=" * 70)

    all_results = {}

    # Control 1: Random noise
    print("\n--- Control 1: Random Noise (no structure) ---")
    print("Expected: All methods at chance. Phase ablation should NOT matter.")
    noise_wav, noise_labels = generate_random_noise_audio(
        args.n_classes, args.n_samples_per_class)
    all_results["random_noise"] = run_control_experiment(
        noise_wav, noise_labels, args.n_folds, "noise")

    # Control 2: Pure tones (amplitude-only structure)
    print("\n--- Control 2: Pure Tones (amplitude-only structure) ---")
    print("Expected: FFT magnitude should work. Phase coherence should NOT help.")
    tone_wav, tone_labels = generate_pure_tones_audio(
        args.n_classes, args.n_samples_per_class)
    all_results["pure_tones"] = run_control_experiment(
        tone_wav, tone_labels, args.n_folds, "tones")

    # Control 3: Constant signal
    print("\n--- Control 3: Constant Signal (no information) ---")
    print("Expected: All methods at chance.")
    const_wav, const_labels = generate_constant_signal(
        args.n_classes, args.n_samples_per_class)
    all_results["constant_signal"] = run_control_experiment(
        const_wav, const_labels, args.n_folds, "constant")

    # Control 4: Shuffled SpeechCommands
    print("\n--- Control 4: Shuffled SpeechCommands (temporal structure destroyed) ---")
    print("Expected: All methods should degrade. Phase ablation should NOT matter.")
    real_wav, real_labels, _ = load_speechcommands(
        n_classes=args.n_classes, n_samples_per_class=args.n_samples_per_class)
    shuffled_wav, shuffled_labels = generate_shuffled_speechcommands(
        real_wav, real_labels)
    all_results["shuffled_speechcommands"] = run_control_experiment(
        shuffled_wav, shuffled_labels, args.n_folds, "shuffled")

    # Print all results
    chance = 1.0 / args.n_classes
    print("\n" + "=" * 70)
    print("NEGATIVE CONTROL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nChance level: {chance:.4f}")
    print(f"\n{'Control':<30} {'CBMPC':<12} {'PhaseRand':<12} {'FFT Mag':<12} {'Phase Sig?'}")
    print("-" * 70)

    for control_name, results in all_results.items():
        cbmpc = results["baseline"]["mean_accuracy"]
        phase_rand = results["phase_randomize"]["mean_accuracy"]
        fft_mag = results["fft_magnitude"]["mean_accuracy"]
        phase_sig = results["statistical_tests"].get("phase_randomize", {}).get("significant", False)
        sig_str = "YES (BAD)" if phase_sig else "no (good)"
        print(f"{control_name:<30} {cbmpc:.4f}      {phase_rand:.4f}      {fft_mag:.4f}      {sig_str}")

    save_results(all_results, args.output)

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)

    # Check if phase ablation matters on noise (it shouldn't)
    noise_phase_sig = all_results["random_noise"]["statistical_tests"].get(
        "phase_randomize", {}).get("significant", False)
    # Check if phase ablation matters on constant (it shouldn't)
    const_phase_sig = all_results["constant_signal"]["statistical_tests"].get(
        "phase_randomize", {}).get("significant", False)

    if noise_phase_sig:
        print("WARNING: Phase ablation is significant on RANDOM NOISE.")
        print("  This means our method is detecting artifacts, not structure.")
    elif const_phase_sig:
        print("WARNING: Phase ablation is significant on CONSTANT SIGNAL.")
        print("  This means our method is detecting artifacts, not structure.")
    else:
        print("GOOD: Phase ablation is NOT significant on noise/constant controls.")
        print("  The method is specific to structured data (as expected).")

    # Check if CBMPC beats FFT on pure tones (it shouldn't — tones have no phase structure)
    tones_cbmpc = all_results["pure_tones"]["baseline"]["mean_accuracy"]
    tones_fft = all_results["pure_tones"]["fft_magnitude"]["mean_accuracy"]
    if tones_cbmpc > tones_fft + 0.05:
        print(f"\nWARNING: CBMPC beats FFT on pure tones ({tones_cbmpc:.4f} vs {tones_fft:.4f}).")
        print("  Pure tones have no phase structure — CBMPC should not help here.")
    else:
        print(f"\nGOOD: CBMPC does not beat FFT on pure tones ({tones_cbmpc:.4f} vs {tones_fft:.4f}).")
        print("  Phase coherence correctly does not help when there's no phase structure.")


if __name__ == "__main__":
    main()
