"""
Experiment 3C: Cross-Modal Coherence Alignment

Tests Claim C3 (cross-modal): The same coherence principle captures
semantic structure across audio and image modalities.

Method:
    1. Generate synthetic audio and image pairs with shared semantic categories
    2. Extract CBMPC features (audio) and PhaseCongruency features (image)
    3. Train UnifiedCoherenceMetric with contrastive loss
    4. Test cross-modal transfer: train classifier on audio, test on image

Success criterion: Cross-modal transfer accuracy > chance by at least 5 pp

Usage:
    python3 research_dir/experiment_cross_modal_coherence.py
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

from bifrost.cbmpc import CBMPCExtractor
from bifrost.msc_image import PhaseCongruencyExtractor
from bifrost.unified_coherence import (
    UnifiedCoherenceMetric,
    CrossModalCoherenceLoss,
    CoherenceClassifier,
)


def generate_cross_modal_pairs(
    n_classes: int = 5,
    n_samples_per_class: int = 100,
    sample_rate: int = 16000,
    duration: float = 1.0,
    image_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate paired audio + image data with shared semantic categories.

    Each class has a characteristic structure in BOTH audio and image, but
    the structures are generated INDEPENDENTLY — they share only the label,
    not the exact phase parameters. This prevents circular reasoning where
    the cross-modal correspondence is baked into the data generation.

    Audio: class-specific modulation patterns (temporal phase relationships)
    Image: class-specific spatial phase patterns (spatial frequency relationships)

    The correspondence between audio and image is only through the label c.
    The UCM must learn to map both to the same semantic space.

    Both modalities share the same amplitude spectrum across classes (controlled
    for amplitude), so phase is the only distinguishing feature within each
    modality.
    """
    T = int(sample_rate * duration)
    n_samples = n_classes * n_samples_per_class

    audio = torch.zeros(n_samples, T)
    images = torch.zeros(n_samples, 1, image_size, image_size)
    audio_labels = torch.zeros(n_samples, dtype=torch.long)
    image_labels = torch.zeros(n_samples, dtype=torch.long)

    # Shared audio frequencies for all classes (control for amplitude)
    audio_freqs = [200, 400, 600, 800, 1000]
    audio_amps = [1.0, 0.5, 0.33, 0.25, 0.2]

    # Shared image spatial frequencies for all classes (control for amplitude)
    img_freqs_x = [2, 4, 6, 8, 10]
    img_freqs_y = [3, 5, 7, 9, 11]
    img_amps = [1.0, 0.5, 0.33, 0.25, 0.2]

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, image_size),
        torch.linspace(0, 2 * np.pi, image_size),
    )

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            # === AUDIO ===
            # Class-specific phase relationships between harmonics
            # (INDEPENDENTLY generated from image — different formula)
            t = torch.arange(T) / sample_rate
            sig = torch.zeros(T)
            for i, (freq, amp) in enumerate(zip(audio_freqs, audio_amps)):
                audio_phase = c * 0.3 * (i + 1) + s * 0.005 * i
                sig += amp * torch.sin(2 * np.pi * freq * t + audio_phase)
            # Shared amplitude modulation (same for all classes)
            mod_freq = 4.0
            envelope = 1 + 0.2 * torch.sin(2 * np.pi * mod_freq * t)
            sig = sig * envelope
            sig += 0.05 * torch.randn(T)
            audio[idx] = sig
            audio_labels[idx] = c

            # === IMAGE ===
            # Class-specific spatial phase relationships
            # (INDEPENDENTLY generated from audio — DIFFERENT formula)
            img = torch.zeros(image_size, image_size)
            for i, (fx, fy, amp) in enumerate(zip(img_freqs_x, img_freqs_y, img_amps)):
                img_phase_x = c * 0.25 * (i + 1) + s * 0.004 * i
                img_phase_y = c * 0.18 * (i + 1) + s * 0.003 * i
                img += amp * torch.sin(fx * x_grid + img_phase_x) * \
                       torch.cos(fy * y_grid + img_phase_y)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img += 0.05 * torch.randn_like(img)
            img = torch.clamp(img, 0, 1)
            images[idx, 0] = img
            image_labels[idx] = c

            idx += 1

    # Shuffle (maintain pairing)
    perm = torch.randperm(n_samples)
    return audio[perm], images[perm], audio_labels[perm], image_labels[perm]


def run_experiment(
    audio: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    n_folds: int = 5,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> Dict:
    """Run cross-modal coherence alignment experiment."""

    audio_extractor = CBMPCExtractor(
        sample_rate=16000, n_fft=1024, hop_length=512,
        n_mels=32, feature_mode="compact",
    )
    image_extractor = PhaseCongruencyExtractor(
        n_scales=4, n_orientations=4, image_size=32,
    )

    # Extract features
    print("  Extracting audio features (CBMPC)...")
    audio_features = audio_extractor(audio)
    print(f"  Audio features: {audio_features.shape}")

    print("  Extracting image features (PhaseCongruency)...")
    image_features = image_extractor(images)
    print(f"  Image features: {image_features.shape}")

    audio_dim = audio_features.shape[1]
    image_dim = image_features.shape[1]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(audio_features, labels)):
        print(f"  Fold {fold + 1}/{n_folds}...")

        train_audio = audio_features[train_idx]
        test_audio = audio_features[test_idx]
        train_image = image_features[train_idx]
        test_image = image_features[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Train UnifiedCoherenceMetric
        ucm = UnifiedCoherenceMetric(
            audio_dim=audio_dim,
            image_dim=image_dim,
            sensor_dim=10,  # unused but required
            target_dim=64,
        )
        loss_fn = CrossModalCoherenceLoss(temperature=0.07)
        optimizer = torch.optim.Adam(ucm.parameters(), lr=lr)

        ucm.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            a_emb = ucm(train_audio, "audio")
            i_emb = ucm(train_image, "image")
            loss = loss_fn(a_emb, i_emb, train_labels)
            loss.backward()
            optimizer.step()

        # Evaluate cross-modal transfer
        ucm.eval()
        with torch.no_grad():
            # Train classifier on audio, test on image
            train_a_emb = ucm(train_audio, "audio")
            test_i_emb = ucm(test_image, "image")

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_a_emb.numpy(), train_labels.numpy())
            acc_audio_to_image = clf.score(test_i_emb.numpy(), test_labels.numpy())
            results["audio_to_image"]["accuracies"].append(acc_audio_to_image)

            # Train classifier on image, test on audio
            train_i_emb = ucm(train_image, "image")
            test_a_emb = ucm(test_audio, "audio")

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_i_emb.numpy(), train_labels.numpy())
            acc_image_to_audio = clf.score(test_a_emb.numpy(), test_labels.numpy())
            results["image_to_audio"]["accuracies"].append(acc_image_to_audio)

            # Within-modal baselines
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_audio.numpy(), train_labels.numpy())
            acc_audio = clf.score(test_audio.numpy(), test_labels.numpy())
            results["audio_within"]["accuracies"].append(acc_audio)

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_image.numpy(), train_labels.numpy())
            acc_image = clf.score(test_image.numpy(), test_labels.numpy())
            results["image_within"]["accuracies"].append(acc_image)

        # Amplitude-only baseline: train a SEPARATE UCM on amplitude-only
        # features, then test cross-modal transfer. This is a fair
        # comparison — same architecture, same training, same loss,
        # but amplitude features instead of coherence features.
        #
        # NOTE: This must be OUTSIDE the torch.no_grad() block above
        # because we need gradients for training.
        #
        # For audio: zero out PLV components (phase-based), keep amplitude
        n_mod = len(audio_extractor.modulation_freqs)
        audio_amp_only = audio_features.clone()
        audio_amp_only[:, :n_mod] = 0  # zero out PLV (phase-based)

        # For image: zero out PC histogram (phase-based), keep amplitude
        img_amp = image_features.clone()
        n_pc_bins = image_extractor.n_pc_bins
        img_amp[:, :n_pc_bins] = 0

        train_amp_a = audio_amp_only[train_idx]
        test_amp_a = audio_amp_only[test_idx]
        train_amp_i = img_amp[train_idx]
        test_amp_i = img_amp[test_idx]

        # Train a separate UCM on amplitude-only features
        amp_ucm = UnifiedCoherenceMetric(
            audio_dim=audio_dim,
            image_dim=image_dim,
            sensor_dim=10,
            target_dim=64,
        )
        amp_loss_fn = CrossModalCoherenceLoss(temperature=0.07)
        amp_optimizer = torch.optim.Adam(amp_ucm.parameters(), lr=lr)

        amp_ucm.train()
        for epoch in range(n_epochs):
            amp_optimizer.zero_grad()
            a_emb = amp_ucm(train_amp_a, "audio")
            i_emb = amp_ucm(train_amp_i, "image")
            loss = amp_loss_fn(a_emb, i_emb, train_labels)
            loss.backward()
            amp_optimizer.step()

        # Evaluate cross-modal transfer with amplitude-only UCM
        amp_ucm.eval()
        with torch.no_grad():
            train_a_emb_amp = amp_ucm(train_amp_a, "audio")
            test_i_emb_amp = amp_ucm(test_amp_i, "image")

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(train_a_emb_amp.numpy(), train_labels.numpy())
            acc_amp = clf.score(test_i_emb_amp.numpy(), test_labels.numpy())
            results["amplitude_only"]["accuracies"].append(acc_amp)

    # Compute statistics
    n_classes = len(torch.unique(labels))
    chance = 1.0 / n_classes

    summary = {}
    for condition in ["audio_within", "image_within", "audio_to_image",
                       "image_to_audio", "amplitude_only"]:
        accs = np.array(results[condition]["accuracies"])
        summary[condition] = {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "accuracies": accs.tolist(),
            "delta_vs_chance": float(accs.mean() - chance),
        }

    summary["chance_level"] = chance

    # Statistical tests: cross-modal vs chance
    summary["statistical_tests"] = {}
    for condition in ["audio_to_image", "image_to_audio"]:
        accs = np.array(results[condition]["accuracies"])
        t_stat, p_value = stats.ttest_1samp(accs, chance)
        summary["statistical_tests"][condition] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "delta_vs_chance": float(accs.mean() - chance),
            "significant": bool(p_value < 0.05),
        }

    # Cross-modal vs amplitude-only
    cm_accs = np.array(results["audio_to_image"]["accuracies"])
    amp_accs = np.array(results["amplitude_only"]["accuracies"])
    t_stat, p_value = stats.ttest_rel(cm_accs, amp_accs)
    summary["statistical_tests"]["cross_modal_vs_amplitude"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "delta": float(cm_accs.mean() - amp_accs.mean()),
        "significant": bool(p_value < 0.05),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3C: Cross-Modal Coherence")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--n_samples_per_class", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--output", type=str, default="research_dir/results/exp3c_cross_modal.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 3C: Cross-Modal Coherence Alignment")
    print("Claim C3: Coherence principle generalizes across modalities")
    print("=" * 70)

    print(f"\nGenerating cross-modal pairs ({args.n_classes} classes, "
          f"{args.n_samples_per_class} pairs/class)...")
    audio, images, a_labels, i_labels = generate_cross_modal_pairs(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    assert torch.all(a_labels == i_labels), "Labels must be paired"
    labels = a_labels
    print(f"Generated {len(audio)} audio-image pairs")

    print(f"\nRunning {args.n_folds}-fold cross-validation, {args.n_epochs} epochs/fold...")
    results = run_experiment(audio, images, labels,
                              n_folds=args.n_folds, n_epochs=args.n_epochs)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    chance = results["chance_level"]
    print(f"\nChance level: {chance:.4f}")
    print(f"\n{'Condition':<25} {'Accuracy':<20} {'Delta vs chance':<15}")
    print("-" * 60)
    for condition in ["audio_within", "image_within", "audio_to_image",
                       "image_to_audio", "amplitude_only"]:
        acc = results[condition]["mean_accuracy"]
        std = results[condition]["std_accuracy"]
        delta = results[condition]["delta_vs_chance"]
        print(f"{condition:<25} {acc:.4f} ± {std:.4f}     {delta:+.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)
    print(f"\n{'Test':<30} {'Delta':<10} {'t-stat':<10} {'p-value':<10} {'Sig?'}")
    print("-" * 70)
    for test_name, test in results["statistical_tests"].items():
        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
        delta_key = "delta_vs_chance" if "delta_vs_chance" in test else "delta"
        print(f"{test_name:<30} {test[delta_key]:+.4f}    {test['t_statistic']:+.3f}     {test['p_value']:.4f}    {sig}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    cm_acc = results["audio_to_image"]["mean_accuracy"]
    cm_delta = results["audio_to_image"]["delta_vs_chance"]
    cm_sig = results["statistical_tests"]["audio_to_image"]["significant"]
    amp_acc = results["amplitude_only"]["mean_accuracy"]
    amp_delta = results["amplitude_only"]["delta_vs_chance"]
    cm_vs_amp = results["statistical_tests"]["cross_modal_vs_amplitude"]
    print(f"Cross-modal transfer (audio→image, coherence): {cm_acc:.4f}")
    print(f"Cross-modal transfer (audio→image, amplitude):  {amp_acc:.4f}")
    print(f"Chance level: {chance:.4f}")
    print(f"Coherence delta vs chance: {cm_delta:+.4f} (target: > 0.05)")
    print(f"Amplitude delta vs chance: {amp_delta:+.4f}")
    print(f"Coherence vs amplitude:     {cm_vs_amp['delta']:+.4f} (p={cm_vs_amp['p_value']:.4f})")
    print()
    print("NOTE: UCM is trained with SUPERVISED contrastive loss using class")
    print("  labels. This is NOT zero-shot cross-modal transfer. The test is")
    print("  whether coherence features enable better cross-modal alignment")
    print("  than amplitude features under the same supervised training.")
    print()
    if cm_delta > 0.05 and cm_sig and cm_vs_amp["significant"]:
        print("SUPPORTS C3: Coherence features enable cross-modal transfer that")
        print("  significantly exceeds both chance AND amplitude-only baseline.")
    elif cm_delta > 0.05 and cm_sig:
        print("PARTIAL SUPPORT for C3: Cross-modal transfer exceeds chance,")
        print("  but does not significantly beat amplitude-only baseline.")
    else:
        print("INSUFFICIENT EVIDENCE for C3 cross-modal on this dataset.")


if __name__ == "__main__":
    main()
