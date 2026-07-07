"""
Experiment 3D: Coherence Space Visualization

Tests Claim C3 (visualization): In the unified coherence space, samples
cluster by semantic category, not by modality.

Method:
    1. Generate synthetic audio, image, and sensor data with shared
       semantic categories (same labels across modalities)
    2. Extract modality-specific coherence features
       (CBMPC, PhaseCongruency, WaveletCoherence)
    3. Project all features to unified coherence space via UCM
    4. Visualize with t-SNE, color by category and modality
    5. Quantify: silhouette score by category vs by modality

Success criterion: Category silhouette > modality silhouette.

Usage:
    python3 research_dir/experiment_coherence_visualization.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from bifrost.cbmpc import CBMPCExtractor
from bifrost.msc_image import PhaseCongruencyExtractor
from bifrost.msc_sensor import WaveletCoherenceExtractor
from bifrost.unified_coherence import UnifiedCoherenceMetric, CrossModalCoherenceLoss


def generate_shared_category_data(
    n_classes: int = 5,
    n_samples_per_class: int = 50,
    sample_rate: int = 16000,
    duration: float = 1.0,
    image_size: int = 32,
    n_channels: int = 6,
    sensor_rate: float = 50.0,
    sensor_duration: float = 4.0,
) -> Dict:
    """Generate audio, image, and sensor data with shared semantic categories.

    All modalities share the same class labels. Within each modality,
    classes differ only in phase structure (amplitude controlled).
    Modalities are generated independently — no shared phase parameters.
    """
    # Audio
    T_audio = int(sample_rate * duration)
    audio_freqs = [200, 400, 600, 800, 1000]
    audio_amps = [1.0, 0.5, 0.33, 0.25, 0.2]

    audio = torch.zeros(n_classes * n_samples_per_class, T_audio)
    audio_labels = torch.zeros(n_classes * n_samples_per_class, dtype=torch.long)

    # Image
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, image_size),
        torch.linspace(0, 2 * np.pi, image_size),
    )
    img_freqs_x = [2, 4, 6, 8, 10]
    img_freqs_y = [3, 5, 7, 9, 11]
    img_amps = [1.0, 0.5, 0.33, 0.25, 0.2]

    images = torch.zeros(n_classes * n_samples_per_class, 1, image_size, image_size)
    image_labels = torch.zeros(n_classes * n_samples_per_class, dtype=torch.long)

    # Sensor
    T_sensor = int(sensor_rate * sensor_duration)
    sensor_freqs = [1.0, 3.0, 7.0]
    sensor_amps = [1.0, 0.5, 0.3]

    sensors = torch.zeros(n_classes * n_samples_per_class, n_channels, T_sensor)
    sensor_labels = torch.zeros(n_classes * n_samples_per_class, dtype=torch.long)

    idx = 0
    for c in range(n_classes):
        for s in range(n_samples_per_class):
            # Audio
            t = torch.arange(T_audio) / sample_rate
            sig = torch.zeros(T_audio)
            for i, (freq, amp) in enumerate(zip(audio_freqs, audio_amps)):
                phase = c * 0.3 * (i + 1) + s * 0.005 * i
                sig += amp * torch.sin(2 * np.pi * freq * t + phase)
            sig *= 1 + 0.2 * torch.sin(2 * np.pi * 4.0 * t)
            sig += 0.05 * torch.randn(T_audio)
            audio[idx] = sig
            audio_labels[idx] = c

            # Image (independent phase formula)
            img = torch.zeros(image_size, image_size)
            for i, (fx, fy, amp) in enumerate(zip(img_freqs_x, img_freqs_y, img_amps)):
                px = c * 0.25 * (i + 1) + s * 0.004 * i
                py = c * 0.18 * (i + 1) + s * 0.003 * i
                img += amp * torch.sin(fx * x_grid + px) * torch.cos(fy * y_grid + py)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img += 0.05 * torch.randn_like(img)
            img = torch.clamp(img, 0, 1)
            images[idx, 0] = img
            image_labels[idx] = c

            # Sensor (independent phase formula)
            t_s = torch.arange(T_sensor) / sensor_rate
            sig_s = torch.zeros(n_channels, T_sensor)
            for ch in range(n_channels):
                for f, a in zip(sensor_freqs, sensor_amps):
                    phase = c * 0.4 * (ch + 1) / f + s * 0.003
                    sig_s[ch] += a * torch.sin(2 * np.pi * f * t_s + phase)
            sig_s += 0.1 * torch.randn(n_channels, T_sensor)
            sensors[idx] = sig_s
            sensor_labels[idx] = c

            idx += 1

    # Shuffle each independently
    perm_a = torch.randperm(len(audio))
    perm_i = torch.randperm(len(images))
    perm_s = torch.randperm(len(sensors))

    return {
        "audio": (audio[perm_a], audio_labels[perm_a]),
        "image": (images[perm_i], image_labels[perm_i]),
        "sensor": (sensors[perm_s], sensor_labels[perm_s]),
    }


def run_experiment(data: Dict, n_epochs: int = 50, lr: float = 1e-3) -> Dict:
    """Run coherence space visualization experiment."""

    # Extractors
    audio_extractor = CBMPCExtractor(
        sample_rate=16000, n_fft=1024, hop_length=512,
        n_mels=32, feature_mode="compact",
    )
    image_extractor = PhaseCongruencyExtractor(
        n_scales=4, n_orientations=4, image_size=32,
    )
    sensor_extractor = WaveletCoherenceExtractor(
        n_scales=8, n_channels=6, sample_rate=50.0,
    )

    # Extract features
    print("  Extracting audio features (CBMPC)...")
    audio_feat = audio_extractor(data["audio"][0])
    print(f"  Audio features: {audio_feat.shape}")

    print("  Extracting image features (PhaseCongruency)...")
    image_feat = image_extractor(data["image"][0])
    print(f"  Image features: {image_feat.shape}")

    print("  Extracting sensor features (WaveletCoherence)...")
    sensor_feat = sensor_extractor(data["sensor"][0])
    print(f"  Sensor features: {sensor_feat.shape}")

    audio_dim = audio_feat.shape[1]
    image_dim = image_feat.shape[1]
    sensor_dim = sensor_feat.shape[1]

    # Train UCM with all three modalities
    ucm = UnifiedCoherenceMetric(
        audio_dim=audio_dim,
        image_dim=image_dim,
        sensor_dim=sensor_dim,
        target_dim=64,
    )
    loss_fn = CrossModalCoherenceLoss(temperature=0.07)
    optimizer = torch.optim.Adam(ucm.parameters(), lr=lr)

    # Use shared labels for training (all modalities have same classes)
    # For training, align audio-image pairs (they share the same label space)
    labels = data["audio"][1]

    print(f"  Training UCM ({n_epochs} epochs)...")
    ucm.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        a_emb = ucm(audio_feat, "audio")
        i_emb = ucm(image_feat, "image")
        loss = loss_fn(a_emb, i_emb, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}: loss = {loss.item():.4f}")

    # Project all modalities to unified space
    ucm.eval()
    with torch.no_grad():
        audio_emb = ucm(audio_feat, "audio").numpy()
        image_emb = ucm(image_feat, "image").numpy()
        sensor_emb = ucm(sensor_feat, "sensor").numpy()

    # Combine all embeddings
    all_embeddings = np.concatenate([audio_emb, image_emb, sensor_emb], axis=0)
    all_labels = np.concatenate([
        data["audio"][1].numpy(),
        data["image"][1].numpy(),
        data["sensor"][1].numpy(),
    ])
    all_modalities = np.concatenate([
        np.zeros(len(audio_emb)),  # 0 = audio
        np.ones(len(image_emb)),   # 1 = image
        np.full(len(sensor_emb), 2),  # 2 = sensor
    ])

    # t-SNE
    print("  Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Silhouette scores
    # Sample for efficiency
    n_total = len(all_labels)
    if n_total > 1500:
        idx = np.random.RandomState(42).choice(n_total, 1500, replace=False)
        emb_sub = all_embeddings[idx]
        labels_sub = all_labels[idx]
        mod_sub = all_modalities[idx]
    else:
        emb_sub = all_embeddings
        labels_sub = all_labels
        mod_sub = all_modalities

    sil_category = silhouette_score(emb_sub, labels_sub,
                                     sample_size=min(500, len(labels_sub)))
    sil_modality = silhouette_score(emb_sub, mod_sub,
                                     sample_size=min(500, len(mod_sub)))

    return {
        "n_samples_total": n_total,
        "n_classes": int(len(np.unique(all_labels))),
        "n_modalities": 3,
        "silhouette_by_category": float(sil_category),
        "silhouette_by_modality": float(sil_modality),
        "silhouette_ratio": float(sil_category / (sil_modality + 1e-8)),
        "tsne_embeddings": embeddings_2d.tolist(),
        "labels": all_labels.tolist(),
        "modalities": all_modalities.tolist(),
        "audio_dim": audio_dim,
        "image_dim": image_dim,
        "sensor_dim": sensor_dim,
        "final_loss": float(loss.item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3D: Coherence Space Visualization")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--n_samples_per_class", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp3d_coherence_visualization.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment 3D: Coherence Space Visualization")
    print("Claim C3 (visualization): Samples cluster by category, not modality")
    print("=" * 70)

    print(f"\nGenerating shared-category data ({args.n_classes} classes, "
          f"{args.n_samples_per_class} samples/class/modality)...")
    data = generate_shared_category_data(
        n_classes=args.n_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    print(f"Generated {args.n_classes * args.n_samples_per_class} samples per modality "
          f"({args.n_classes * args.n_samples_per_class * 3} total)")

    print(f"\nRunning experiment ({args.n_epochs} epochs)...")
    results = run_experiment(data, n_epochs=args.n_epochs)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nTotal samples: {results['n_samples_total']}")
    print(f"Classes: {results['n_classes']}")
    print(f"Modalities: {results['n_modalities']}")
    print(f"\nSilhouette by category:  {results['silhouette_by_category']:.4f}")
    print(f"Silhouette by modality:  {results['silhouette_by_modality']:.4f}")
    print(f"Ratio (cat/mod):         {results['silhouette_ratio']:.4f}")
    print(f"Final training loss:     {results['final_loss']:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if results["silhouette_by_category"] > results["silhouette_by_modality"]:
        print("SUPPORTS C3: Samples cluster by semantic category more than by modality.")
        print(f"  Category silhouette ({results['silhouette_by_category']:.4f}) > "
              f"Modality silhouette ({results['silhouette_by_modality']:.4f})")
    else:
        print("INSUFFICIENT EVIDENCE: Samples cluster by modality more than by category.")
        print(f"  Category silhouette ({results['silhouette_by_category']:.4f}) < "
              f"Modality silhouette ({results['silhouette_by_modality']:.4f})")
    print()
    print("NOTE: UCM is trained with supervised contrastive loss (audio-image pairs).")
    print("  Sensor modality is projected without direct alignment training.")
    print("  This is NOT zero-shot — it tests whether coherence features from")
    print("  different modalities can be aligned in a shared space.")


if __name__ == "__main__":
    main()
