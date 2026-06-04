#!/usr/bin/env python3
"""
Train S3 Phase-Lock Attractor Learning Module

Trains the AttractorLearningModule on spectral data to learn:
1. Attractor prototypes in frequency space
2. Phase pattern prototypes
3. Stability predictor convergence

Usage:
    python scripts/train_s3_attractor.py \
        --data-path train_data/text_corpus.txt \
        --d-model 768 \
        --n-bands 8 \
        --n-attractors 16 \
        --epochs 50 \
        --batch-size 32 \
        --save-path checkpoints/s3_attractor_trained.pt
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

from bifrost.s0_canonicalizer.canonicalizer import SpectralCanonicalizer
from bifrost.s3_attractor.attractor_learning import AttractorLearningModule
from bifrost.spectral_tensor import SpectralTensor


def parse_args():
    p = argparse.ArgumentParser(description="Train S3 Attractor Learning Module")
    p.add_argument("--data-path", type=str, required=True, help="Text corpus for training")
    p.add_argument("--llm-name", type=str, default="gpt2", help="LLM for text embeddings")
    p.add_argument("--d-model", type=int, default=768, help="Model dimension")
    p.add_argument("--n-bands", type=int, default=8, help="Number of phase bands")
    p.add_argument("--n-attractors", type=int, default=16, help="Number of attractors")
    p.add_argument("--stability-threshold", type=float, default=0.3, help="Stability threshold")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--max-length", type=int, default=256, help="Max text length")
    p.add_argument("--save-path", type=str, default="checkpoints/s3_attractor_trained.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_texts(path: str) -> List[str]:
    texts, current = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                current.append(line)
            else:
                if current:
                    texts.append("\n".join(current))
                    current = []
    return texts


class SpectralAttractorDataset(Dataset):
    """
    Dataset that converts text to spectral tensors for attractor learning.

    Each sample produces:
    - spectral: SpectralTensor from text embedding FFT
    - target_stability: derived from embedding coherence
    """

    def __init__(self, texts: List[str], tokenizer, embed_model,
                 canonicalizer: SpectralCanonicalizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.embed_model = embed_model
        self.canonicalizer = canonicalizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             return_tensors="pt", padding="max_length")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            emb = self.embed_model(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state

        emb = emb.squeeze(0)  # (seq_len, d_model)
        emb = emb - emb.mean(dim=0, keepdim=True)
        emb = emb / (emb.std(dim=0, keepdim=True) + 1e-8)

        spectral = self.canonicalizer(emb)
        return spectral


def collate_spectral(batch: List[SpectralTensor]):
    """Collate spectral tensors into batched tensors."""
    amps = torch.stack([s.amplitude for s in batch])
    phases = torch.stack([s.phase for s in batch])
    scales = torch.stack([s.scale for s in batch])
    unc = torch.stack([s.uncertainty for s in batch])
    return SpectralTensor(amplitude=amps, phase=phases, scale=scales, uncertainty=unc)


def train_epoch(model: AttractorLearningModule, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_stability_loss = 0.0
    total_coherence_loss = 0.0
    total_contrast_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        attractors, stats = model(batch)

        # Loss 1: Stability should be high for coherent spectral patterns
        # Compute phase coherence per sample
        phase_complex = torch.exp(1j * batch.phase)
        coherence = torch.abs(phase_complex.mean(dim=-1))  # (B,)

        # Stability should correlate with coherence
        stabilities = torch.stack([a.stability for a in attractors])
        stability_loss = F.mse_loss(stabilities, coherence)

        # Loss 2: Attractor prototypes should be distinct (contrastive)
        protos = model.attractor_prototypes
        proto_sim = F.cosine_similarity(protos.unsqueeze(1), protos.unsqueeze(0), dim=-1)
        eye = torch.eye(len(protos), device=device)
        contrast_loss = (proto_sim * eye).mean() - ((1 - eye) * F.relu(proto_sim - 0.5)).mean()
        contrast_loss = -contrast_loss  # Maximize dissimilarity off-diagonal

        # Loss 3: Phase patterns should be smooth (low variance across bands)
        phase_smooth = model.phase_prototypes.var(dim=-1).mean()

        loss = stability_loss + 0.5 * contrast_loss + 0.1 * phase_smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_stability_loss += stability_loss.item()
        total_contrast_loss += contrast_loss.item()
        total_coherence_loss += phase_smooth.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "stability_loss": total_stability_loss / n_batches,
        "contrast_loss": total_contrast_loss / n_batches,
        "phase_smooth_loss": total_coherence_loss / n_batches,
    }


def validate(model: AttractorLearningModule, loader: DataLoader,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_stability = 0.0
    total_coherence = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            attractors, stats = model(batch)

            phase_complex = torch.exp(1j * batch.phase)
            coherence = torch.abs(phase_complex.mean(dim=-1))
            stabilities = torch.stack([a.stability for a in attractors])

            total_loss += F.mse_loss(stabilities, coherence).item()
            total_stability += stabilities.mean().item()
            total_coherence += coherence.mean().item()
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mean_stability": total_stability / n_batches,
        "mean_coherence": total_coherence / n_batches,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load text data
    print(f"Loading data from {args.data_path}")
    texts = load_texts(args.data_path)
    print(f"Loaded {len(texts)} text samples")

    # Split train/val
    split = int(0.9 * len(texts))
    train_texts, val_texts = texts[:split], texts[split:]
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    # Load tokenizer and embedding model
    print(f"Loading {args.llm_name} for embeddings")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    embed_model = AutoModel.from_pretrained(args.llm_name)
    embed_model.eval()
    for p in embed_model.parameters():
        p.requires_grad = False

    # Canonicalizer
    canonicalizer = SpectralCanonicalizer(n_fft=1024, normalize_input=True)

    # Datasets
    train_ds = SpectralAttractorDataset(train_texts, tokenizer, embed_model,
                                        canonicalizer, args.max_length)
    val_ds = SpectralAttractorDataset(val_texts, tokenizer, embed_model,
                                      canonicalizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_spectral)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_spectral)

    # Model
    print(f"Creating AttractorLearningModule: d_model={args.d_model}, "
          f"n_bands={args.n_bands}, n_attractors={args.n_attractors}")
    model = AttractorLearningModule(
        d_model=args.d_model,
        n_bands=args.n_bands,
        n_attractors=args.n_attractors,
        stability_threshold=args.stability_threshold,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs")
    history = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Stability: {val_metrics['mean_stability']:.4f} | "
              f"Val Coherence: {val_metrics['mean_coherence']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
            }, args.save_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    # Final summary
    print("\n=== TRAINING COMPLETE ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")

    history_path = args.save_path.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
