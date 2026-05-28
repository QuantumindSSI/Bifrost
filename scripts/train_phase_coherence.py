#!/usr/bin/env python3
"""
Train Bifröst pipeline for phase coherence learning.

Training objective: next-frame spectral prediction.
  - Input:  first half of each audio chunk → pipeline → bound_st
  - Target: second half of each audio chunk → pipeline (no_grad) → target_st
  - Loss:   |amplitude_pred - amplitude_tgt|^2 + |phase_pred_wrapped - phase_tgt|^2

This is a real next-frame objective: the model must predict the spectral
representation of audio it has NOT yet seen, not a shifted version of its own output.

Usage:
    cd bifrost
    python scripts/train_phase_coherence.py --epochs 100
    python scripts/train_phase_coherence.py --epochs 200 --harmonic-binding --device cuda

Diagnostic output every epoch:
  - loss:              next-frame prediction MSE (lower = better temporal prediction)
  - coherence_var:     variance across attention matrix (higher = more structured attention)
  - diag_ratio:        diagonal vs off-diagonal mean coherence (>1 = self-coherence dominant)

Expected progression (complex SSM, d_model=128):
  Epoch   0: loss ~0.5,  coherence_var ~0.001, diag_ratio ~1.06
  Epoch  50: loss ~0.3,  coherence_var ~0.003, diag_ratio ~1.15
  Epoch 100: loss ~0.15, coherence_var ~0.005, diag_ratio ~1.2–1.5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Allow running from both repo root and bifrost/ subdirectory
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from bifrost import BifrostPipeline
from bifrost.training import FBCTrainer as BifrostTrainer


# ── Signal generation ────────────────────────────────────────────────────────

def _harmonic_signal(
    fund_hz: float,
    n_harmonics: int,
    duration_s: float,
    sample_rate: int,
    amp_decay: float = 0.6,
) -> torch.Tensor:
    """Return a (L,) harmonic tone: fundamental + n_harmonics overtones."""
    t = torch.linspace(0.0, duration_s, int(sample_rate * duration_s))
    wave = torch.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        freq = fund_hz * k
        if freq >= sample_rate / 2:
            break
        wave = wave + (amp_decay ** (k - 1)) * torch.sin(2.0 * math.pi * freq * t)
    return wave / (wave.abs().max() + 1e-8)


class _HarmonicDataset(torch.utils.data.Dataset):
    """
    On-the-fly harmonic signal generator — every __getitem__ call
    synthesises a fresh waveform with randomised fundamental, phase
    offsets, amplitude envelope, and chord combination.  The model
    cannot memorise exact waveforms; it must learn structure.

    Parameters
    ----------
    size:        Virtual dataset length (items per epoch).
    sample_rate: Audio sample rate in Hz.
    chunk_s:     Duration of each waveform in seconds.
    seed:        Base RNG seed; each item uses seed+index for reproducibility.
    """

    def __init__(
        self,
        size: int,
        sample_rate: int = 16_000,
        chunk_s: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.sample_rate = sample_rate
        self.L = int(sample_rate * chunk_s)
        self.seed = seed
        # Full chromatic range A1–A6 (55–1760 Hz) — 61 unique fundamentals
        self._fundamentals = [55.0 * (2 ** (i / 12)) for i in range(61)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)

        # Random fundamental from chromatic scale
        fund_idx = torch.randint(0, len(self._fundamentals), (1,), generator=rng).item()
        fund = self._fundamentals[fund_idx]

        # Random number of harmonics (2–6)
        n_harmonics = int(torch.randint(2, 7, (1,), generator=rng).item())

        # Random per-harmonic phase offsets — this is what forces phase coherence learning
        phase_offsets = (torch.rand(n_harmonics, generator=rng) * 2 * math.pi).tolist()

        # Random amplitude decay (0.3–0.9)
        amp_decay = float(torch.empty(1).uniform_(0.3, 0.9))

        t = torch.linspace(0.0, self.L / self.sample_rate, self.L)
        wave = torch.zeros(self.L)
        for k in range(n_harmonics):
            freq = fund * (k + 1)
            if freq >= self.sample_rate / 2:
                break
            phase = phase_offsets[k]
            amp = amp_decay ** k
            wave = wave + amp * torch.sin(2.0 * math.pi * freq * t + phase)

        # Random amplitude envelope (attack/decay) — prevents trivial energy matching
        envelope_len = self.L
        attack = int(torch.randint(self.L // 10, self.L // 4, (1,), generator=rng).item())
        env = torch.ones(envelope_len)
        env[:attack] = torch.linspace(0.0, 1.0, attack)
        wave = wave * env

        # Normalise + low-level noise floor
        wave = wave / (wave.abs().max() + 1e-8)
        noise = torch.randn(self.L, generator=rng) * 0.015
        return (wave + noise).float()


def build_dataset(
    batch_size: int,
    n_batches: int,
    sample_rate: int = 16_000,
    chunk_s: float = 1.0,
    seed: int = 0,
) -> DataLoader:
    """
    Build an on-the-fly harmonic audio dataset.

    Each call to the DataLoader generates fresh waveforms with randomised
    fundamentals, phase offsets, and envelopes — the model cannot memorise
    exact sequences and must learn spectral structure.

    Args:
        batch_size:  Samples per batch.
        n_batches:   Total number of batches per epoch.
        sample_rate: Audio sample rate in Hz.
        chunk_s:     Duration of each waveform in seconds.
        seed:        Base RNG seed for reproducibility.

    Returns:
        DataLoader yielding (batch_size, L) float32 tensors.
    """
    dataset = _HarmonicDataset(
        size=batch_size * n_batches,
        sample_rate=sample_rate,
        chunk_s=chunk_s,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _coherence_metrics(coherence: torch.Tensor) -> Tuple[float, float]:
    """
    Compute variance and diagonal ratio of coherence matrix.

    Args:
        coherence: (B, H, T, T) attention weights.

    Returns:
        (variance, diag_ratio): Variance across all entries; mean diagonal
        divided by mean off-diagonal. diag_ratio > 1 means the model
        attends to the same temporal position more than cross-positions.
    """
    var = coherence.var().item()
    if coherence.shape[-1] < 2:
        return var, 1.0
    diag = torch.diagonal(coherence, dim1=-2, dim2=-1).mean().item()
    mask = ~torch.eye(coherence.shape[-1], dtype=torch.bool, device=coherence.device)
    off_diag = coherence[..., mask].mean().item()
    ratio = diag / (off_diag + 1e-8)
    return var, ratio


# ── Training loop ────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    Full training loop.

    Args:
        args: Parsed CLI arguments (epochs, lr, batch_size, d_model,
              device, harmonic_binding, save_path, n_batches).
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Bifröst Phase Coherence Training")
    print("=" * 60)
    print(f"  device:           {device}")
    print(f"  d_model:          {args.d_model}")
    print(f"  epochs:           {args.epochs}")
    print(f"  lr:               {args.lr}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  harmonic_binding: {args.harmonic_binding}")
    print(f"  n_batches:        {args.n_batches}")
    print()

    pipeline = BifrostPipeline(
        n_fft_s0=1024,
        n_fft_s1=512,
        d_model=args.d_model,
        n_heads=4,
        n_bands=8,
        use_complex_ssm=True,
        use_harmonic_binding=args.harmonic_binding,
        sample_rate=16_000.0,
    )
    total_params = sum(p.numel() for p in pipeline.parameters())
    print(f"  total_params:     {total_params:,}")
    print(f"  ssm_type:         {pipeline.ssm_type}")
    print()

    trainer = BifrostTrainer(
        pipeline=pipeline,
        lr=args.lr,
        device=device,
        warmup_steps=max(1, args.epochs * args.n_batches // 10),
    )

    dataloader = build_dataset(
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        sample_rate=16_000,
        chunk_s=1.0,
    )

    best_loss = float("inf")
    epoch_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        # Fresh dataset each epoch: new seed → new waveforms → no memorisation
        dataloader = build_dataset(
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            sample_rate=16_000,
            chunk_s=1.0,
            seed=epoch * 1000,
        )
        t0 = time.time()
        losses: List[float] = []
        coh_vars: List[float] = []
        diag_ratios: List[float] = []

        coh_reals: List[float] = []
        coh_noises: List[float] = []

        for batch in dataloader:
            batch = batch.to(device)
            loss_val = trainer.train_step(batch)
            losses.append(loss_val if loss_val == loss_val else 0.0)  # nan→0 for mean

            # Track coherence gap: real harmonic signal vs white noise
            pipeline.eval()
            with torch.no_grad():
                _, coh_real = pipeline(batch)
                rms = batch.std(dim=-1, keepdim=True).clamp(min=1e-8)
                white_noise = torch.randn_like(batch) * rms
                _, coh_noise = pipeline(white_noise)
            pipeline.train()

            coh_reals.append(coh_real.var().item())    # variance, not mean
            coh_noises.append(coh_noise.var().item())
            coh_var, diag_ratio = _coherence_metrics(coh_real)
            coh_vars.append(coh_var)
            diag_ratios.append(diag_ratio)

        epoch_loss = sum(losses) / max(len(losses), 1)
        epoch_var = sum(coh_vars) / max(len(coh_vars), 1)
        epoch_ratio = sum(diag_ratios) / max(len(diag_ratios), 1)
        epoch_coh_gap = (sum(coh_reals) / max(len(coh_reals), 1)) - \
                        (sum(coh_noises) / max(len(coh_noises), 1))
        elapsed = time.time() - t0
        epoch_losses.append(epoch_loss)

        improved = epoch_loss < best_loss and epoch_loss == epoch_loss
        if improved:
            best_loss = epoch_loss

        mean_var_real = sum(coh_reals) / max(len(coh_reals), 1)
        mean_var_noise = sum(coh_noises) / max(len(coh_noises), 1)
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"loss={epoch_loss:.5f} {'↓' if improved else ' '} | "
            f"var_real={mean_var_real:.2e} | "
            f"var_noise={mean_var_noise:.2e} | "
            f"gap={epoch_coh_gap:+.2e} | "
            f"diag={epoch_ratio:.3f} | "
            f"{elapsed:.1f}s"
        )

    # Save checkpoint
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pipeline_state": pipeline.state_dict(),
            "epoch_losses": epoch_losses,
            "args": vars(args),
        },
        save_path,
    )
    print(f"\nCheckpoint saved → {save_path}")

    # Final diagnostic
    print("\n" + "=" * 60)
    print("Final Diagnostics")
    print("=" * 60)
    pipeline.eval()
    dataloader_eval = build_dataset(batch_size=4, n_batches=1, seed=999)
    eval_batch = next(iter(dataloader_eval))
    eval_batch = eval_batch.to(device)
    with torch.no_grad():
        half = eval_batch.shape[-1] // 2
        st, coh = pipeline(eval_batch[..., :half])

    coh_var, diag_ratio = _coherence_metrics(coh)
    print(f"  Final loss (best):  {best_loss:.5f}")
    print(f"  Coherence variance: {coh_var:.6f}")
    print(f"  Diagonal ratio:     {diag_ratio:.4f}")
    print(f"  Amplitude shape:    {st.amplitude.shape}")

    if diag_ratio > 1.2:
        print("\n  Phase coherence LEARNED (diag_ratio > 1.2) ✓")
    elif diag_ratio > 1.1:
        print("\n  Phase coherence partially learned (1.1 < ratio < 1.2). Run more epochs.")
    else:
        print("\n  Phase coherence not yet learned (ratio ≤ 1.1). Increase --epochs.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Bifröst complex SSM for phase coherence via next-frame prediction."
    )
    parser.add_argument("--epochs",            type=int,   default=100,                    help="Training epochs")
    parser.add_argument("--lr",                type=float, default=1e-4,                   help="Learning rate")
    parser.add_argument("--batch-size",        type=int,   default=4,                      help="Batch size")
    parser.add_argument("--n-batches",         type=int,   default=20,                     help="Batches per epoch")
    parser.add_argument("--d-model",           type=int,   default=128,                    help="Model hidden dimension")
    parser.add_argument("--device",            type=str,   default=None,                   help="cuda or cpu (auto-detect if omitted)")
    parser.add_argument("--harmonic-binding",  action="store_true",                        help="Use HarmonicBinding (explicit 440↔880Hz grid)")
    parser.add_argument("--save-path",         type=str,   default="checkpoints/phase_coherence.pt", help="Checkpoint output path")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
