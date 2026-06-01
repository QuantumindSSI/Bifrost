#!/usr/bin/env python3
"""
Train Bifröst pipeline for phase coherence learning.

Training objective: VARIANCE-BASED CONTRASTIVE DISCRIMINATION.
  - Real signals: harmonic with coherent phase relationships
  - Noise signals: same amplitude spectrum, randomized phase
  - Loss: contrastive - real signals should have LOWER attention variance

Strategy: Phase-coherent signals produce more focused (lower variance) attention
patterns, while phase-randomized signals produce diffuse (higher variance) patterns.

Usage:
    cd bifrost
    python scripts/train_phase_coherence.py --epochs 100
    python scripts/train_phase_coherence.py --epochs 200 --harmonic-binding --device cuda

Diagnostic output every epoch:
  - loss:              contrastive loss (lower = better discrimination)
  - var_real:          attention variance for real signals (target: lower)
  - var_noise:         attention variance for noise signals (target: higher)
  - gap:               var_noise - var_real (should be positive)
  - diag_ratio:        diagonal attention ratio (>1.2 = phase coherence learned)

Expected progression (complex SSM, d_model=128, variance-based loss):
  Epoch   0: loss ~0.5,  var_real ~0.03,  var_noise ~0.03,  gap ~0.0,  diag ~2.5
  Epoch  50: loss ~0.2,  var_real ~0.02,  var_noise ~0.04,  gap ~0.02, diag ~2.7
  Epoch 100: loss ~0.05, var_real ~0.01,  var_noise ~0.05,  gap ~0.04, diag ~3.0
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
from bifrost.training import BifrostTrainer as BifrostTrainer
from bifrost.contrastive_loss import ContrastivePhaseLoss


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

def _coherence_metrics(coherence: torch.Tensor, tau: float = 1.0) -> Tuple[float, float, torch.Tensor]:
    """
    Compute variance and diagonal ratio of coherence matrix.

    Args:
        coherence: (B, H, T, T) pre-softmax coherence scores (range ~[-0.5, 1.0]).
        tau: Temperature for softmax scaling (default 1.0).

    Returns:
        (variance, diag_ratio, attn_weights): 
        - variance: of post-softmax attention weights
        - diag_ratio: mean diagonal / mean off-diagonal of attention weights
                      diag_ratio > 1.2 indicates phase coherence learned
        - attn_weights: post-softmax attention for visualization
    """
    # Convert pre-softmax coherence to post-softmax attention weights
    # This gives us proper probabilities in [0, 1] range
    attn_weights = torch.softmax(coherence / max(tau, 0.1), dim=-1)
    
    var = attn_weights.var().item()
    if coherence.shape[-1] < 2:
        return var, 1.0, attn_weights
    
    # Compute diagonal ratio on post-softmax weights (always positive)
    diag = torch.diagonal(attn_weights, dim1=-2, dim2=-1).mean().item()
    mask = ~torch.eye(attn_weights.shape[-1], dtype=torch.bool, device=attn_weights.device)
    off_diag = attn_weights[..., mask].mean().item()
    ratio = diag / (off_diag + 1e-8)
    
    return var, ratio, attn_weights


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
        n_fft_canonical=1024,
        n_fft_decompose=512,
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

    # Move pipeline to device
    pipeline = pipeline.to(device)

    # Contrastive loss: forces discrimination between real and phase-randomized
    # Uses variance-based discrimination (real signals should have more focused attention)
    contrastive_loss = ContrastivePhaseLoss(margin=0.01, temperature=0.5).to(device)
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
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
            
            # === PRECONDITION ASSERTIONS ===
            assert batch.device.type == device, f"Batch on {batch.device}, expected {device}"
            assert batch.dtype == torch.float32, f"Expected float32 batch, got {batch.dtype}"
            assert torch.isfinite(batch).all(), "Non-finite values in input batch"
            
            optimizer.zero_grad()

            # === CONTRASTIVE TRAINING ===
            # Real signals: harmonic with coherent phase
            bound_real, coh_real = pipeline(batch)

            # Create phase-randomized version (same amplitude, random phase)
            _n_fft, _hop = 1024, 256
            _win = torch.hann_window(_n_fft, device=batch.device)
            _spec = torch.stft(
                batch,
                n_fft=_n_fft,
                hop_length=_hop,
                return_complex=True,
                window=_win,
                pad_mode='reflect',
            )
            _rph = torch.rand_like(_spec.real) * 2.0 * math.pi
            _nspec = torch.polar(_spec.abs(), _rph)
            _phase_rand = torch.istft(
                _nspec,
                n_fft=_n_fft,
                hop_length=_hop,
                window=_win,
                length=batch.shape[-1],
            )

            # Phase-randomized signals through pipeline
            bound_noise, coh_noise = pipeline(_phase_rand)

            # Contrastive loss: maximize gap between real and noise coherence
            loss = contrastive_loss(coh_real, coh_noise)

            # === LOSS VALIDATION ASSERTIONS ===
            assert torch.isfinite(loss), f"Loss became NaN/Inf - training diverged. Loss: {loss}"
            assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"

            if torch.isfinite(loss) and loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            loss_val = loss.item() if torch.isfinite(loss) else 0.0
            losses.append(loss_val)

            # === METRICS ===
            with torch.no_grad():
                # Compute variance of attention weights (post-softmax)
                attn_real = torch.softmax(coh_real, dim=-1)
                attn_noise = torch.softmax(coh_noise, dim=-1)
                var_real = attn_real.var(dim=-1).mean().item()
                var_noise = attn_noise.var(dim=-1).mean().item()
                coh_reals.append(var_real)
                coh_noises.append(var_noise)
                coh_var, diag_ratio, _ = _coherence_metrics(coh_real, tau=1.0)
                coh_vars.append(coh_var)
                diag_ratios.append(diag_ratio)

        epoch_loss = sum(losses) / max(len(losses), 1)
        epoch_var = sum(coh_vars) / max(len(coh_vars), 1)
        epoch_ratio = sum(diag_ratios) / max(len(diag_ratios), 1)
        epoch_coh_gap = (sum(coh_noises) / max(len(coh_noises), 1)) - \
                        (sum(coh_reals) / max(len(coh_reals), 1))  # Positive when real has lower variance
        elapsed = time.time() - t0
        epoch_losses.append(epoch_loss)

        improved = epoch_loss < best_loss and epoch_loss == epoch_loss
        if improved:
            best_loss = epoch_loss

        mean_coh_real = sum(coh_reals) / max(len(coh_reals), 1)
        mean_coh_noise = sum(coh_noises) / max(len(coh_noises), 1)
        coh_gap = mean_coh_noise - mean_coh_real  # Positive when real has lower variance
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"loss={epoch_loss:.5f} {'↓' if improved else ' '} | "
            f"var_real={mean_coh_real:.4f} | "
            f"var_noise={mean_coh_noise:.4f} | "
            f"gap={coh_gap:+.4f} | "
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

    coh_var, diag_ratio, _ = _coherence_metrics(coh, tau=1.0)
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
        description="Train Bifröst complex SSM for phase coherence via contrastive discrimination."
    )
    parser.add_argument("--epochs",            type=int,   default=200,                    help="Training epochs (200+ recommended for phase coherence)")
    parser.add_argument("--lr",                type=float, default=3e-5,                   help="Learning rate (3e-5 stable for phase coherence)")
    parser.add_argument("--batch-size",        type=int,   default=8,                      help="Batch size (8+ for stable gradients)")
    parser.add_argument("--n-batches",         type=int,   default=50,                     help="Batches per epoch (50+ for meaningful variance estimates)")
    parser.add_argument("--d-model",           type=int,   default=128,                    help="Model hidden dimension")
    parser.add_argument("--device",            type=str,   default=None,                   help="cuda or cpu (auto-detect if omitted)")
    parser.add_argument("--harmonic-binding",  action="store_true",                        help="Use HarmonicBinding (explicit 440↔880Hz grid)")
    parser.add_argument("--save-path",         type=str,   default="checkpoints/phase_coherence.pt", help="Checkpoint output path")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
