"""
Benchmark: ResonanceAttention vs standard dot-product MultiHeadAttention.

Measures:
    1. Forward latency (ms)
    2. Forward + backward latency (ms)
    3. Peak memory (MB)
    4. Output quality: coherence structure for phase-aligned vs random inputs

Run:
    python benchmarks/bench_attention.py
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bifrost.resonance_attention.attention import ResonanceAttention


# ── Standard dot-product MHA baseline ──────────────────────────────────────
class DotProductAttention(nn.Module):
    """Vanilla multi-head dot-product attention (baseline)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.W_o(out)
        out = self.norm(out + x)
        return out, weights


# ── Timing helpers ─────────────────────────────────────────────────────────
@dataclass
class BenchResult:
    name: str
    fwd_ms: float
    fwd_bwd_ms: float
    peak_mb: float
    param_count: int


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_module(name: str, module: nn.Module, x: torch.Tensor,
                 warmup: int = 5, repeats: int = 20) -> BenchResult:
    """Benchmark forward and forward+backward latency & peak memory."""
    device = x.device
    module = module.to(device)
    module.train()

    # Warmup
    for _ in range(warmup):
        out, _ = module(x)
        out.sum().backward()
        module.zero_grad()

    # Forward only
    _sync()
    fwd_times: List[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = module(x)
        _sync()
        fwd_times.append((time.perf_counter() - t0) * 1000)

    # Forward + backward
    fwd_bwd_times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out, _ = module(x)
        out.sum().backward()
        _sync()
        fwd_bwd_times.append((time.perf_counter() - t0) * 1000)
        module.zero_grad()

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    param_count = sum(p.numel() for p in module.parameters())

    return BenchResult(
        name=name,
        fwd_ms=sum(fwd_times) / len(fwd_times),
        fwd_bwd_ms=sum(fwd_bwd_times) / len(fwd_bwd_times),
        peak_mb=peak_mb,
        param_count=param_count,
    )


# ── Coherence quality test ─────────────────────────────────────────────────
def coherence_quality_test(d_model: int = 128, n_heads: int = 4, n_bands: int = 8):
    """
    Test whether ResonanceAttention distinguishes phase-aligned signals
    from random signals better than dot-product attention.
    """
    torch.manual_seed(42)
    res_attn = ResonanceAttention(d_model=d_model, n_heads=n_heads, n_bands=n_bands, dropout=0.0)
    dot_attn = DotProductAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    res_attn.eval()
    dot_attn.eval()

    # Scenario A: phase-aligned tokens (identical signals repeated)
    base = torch.randn(1, 1, d_model)
    aligned = base.expand(1, 8, d_model).clone()

    # Scenario B: random tokens
    random_input = torch.randn(1, 8, d_model)

    with torch.no_grad():
        _, res_w_aligned = res_attn(aligned)
        _, res_w_random = res_attn(random_input)
        _, dot_w_aligned = dot_attn(aligned)
        _, dot_w_random = dot_attn(random_input)

    # Measure uniformity of attention (entropy proxy)
    def attn_entropy(w: torch.Tensor) -> float:
        """Mean entropy across heads and queries."""
        w = w.clamp(min=1e-9)
        ent = -(w * w.log()).sum(dim=-1)
        return ent.mean().item()

    # Measure coherence discrimination = entropy(aligned) - entropy(random)
    # Higher means the model sees aligned signals as more uniform (correct)
    res_ent_a = attn_entropy(res_w_aligned)
    res_ent_r = attn_entropy(res_w_random)
    dot_ent_a = attn_entropy(dot_w_aligned)
    dot_ent_r = attn_entropy(dot_w_random)

    print("\n── Coherence Quality ──────────────────────────")
    print(f"{'':20s} {'Aligned':>10s} {'Random':>10s} {'Δ (A-R)':>10s}")
    print(f"{'Resonance Attn':20s} {res_ent_a:10.4f} {res_ent_r:10.4f} {res_ent_a - res_ent_r:10.4f}")
    print(f"{'Dot-Product Attn':20s} {dot_ent_a:10.4f} {dot_ent_r:10.4f} {dot_ent_a - dot_ent_r:10.4f}")
    print()
    print("Higher aligned entropy → more uniform weights (correct for identical inputs).")
    print("Δ close to 0 → can't distinguish aligned from random.")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    configs = [
        # (d_model, n_heads, batch, seq_len)
        (64,  4,  8, 32),
        (128, 4,  4, 64),
        (256, 8,  2, 128),
        (512, 8,  1, 256),
    ]

    print(f"{'Config':>25s} │ {'Module':>18s} │ {'Fwd(ms)':>8s} │ {'F+B(ms)':>8s} │ {'Peak(MB)':>9s} │ {'Params':>8s}")
    print("─" * 100)

    for d_model, n_heads, batch, seq_len in configs:
        n_bands = min(8, d_model // n_heads)
        tag = f"d={d_model} h={n_heads} b={batch} s={seq_len}"

        x = torch.randn(batch, seq_len, d_model, device=device, requires_grad=True)

        res = ResonanceAttention(d_model=d_model, n_heads=n_heads, n_bands=n_bands, dropout=0.0)
        dot = DotProductAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)

        r_res = bench_module("Resonance", res, x)
        gc.collect()
        r_dot = bench_module("DotProduct", dot, x)
        gc.collect()

        for r in (r_res, r_dot):
            print(f"{tag:>25s} │ {r.name:>18s} │ {r.fwd_ms:8.2f} │ {r.fwd_bwd_ms:8.2f} │ {r.peak_mb:9.2f} │ {r.param_count:>8d}")
        print()

    coherence_quality_test()


if __name__ == "__main__":
    main()
