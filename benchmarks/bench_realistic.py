"""
Production-style benchmark for ResonanceAttention on realistic inputs.

Reports p50 / p95 / p99 latency, throughput (signals/sec), and memory
high-water mark over many repeats, on inputs sized to match real audio
and image pipelines.

Run:
    PYTHONPATH=. python benchmarks/bench_realistic.py
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.io import wavfile

from bifrost.bridge import bridge_to_canonicalizer
from bifrost.pipeline import BifrostPipeline


@dataclass
class Percentiles:
    p50: float
    p95: float
    p99: float
    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def from_times_ms(cls, times: List[float]) -> "Percentiles":
        sorted_t = sorted(times)
        return cls(
            p50=statistics.median(sorted_t),
            p95=sorted_t[int(0.95 * (len(sorted_t) - 1))],
            p99=sorted_t[int(0.99 * (len(sorted_t) - 1))],
            mean=statistics.mean(sorted_t),
            std=statistics.stdev(sorted_t) if len(sorted_t) > 1 else 0.0,
            min=min(sorted_t),
            max=max(sorted_t),
        )


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_wav(path: str):
    sr, data = wavfile.read(path)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data, sr


def benchmark_signal(
    pipeline: BifrostPipeline,
    signal: np.ndarray,
    meta: Dict,
    label: str,
    warmup: int = 5,
    repeats: int = 100,
) -> Dict:
    """Run a signal through the pipeline repeats times; report percentiles."""
    sig_t, enriched = bridge_to_canonicalizer(signal, meta)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = pipeline(sig_t, enriched)
    _sync()

    # Measurement
    times_ms: List[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = pipeline(sig_t, enriched)
        _sync()
        times_ms.append((time.perf_counter() - t0) * 1000)

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    pct = Percentiles.from_times_ms(times_ms)
    throughput = 1000.0 / pct.p50  # signals per second using p50

    return {
        "label": label,
        "input_shape": list(sig_t.shape),
        "channels": enriched.get("channels"),
        "samples": enriched.get("num_samples"),
        "sample_rate": enriched.get("sample_rate"),
        "repeats": repeats,
        "latency_ms": asdict(pct),
        "throughput_per_sec": throughput,
        "peak_mb": peak_mb,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    torch.manual_seed(0)

    # Build a single shared pipeline
    pipeline = BifrostPipeline(
        n_fft_canonical=1024, n_fft_decompose=256, d_model=128, n_heads=4, n_bands=8,
    )
    pipeline.eval()

    results: List[Dict] = []

    # ── Real WAV files ────────────────────────────────────────────────
    wav_specs = [
        ("sample_data/mono_8khz.wav",   "mono_8khz"),
        ("sample_data/mono_16khz.wav",  "mono_16khz"),
        ("sample_data/stereo_44khz.wav","stereo_44khz"),
    ]
    for path, label in wav_specs:
        try:
            data, sr = _load_wav(path)
            meta = {
                "format": "wav",
                "channels": 1 if data.ndim == 1 else data.shape[-1],
                "sample_rate": sr,
            }
            r = benchmark_signal(pipeline, data, meta, label)
            results.append(r)
            gc.collect()
        except FileNotFoundError:
            print(f"  skip {path} (not found)")

    # ── Synthetic worst-case ──────────────────────────────────────────
    # 1 second of stereo at 48 kHz — a realistic production frame
    synth_specs = [
        ("stereo_48khz_1s", np.random.randn(48000, 2).astype(np.float32),
         {"format": "wav", "channels": 2, "sample_rate": 48000}),
        ("mono_16khz_4s", np.random.randn(64000).astype(np.float32),
         {"format": "wav", "channels": 1, "sample_rate": 16000}),
        ("8ch_16khz_1s", np.random.randn(8, 16000).astype(np.float32),
         {"format": "npy", "channels": 8, "channel_axis": 0,
          "sample_rate": 16000}),
    ]
    for label, data, meta in synth_specs:
        r = benchmark_signal(pipeline, data, meta, label)
        results.append(r)
        gc.collect()

    # ── Report ────────────────────────────────────────────────────────
    header = (f"{'Label':>18s} │ {'Shape':>15s} │ "
              f"{'p50 (ms)':>9s} │ {'p95 (ms)':>9s} │ "
              f"{'p99 (ms)':>9s} │ {'sig/sec':>8s}")
    print(header)
    print("─" * len(header))
    for r in results:
        shape = "×".join(str(d) for d in r["input_shape"])
        lat = r["latency_ms"]
        print(f"{r['label']:>18s} │ {shape:>15s} │ "
              f"{lat['p50']:>9.2f} │ {lat['p95']:>9.2f} │ "
              f"{lat['p99']:>9.2f} │ {r['throughput_per_sec']:>8.1f}")

    # Persist as artifact for CI consumption
    out_path = Path("benchmarks/results_realistic.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults persisted to {out_path}")


if __name__ == "__main__":
    main()
