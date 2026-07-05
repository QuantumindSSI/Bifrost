# Bifrost

**Spectral neural processing with phase-coherent representations**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](./pyproject.toml)

---

Bifrost is a PyTorch framework for processing continuous signals through the frequency domain. It represents all inputs as complex spectra — carrying both amplitude and phase — and routes information based on phase coherence rather than scalar similarity.

The core claim: **phase alignment encodes structural relationships that amplitude alone cannot**. Two signals with identical spectra but opposite phase are structurally antithetical. Bifrost makes this distinction computationally tractable.

---

## Pipeline

Every input passes through four stages, each operating on `SpectralTensor(amplitude, phase, scale, uncertainty)`:

```
Input (audio / image / text / sensor)
  │
  ▼  S0  Canonicalization     STFT → complex spectrum z = A·exp(iφ)
  │
  ▼  S1  Complex SSM          h[t] = exp(−Δ·A)·h[t−1] + Δ·B[t]·x[t]
  │                           Parallel Blelloch scan — O(log L) depth
  │
  ▼  S2  Spectral Binding     C(i,j) = Σ_b w_b · mean_f[cos(φ_q[i,f] − φ_k[j,f])]
  │                           Attention over phase alignment, not dot products
  │
  ▼  S3  Phase-Lock Bridge    Stable attractor extraction via Adler equation
                              Cross-modal transfer grounded in oscillator physics
```

An optional S4 stage fits a Riemannian metric (G = LLᵀ via Cholesky) over attractor embeddings and computes geodesic distances for semantic coherence scoring.

---

## Installation

```bash
git clone https://github.com/quantumind/bifrost.git
cd bifrost
pip install -e .

# With development tooling
pip install -e ".[dev]"
```

**Requirements:** Python 3.9+, PyTorch 2.3+, numpy, scipy, librosa, Pillow

---

## Quick start

```python
import torch
from bifrost.pipeline import BifrostPipeline

pipe = BifrostPipeline(d_model=256, use_complex_ssm=True, use_s3_attractor=True)

signal = torch.randn(1, 16000)          # 1s at 16 kHz
output, coherence = pipe(signal)

print(output.amplitude.shape)           # spectral embedding
print(coherence.mean().item())          # phase coherence score [0, 1]
```

```python
# Stateful streaming — SSM state persists across chunks
h = None
for chunk in audio_stream:
    output, coherence, h = pipe.forward_stateful(chunk, h_0=h)
```

```python
# Multi-modal: audio, image, text, or sensor in the same pipeline
from bifrost.multimodal_pipeline import create_multimodal_pipeline, Modality

pipe = create_multimodal_pipeline(d_model=256)
output = pipe(signal, modality=Modality.AUDIO)
```

---

## Key components

| Module | Class | What it does |
|---|---|---|
| `canonicalizer` | `SpectralCanonicalizer` | STFT → `SpectralTensor` with per-channel normalisation and uncertainty |
| `decomposer` | `ComplexSpectralDecomposer` | Complex-valued SSM; parallel Blelloch scan for O(log L) depth |
| `decomposer` | `ComplexSpectralNorm` | Spectral normalisation for complex layers; stabilises Lipschitz constant |
| `resonance_attention` | `ResonanceAttention` | Multi-band phase-coherence attention |
| `resonance_attention` | `HarmonicBinding` | Explicit overtone-series routing (f, 2f, 3f, …) |
| `resonance_attention` | `SpectralBinding` | Collapse-proof pipeline; coherence derived from canonical phase |
| `phase_lock_bridge` | `PhaseLockBridge` | Cross-modal attractor bridging |
| `phase_lock_bridge` | `TruePhaseLockDetector` | Adler-equation phase-lock detection |
| `s3_attractor` | `AttractorLearningModule` | VQ-VAE codebook over stable spectral patterns |
| `riemannian_coherence` | `RiemannianMetricLearner` | Learned Riemannian metric; geodesic semantic distances |
| `llm_adapter` | `BifrostEnhancedLLM` | Injects spectral representations into frozen LLMs |
| `training` | `BifrostTrainer` | Contrastive phase loss + InfoNCE |
| `uncertainty_calibration` | `UncertaintyCalibrator` | Temperature scaling; evaluates via ECE |

---

## CLI

```bash
# Process a file through the full pipeline
bifrost process audio.wav -o output.pt

# Extract frequency attractors
bifrost attractors audio.wav -o attractors.pt

# Evaluate cross-domain phase-lock bridge
bifrost bridge set_a.pt set_b.pt --min-locked 3

# Run built-in demos
bifrost demo 1          # anti-phase discrimination
bifrost demo 2          # harmonic binding
bifrost demo 3          # cross-modal retrieval

# Benchmarks
bifrost bench attention  # ResonanceAttention vs dot-product
bifrost bench realistic  # throughput benchmark
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=bifrost --cov-report=html
```

---

## Project layout

```
src/bifrost/
├── spectral_tensor.py          # SpectralTensor dataclass
├── pipeline.py                 # BifrostPipeline (end-to-end)
├── multimodal_pipeline.py      # Per-modality routing
├── llm_adapter.py              # LLM integration
├── canonicalizer/              # S0 — STFT canonicalisation
├── decomposer/                 # S1 — complex SSM + associative scan
├── resonance_attention/        # S2 — phase-coherence binding
├── phase_lock_bridge/          # S3 — attractor extraction + bridging
├── s3_attractor/               # S3 — learned attractor dynamics
├── riemannian_coherence/       # S4 — Riemannian metric + geodesics
├── tokenization/               # Discrete attractor tokeniser
├── semantic_coherence/         # Training objectives and metrics
├── validation/                 # Empirical claim validation suite
├── ingest/                     # Raw media ingestion (audio, image, text)
├── training.py                 # BifrostTrainer
├── distributed_training.py     # DDP multi-GPU/multi-node
├── checkpoint_manager.py       # Versioned checkpointing
├── contrastive_loss.py         # ContrastivePhaseLoss + InfoNCE
└── evaluation.py               # Evaluation metrics
```

---

## References

- Gu & Dao (2023) — Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Trabelsi et al. (2018) — Deep Complex Networks
- Blelloch (1990) — Prefix Sums and Their Applications
- Adler (1946) — A Study of Locking Phenomena in Oscillators
- Nickel & Kiela (2017) — Poincaré Embeddings for Learning Hierarchical Representations

---

## License

MIT. See [LICENSE](./LICENSE).

## Contact

**Quantumind** — engineering@quantumind.io
