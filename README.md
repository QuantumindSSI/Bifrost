# Bifrost

**Spectral neural processing with phase-coherent representations**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-blue.svg)](./pyproject.toml)

---

## What it is

Bifrost is a framework for learning representations of meaning from the structure of signals — not from token statistics.

The premise is that **semantic understanding is not a single property but the product of several distinct structural layers**, each requiring different mathematics. Current deep learning (transformers, SSMs, multimodal models) captures distributional statistics well. It does not capture causal direction, topological structure, compositional hierarchy, symmetry, or disentangled factors — not because these are unsolvable but because the representations used discard the information required to reason about them.

Bifrost is being built to address this. Every design decision derives from what structure is actually present in continuous signals and what mathematical object is required to represent it faithfully.

---

## The central mechanism: phase coherence

All inputs — audio, image, text, sensor data — are converted to complex spectra: `z = A·exp(iφ)`, carrying both amplitude and phase. The system then routes information based on **phase coherence** — the alignment of oscillatory phases across frequencies and time — rather than dot-product similarity.

Phase coherence is physically grounded. The Adler equation describes how two coupled oscillators synchronise:

```
dφ/dt = Δω + K·sin(φ_target − φ)
```

Phase-lock occurs when `|Δω| < K`: the coupling strength overcomes the frequency difference. Bifrost implements this as a learned detector over spectral attractors. Two signals that share structural relationships — a spoken word and its visual referent, two harmonically related frequencies, a question and its answer — will phase-lock in ways that structurally unrelated pairs will not.

This is not metaphor. It is a testable, measurable signal with quantifiable thresholds.

---

## The pipeline

Every input passes through the same four-stage pipeline, with each stage operating on `SpectralTensor(amplitude, phase, scale, uncertainty)`:

```
Input  (audio / image / text / sensor / any continuous signal)
  │
  ▼  S0  Canonicalization
  │      STFT → complex spectrum z = A·exp(iφ)
  │      Per-channel normalisation, uncertainty quantification
  │
  ▼  S1  Complex SSM
  │      h[t] = exp(−Δ·A)·h[t−1] + Δ·B[t]·x[t]   (A complex diagonal)
  │      Parallel Blelloch associative scan — O(log L) depth
  │      Learns temporal phase coherence over arbitrary sequence lengths
  │
  ▼  S2  Spectral Binding
  │      C(i,j) = Σ_b w_b · mean_f[cos(φ_q[i,f] − φ_k[j,f])]
  │      Attention over phase alignment across frequency bands
  │      Harmonic overtone routing: energy at f, 2f, 3f, …
  │      Collapse-proof: base coherence derived from canonical phase, no learned params
  │
  ▼  S3  Phase-Lock Bridge
         Stable frequency attractors identified via Adler equation dynamics
         VQ-VAE codebook (65K entries) over attractor feature space
         Cross-modal transfer: attractors from different domains phase-lock
         when they share structural relationships
```

**S4 (optional):** Riemannian metric over attractor embeddings via Cholesky factorisation `G = LLᵀ`. Geodesic distances between attractors serve as semantic coherence scores. Triplet loss trains the metric so synonyms are close and antonyms are far on the manifold.

---

## Where this is going

The current pipeline addresses two of the seven structural layers that constitute semantic understanding. The roadmap extends it to all seven:

```
Layer 1  DISTRIBUTIONAL        Statistical co-occurrence in context
         Current state:  SpectralBinding captures spectral covariance
         LLM comparison: LLMs do this well; Bifrost adds structural co-occurrence

Layer 2  COMPOSITIONAL         Recursive part-whole structure
         Current state:  Not yet implemented
         Roadmap:        Hierarchical SSM — five-level timescale pyramid
                         10ms (phoneme) → 100ms (syllable) → 500ms (word)
                         → 2s (phrase) → 10s (discourse)
                         Cross-level attention learns which timescale is active per frame

Layer 3  CAUSAL                Directed influence, counterfactual reasoning
         Current state:  Not yet implemented. All current representations are symmetric.
         Roadmap:        Granger causality over spectral bands from SSM transition matrices
                         GC(i→j) = log( Var[ê_j^(−i)] / Var[ê_j] )
                         First asymmetric (directed) signal in the pipeline
                         Enables: "band A predicts band B" ≠ "band B predicts band A"

Layer 4  TOPOLOGICAL           Curved manifold structure of concept space
         Current state:  RiemannianMetricLearner — learned metric G = LLᵀ (implemented)
         Roadmap:        TDA persistence diagrams — Betti numbers [β₀, β₁, β₂]
                         Parameter-free topological fingerprints, no training required
                         Different phonemes, chords, and word classes have distinct
                         topological signatures that Riemannian geometry alone misses

Layer 5  TEMPORAL              Discourse structure, event sequences, narrative arcs
         Current state:  Complex SSM captures temporal phase coherence (implemented)
         Roadmap:        Allen interval algebra over attractor activations
                         13 qualitative temporal relations: before, meets, overlaps,
                         starts, during, finishes, equals, and their inverses
                         Enables explicit narrative structure representation

Layer 6  SYMMETRY              What transformations leave meaning invariant
         Current state:  HarmonicBinding hardcodes octave invariance (f → 2f)
         Roadmap:        SymmetryTensor — detect the invariance group the signal
                         actually obeys rather than assuming it
                         Generalises to speech formants, image rotations, sensor
                         periodicities at non-integer frequency ratios

Layer 7  DISENTANGLEMENT       Statistically independent generative factors
         Current state:  Not yet implemented. VQ-VAE mixes content, style, and noise.
         Roadmap:        Total Correlation VAE — penalise TC(z) = KL(q(z) || ∏ q(z_i))
                         Separate content (what) from style (how) from temporal (when)
                         MI matrix I(z_i; z_j) quantifies residual entanglement
                         Enables controlled style transfer without retraining
```

LLMs address Layer 1 well and Layer 5 partially within context window limits. Layers 2–4 and 6–7 are structurally unaddressed by token-based architectures — not because transformers are inadequate at their task, but because the information required for these layers is discarded before the architecture ever sees the input.

---

## LLM integration

Bifrost is not a replacement for language models. It is a complement. Three integration modes are implemented:

**Spectral prefix** — Bifrost encodes audio, image, or sensor input into spectral embeddings projected as prefix tokens. A frozen LLM receives `[spectral prefix | text tokens]` and processes them jointly. Provides grounded multimodal context without retraining the language model.

**Parameter-efficient adapter** — The complex SSM is injected between frozen LLM layers. ~5M trainable parameters (~4% of GPT-2). Provides long-context phase coherence tracking and per-token uncertainty estimates beyond the context window.

**Structural coherence verifier** — For chain-of-thought reasoning: each reasoning step is encoded via Bifrost alongside the problem context. Phase-lock score and prediction error provide a structural consistency signal — a step that is structurally incoherent with the problem will have low phase-lock and high prediction error. This signal is grounded in physics, not in a learned reward model, and cannot be gamed by reward hacking.

---

## What the pipeline enables beyond language

**Pure spectral sequence model** — EEG, ECG, industrial vibration, sonar, IMU data processed directly in the frequency domain. No tokenisation, no vocabulary, no discrete bottleneck. The complex SSM maintains state across arbitrary sequence lengths.

**Phase-lock cross-modal fusion** — Structural correspondence between modalities computed via oscillator physics, not cross-attention over tokens. Audio and visual patterns that share temporal structure will phase-lock; structurally unrelated pairs will not. No language intermediary required.

**Attractor-based geometric reasoning** — The Riemannian manifold as a reasoning substrate. Analogy completion is a geodesic midpoint computation. Problem solving is a path from a problem attractor to a solution attractor. The geometry is learned, not hand-designed.

**Hierarchical timescale processor** — Five parallel SSMs at 10ms, 100ms, 500ms, 2s, 10s capture phoneme, syllable, word, phrase, and discourse structure simultaneously in a single forward pass.

---

## Current implementation status

| Component | Status |
|---|---|
| SpectralCanonicalizer (S0) | Complete |
| ComplexSpectralDecomposer — complex SSM, Blelloch scan (S1) | Complete |
| SpectralNormalization for complex layers (S1) | Complete |
| ResonanceAttention — multi-band phase coherence (S2) | Complete |
| HarmonicBinding — overtone-series routing (S2) | Complete |
| SpectralBinding — collapse-proof pipeline (S2) | Complete |
| PhaseLockBridge — cross-modal attractor bridging (S3) | Complete |
| TruePhaseLockDetector — Adler equation (S3) | Complete |
| AttractorLearningModule — VQ-VAE 65K codebook (S3) | Complete |
| RiemannianMetricLearner + GeodesicComputer (S4) | Complete |
| BifrostEnhancedLLM — 3 adapter modes | Architecture complete, untrained |
| Distributed training infrastructure (DDP, 8× A100) | Complete |
| Empirical validation suite | Complete |
| Multi-modal data curation pipeline | Complete — data ingestion in progress |
| Hierarchical multi-timescale SSM | Planned |
| Granger causal graph | Planned |
| TDA persistence diagrams | Planned |
| Symmetry detection | Planned |
| Disentangled VAE | Planned |
| Allen interval algebra | Planned |

The full engineering specification — component interfaces, mathematical derivations, timelines, and success criteria for each layer — is in [ENGINEERING_PLAN.md](./ENGINEERING_PLAN.md).

---

## Installation

```bash
git clone https://github.com/QuantumindSSI/Bifrost.git
cd Bifrost
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

signal = torch.randn(1, 16000)       # 1s at 16 kHz
output, coherence = pipe(signal)

print(output.amplitude.shape)        # spectral embedding
print(coherence.mean().item())       # phase coherence score ∈ [0, 1]
```

```python
# Stateful streaming — SSM state persists across chunk boundaries
h = None
for chunk in audio_stream:
    output, coherence, h = pipe.forward_stateful(chunk, h_0=h)
```

```python
# Multi-modal
from bifrost.multimodal_pipeline import create_multimodal_pipeline, Modality

pipe = create_multimodal_pipeline(d_model=256)
output = pipe(signal, modality=Modality.AUDIO)
```

```python
# LLM integration
from bifrost.llm_adapter import BifrostEnhancedLLM

model = BifrostEnhancedLLM(
    llm_name="gpt2",
    adapter_mode="intermediate",
    spectral_dim=128,
    freeze_llm=True,
)
result = model.generate_with_spectral("The experiment showed", max_length=50)
```

---

## CLI

```bash
bifrost process audio.wav -o output.pt       # full pipeline
bifrost attractors audio.wav -o att.pt       # attractor extraction
bifrost bridge a.pt b.pt --min-locked 3      # cross-modal bridge evaluation
bifrost demo 1                               # anti-phase discrimination
bifrost demo 2                               # harmonic binding
bifrost demo 3                               # cross-modal retrieval
bifrost bench attention                      # ResonanceAttention vs dot-product
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
├── pipeline.py                 # BifrostPipeline
├── multimodal_pipeline.py      # Per-modality routing
├── llm_adapter.py              # LLM integration
├── canonicalizer/              # S0
├── decomposer/                 # S1 — complex SSM, associative scan, spectral norm
├── resonance_attention/        # S2 — phase coherence, harmonic binding
├── phase_lock_bridge/          # S3 — attractor extraction, Adler detection
├── s3_attractor/               # S3 — VQ-VAE attractor learning
├── riemannian_coherence/       # S4 — Riemannian metric, geodesics
├── tokenization/               # Discrete attractor tokeniser
├── semantic_coherence/         # Training objectives
├── validation/                 # Empirical validation suite
├── ingest/                     # Raw media ingestion
├── training.py                 # BifrostTrainer, contrastive phase loss
├── distributed_training.py     # DDP multi-GPU/multi-node
├── checkpoint_manager.py       # Versioned checkpointing
└── evaluation.py               # Evaluation metrics
```

---

## References

- Gu & Dao (2023) — Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Trabelsi et al. (2018) — Deep Complex Networks
- Blelloch (1990) — Prefix Sums and Their Applications
- Adler (1946) — A Study of Locking Phenomena in Oscillators
- Nickel & Kiela (2017) — Poincaré Embeddings for Learning Hierarchical Representations
- Pearl (2009) — Causality: Models, Reasoning, and Inference
- Carlsson (2009) — Topology and Data
- Allen (1983) — Maintaining Knowledge about Temporal Intervals
- Higgins et al. (2017) — beta-VAE: Learning Basic Visual Concepts with a Constrained VAE

---

## License

MIT. See [LICENSE](./LICENSE).

## Contact

**Quantumind** — engineering@quantumind.io
