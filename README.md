# Bifrost

**Testing the Structured Resonance Thesis: intelligence is structured resonance, and phase-coherent multi-scale representations capture semantic structure across modalities.**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

---

## The thesis

**Intelligence is structured resonance. Semantic structure is encoded in the phase coherence of oscillatory components across multiple scales, and this principle generalizes across all modalities.**

This is not a metaphor. It is a testable claim with five load-bearing sub-claims:

| Claim | Statement | Evidence required |
|---|---|---|
| C1 | Phase coherence captures semantic structure | Phase-coherent features > phase-invariant features on semantic tasks |
| C2 | Multi-scale coherence is necessary | Cross-scale coherence > single-scale coherence |
| C3 | The principle generalizes across modalities | Same coherence metric works on audio, image, sensor |
| C4 | Cross-modal alignment is possible | Phase coherence patterns align across modalities |
| C5 | This enables AGI-level generalization | Compositional generalization from phase structure |

Bifrost is the engineering framework designed to prove or falsify each claim. The engineering IS the experiment.

---

## Research backing the thesis

The thesis is grounded in converging evidence from multiple independent research traditions:

### Phase congruency detects image structure

Kovesi, P. (1999). "Image Features From Phase Congruency." *Videre: Journal of Computer Vision Research*, 1(3). — Image features (edges, lines, corners) appear at points where Fourier components are maximally in phase. Phase congruency is invariant to illumination and contrast, providing an absolute measure of feature significance.

### Wavelet coherence captures cross-scale phase relationships

Grinsted, A., Moore, J.C., Jevrejeva, S. (2004). "Application of the cross wavelet transform and wavelet coherence to geophysical time series." *Nonlinear Processes in Geophysics*, 11, 561-566. — Wavelet coherence finds significant phase relationships between time series even when common power is low. Phase angle statistics reveal causal relationships.

### Neural networks have spectral bias toward low frequencies

Rahaman, N. et al. (2019). "On the Spectral Bias of Neural Networks." *ICML 2019*, PMLR 97:5301-5310. — Deep ReLU networks learn low-frequency functions first, with frequency-dependent learning speed. This bias is key to generalization. Operating in the frequency domain allows direct control of which frequencies the system learns.

### Theta-gamma coupling is a ubiquitous brain mechanism

(2024). "Theta-gamma coupling as a ubiquitous brain mechanism: implications for memory, attention, dreaming, imagination, and consciousness." *Current Opinion in Behavioral Sciences*. — Cross-frequency coupling between slow (theta) and fast (gamma) oscillations is a primary mechanism for information integration, memory, and consciousness. Different frequency bands serve different cognitive functions.

### Phase synchronization encodes semantic structure

(2019). "Neural theta oscillations support semantic memory retrieval." *Scientific Reports*. — Theta-band phase synchronization is causally involved in semantic memory retrieval. Phase-specific stimulation modulates semantic processing.

(2019). "EEG phase synchronization during semantic unification relates to individual differences in children's vocabulary skill." *Developmental Cognitive Neuroscience*. — Delta-band phase synchrony supports top-down semantic unification. Children with stronger vocabulary show greater phase synchrony.

(2024). "Binding of cortical functional modules by synchronous high-frequency oscillations." *Nature Human Behaviour*. — Cortico-cortical co-ripples (~90 Hz) increase during reading and semantic decisions. Phase-locked at zero lag over long distances.

### Cross-modal integration relies on phase alignment

(2015). "Neuro-Oscillatory Phase Alignment Drives Speeded Multisensory Response Times." *Journal of Neuroscience*. — Delta-band phase alignment between auditory and sensorimotor cortex drives faster multisensory responses. Phase alignment is the mechanism of cross-modal integration.

(2010). "Auditory Cortex Tracks Both Auditory and Visual Stimulus Dynamics Using Low-Frequency Neuronal Phase Modulation." *PLOS Biology*. — Delta-theta (2-7 Hz) phase modulation carries dynamic multi-sensory information, tracking both auditory and visual stimulus dynamics concurrently.

### Resonance theory of consciousness

(2019). "The Easy Part of the Hard Problem: A Resonance Theory of Consciousness." *Frontiers in Human Neuroscience*, 13, 378. — Shared resonance allows different brain regions to achieve a phase transition in information flow. The combination problem of consciousness is solved by shared resonance: phase-locked oscillation enables unified experience.

### Spectral neuro-symbolic reasoning

Kiruluta, A., Burity, P. (2025). "From Eigenmodes to Proofs: Integrating Graph Spectral Operators with Symbolic Interpretable Reasoning." *arXiv:2509.07017*. — Spectral NSR performs reasoning directly in the graph spectral domain, embedding logical rules as spectral templates. Outperforms transformers on reasoning benchmarks. Reasoning can be done in the frequency domain.

### Wavelet scattering preserves phase across scales

Bruna, J., Mallat, S. (2013). "Invariant Scattering Convolution Networks." *IEEE TPAMI*. — Wavelet scattering networks cascade wavelet convolutions with modulus nonlinearity for translation-invariant, deformation-stable features. Incorporates higher-order moments that capture phase relationships. Discriminates textures with the same Fourier power spectrum.

### Complex-valued networks preserve phase

Trabelsi, C. et al. (2018). "Deep Complex Networks." *ICLR 2018*. — Complex-valued neural networks with complex convolutions, batch normalization, and weight initialization. Complex numbers provide richer representational capacity and preserve phase as a first-class citizen.

### Fourier neural operator learns in spectral domain

Li, Z. et al. (2020). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*. — Parameterizes integral kernels in Fourier space. Resolution-invariant learning. 1000x faster than traditional PDE solvers. Demonstrates that frequency-domain learning can be both efficient and generalizable.

### Graph wavelets extend multi-scale analysis to graphs

Hammond, D.K., Vandergheynst, P., Gribonval, R. (2011). "Wavelets on graphs via spectral graph theory." *Applied and Computational Harmonic Analysis*, 30(2), 129-150. — Constructs wavelet transforms on arbitrary weighted graphs via Laplacian spectral decomposition. Extends multi-scale analysis to non-Euclidean data.

### Structural general intelligence

(2025). "Structural General Intelligence (SGI): A System-Level Framework for the Shift from Scaling to Coherence." — Intelligence arises from coherence rather than magnitude. The interaction of heterogeneous subsystems converges toward a structural fixed point. Directly aligns with the Bifrost thesis.

### Full literature survey

See [dev-docs/12_LITERATURE_SURVEY_EXTERNAL.md](./dev-docs/12_LITERATURE_SURVEY_EXTERNAL.md) for the complete survey with 30+ verified citations across phase congruency, wavelet coherence, spectral neuro-symbolic reasoning, neural oscillations, cross-modal integration, consciousness theory, topological data analysis, and AGI frameworks.

---

## The framework

Bifrost processes all inputs through a phase-preserving spectral pipeline:

```
Input  (audio / image / text / sensor / any continuous signal)
  │
  ▼  S0  Canonicalization
  │      FFT → complex spectrum z = A·exp(iφ)
  │      SpectralTensor(amplitude, phase, scale, uncertainty)
  │
  ▼  S1  Complex SSM
  │      h[t] = exp(−Δ·A)·h[t−1] + Δ·B[t]·x[t]   (A complex diagonal)
  │      Parallel Blelloch associative scan — O(log L) depth
  │      Phase preserved through complex state transitions
  │
  ▼  S2  Spectral Binding
  │      C(i,j) = Σ_b w_b · mean_f[cos(φ_q[i,f] − φ_k[j,f])]
  │      Attention over phase alignment across frequency bands
  │      Harmonic overtone routing: energy at f, 2f, 3f, …
  │
  ▼  S3  Phase-Lock Bridge
  │      Stable frequency attractors via Adler equation dynamics
  │      VQ-VAE codebook (65K entries) over attractor space
  │      Cross-modal transfer via phase-locked attractors
  │
  ▼  S4  Riemannian Coherence (optional)
         Learned metric tensor G = LLᵀ over attractor embeddings
         Geodesic distances as semantic coherence scores
```

### Multi-Scale Structural Coherence (MSC)

The MSC framework is the operationalization of the thesis. Each modality has a specific MSC instance that measures phase coherence across scales:

| Modality | MSC instance | What it measures | Reference |
|---|---|---|---|
| Audio | CBMPC (Cross-Band Modulation Phase Coherence) | Phase locking of temporal modulations across mel frequency bands | Validated: +13.65 pp on SpeechCommands (p = 0.0033) |
| Image | Phase Congruency | Phase alignment across log-Gabor spatial frequency scales | Kovesi (1999) |
| Sensor | Wavelet Coherence | Cross-channel wavelet coherence at multiple time scales | Grinsted et al. (2004) |
| Text | Graph Spectral Coherence | Cross-role phase locking in dependency parse tree via graph wavelets | Hammond et al. (2011) |

All instances produce coherence features that can be projected to a unified coherence space for cross-modal comparison.

---

## Current implementation status

| Component | Status | File |
|---|---|---|
| SpectralTensor (amplitude, phase, scale, uncertainty) | Complete | `src/bifrost/spectral_tensor.py` |
| SpectralCanonicalizer (1D/2D FFT) | Complete | `src/bifrost/canonicalizer/` |
| ComplexSpectralDecomposer (complex SSM + Blelloch scan) | Complete | `src/bifrost/decomposer/complex_decomposer.py` |
| CBMPC (audio cross-band phase coherence) | Complete + validated | `src/bifrost/cbmpc.py` |
| PhaseCongruencyExtractor (image MSC) | Complete, validation pending | `src/bifrost/msc_image.py` |
| ResonanceAttention (phase-based attention) | Complete | `src/bifrost/resonance_attention/` |
| HarmonicBinding (overtone routing) | Complete | `src/bifrost/resonance_attention/harmonic_binding.py` |
| PhaseLockBridge (Adler equation cross-modal) | Complete | `src/bifrost/phase_lock_bridge/` |
| RiemannianMetricLearner + GeodesicComputer | Complete | `src/bifrost/riemannian_coherence/` |
| MultiModalSpectralPipeline (audio/image/text/tensor) | Complete | `src/bifrost/multimodal_pipeline.py` |
| Complex training (loss, optimizer, distributed) | Complete | `src/bifrost/complex_training.py` |
| WaveletCoherenceExtractor (sensor MSC) | To build | — |
| CrossScaleCoherence module | To build | — |
| Phase ablation harness | To build | — |
| Unified coherence metric | To build | — |

---

## The proof plan

The thesis is tested in five steps, each building on the one below. Steps 1-3 are the minimal viable proof.

### Step 1: Prove phase coherence captures semantic structure (C1)

Build a phase ablation harness that selectively destroys phase at any pipeline layer. Run on SpeechCommands (audio) and CIFAR-10 (image). If phase-coherent features outperform phase-ablated features (p < 0.05), phase coherence captures semantic structure.

### Step 2: Prove multi-scale coherence is necessary (C2)

Build a cross-scale coherence module that computes PLV between wavelet scales. Run scale ablations: single-scale vs multi-scale vs cross-scale-destroyed. If cross-scale coherence outperforms single-scale and cross-scale-destroyed (p < 0.05), multi-scale coherence is necessary.

### Step 3: Prove cross-modal generalization (C3)

Implement wavelet coherence for sensors. Validate all three MSC instances (audio, image, sensor). Build a unified coherence metric. Test cross-modal transfer: can audio coherence features classify image samples? If yes (above chance, p < 0.05), the principle generalizes.

### Step 4: Prove cross-modal alignment (C4) — future

Build cross-modal phase binding. Test cross-modal retrieval: given an audio sample, retrieve the image with the most similar coherence pattern.

### Step 5: Prove compositional generalization (C5) — future

Build compositional structure extractor. Test compositional generalization: train on compositions A/B/C, test on novel D/E/F.

**Full plan**: [dev-docs/14_REFINED_ENGINEERING_PLAN_STEPS_1_3.md](./dev-docs/14_REFINED_ENGINEERING_PLAN_STEPS_1_3.md)

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
# Multi-modal
from bifrost.multimodal_pipeline import create_multimodal_pipeline, Modality

pipe = create_multimodal_pipeline(d_model=256)
output = pipe(signal, modality=Modality.AUDIO)
```

```python
# CBMPC feature extraction (audio MSC)
from bifrost.cbmpc import CBMPCExtractor

extractor = CBMPCExtractor()
features = extractor(signal, sample_rate=16000)  # cross-band phase coherence features
```

```python
# Phase congruency (image MSC)
from bifrost.msc_image import PhaseCongruencyExtractor

extractor = PhaseCongruencyExtractor(n_scales=5, n_orientations=6)
features = extractor(image)  # phase congruency across spatial frequency scales
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
├── spectral_tensor.py          # SpectralTensor (amplitude, phase, scale, uncertainty)
├── pipeline.py                 # BifrostPipeline
├── multimodal_pipeline.py      # Per-modality routing
├── cbmpc.py                    # Audio MSC: cross-band modulation phase coherence
├── msc_image.py                # Image MSC: phase congruency
├── canonicalizer/              # S0: signal → SpectralTensor
├── decomposer/                 # S1: complex SSM, associative scan, wavelet bank
├── resonance_attention/        # S2: phase coherence attention, harmonic binding
├── phase_lock_bridge/          # S3: Adler equation, attractor extraction
├── s3_attractor/               # S3: VQ-VAE attractor learning
├── riemannian_coherence/       # S4: Riemannian metric, geodesics
├── semantic_coherence/         # Training objectives
├── validation/                 # Empirical validation suite
├── ingest/                     # Raw media ingestion
├── training.py                 # BifrostTrainer, contrastive phase loss
├── complex_training.py         # Complex-valued training
├── distributed_training.py     # DDP multi-GPU/multi-node
└── checkpoint_manager.py       # Versioned checkpointing

dev-docs/                       # Research documentation
├── RESEARCH_DOSSIER.md         # Top-level synthesis and navigation
├── 01_EPISTEMIC_AUDIT_SUMMARY.md
├── 02_CBMPC_TECHNIQUE_OVERVIEW.md
├── 03_CBMPC_PRE_SSM_INTEGRATION_PLAN.md
├── 04_MODULATION_PRESERVING_SSM_INVESTIGATION.md
├── 05_ESC50_GENERALIZATION_TEST_PLAN.md
├── 06_MSC_FRAMEWORK.md
├── 07_MSC_MODALITY_INSTANCES.md
├── 08_CROSS_MODAL_VALIDATION_PROTOCOL.md
├── 09_RESEARCH_PATHS_COMPENDIUM.md
├── 10_FREQUENCY_LEVEL_DATA_MODELS.md
├── 11_AGI_ASI_STRUCTURAL_INTELLIGENCE.md
├── 12_LITERATURE_SURVEY_EXTERNAL.md
├── 13_ENGINEERING_REQUIREMENTS.md
└── 14_REFINED_ENGINEERING_PLAN_STEPS_1_3.md

research_dir/                   # Experiment scripts
├── experiment_cbmpc_comparison.py
├── experiment_cbmpc_esc50.py
├── experiment_cbmpc_pre_ssm.py
├── experiment_msc_image_cifar10.py
└── results/                    # Experiment outputs
```

---

## Key references

| Reference | Contribution |
|---|---|
| Kovesi (1999) | Phase congruency for image features |
| Grinsted et al. (2004) | Wavelet coherence for cross-scale phase |
| Rahaman et al. (2019) | Spectral bias of neural networks |
| Trabelsi et al. (2018) | Deep complex networks |
| Li et al. (2020) | Fourier neural operator |
| Hammond et al. (2011) | Graph wavelets via spectral graph theory |
| Bruna & Mallat (2013) | Wavelet scattering transform |
| Kiruluta & Burity (2025) | Spectral neuro-symbolic reasoning |
| Frontiers Hum Neurosci (2019) | Resonance theory of consciousness |
| J Neurosci (2015) | Phase alignment drives multisensory integration |
| Sci Rep (2019) | Theta oscillations in semantic retrieval |
| Nat Hum Behav (2024) | Synchronous oscillations bind cortical modules |
| Gu & Dao (2023) | Mamba: selective state spaces |
| Adler (1946) | Oscillator locking phenomena |

**Full survey**: [dev-docs/12_LITERATURE_SURVEY_EXTERNAL.md](./dev-docs/12_LITERATURE_SURVEY_EXTERNAL.md)

---

## License

MIT. See [LICENSE](./LICENSE).

## Contact

**Quantumind** — engineering@quantumind.io
