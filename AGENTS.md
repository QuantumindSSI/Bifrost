# Bifrost Agent Guide

**Project**: Bifrost — Spectral neural processing with phase-coherent representations
**Current state**: Phase 1 foundation complete (≈75%). Core S0→S4 pipeline implemented.
**Source of truth**: `ENGINEERING_PLAN.md` and `PHASE_1_STATUS.md`

---

## What this project is

Bifrost learns representations of meaning from the **structure of continuous signals** (audio, image, text-as-signal, sensors) rather than from token statistics. It converts every input to a complex spectrum `z = A·exp(iφ)` and routes information via **phase coherence** — the physical synchronization of oscillatory phases.

---

## What is implemented vs. what is missing

| Layer | Component | Status | Location |
|---|---|---|---|
| L1 Distributional | SpectralBinding, ResonanceAttention | Complete | `src/bifrost/resonance_attention/` |
| L2 Compositional | Hierarchical Multi-Timescale SSM | **Planned** | not yet created |
| L3 Causal | Granger Causal Graph | **Planned** | not yet created |
| L4 Topological | TDA Persistence Diagrams | **Planned** | `src/bifrost/riemannian_coherence/` (geometry only) |
| L5 Temporal | Allen Interval Algebra | **Planned** | not yet created |
| L6 Symmetry | SymmetryTensor / Adaptive Harmonic Binding | **Planned** | not yet created |
| L7 Disentanglement | DisentangledTensor / TC-VAE | **Planned** | not yet created |

Infrastructure (distributed training, checkpointing, evaluation, multi-modal curation) is complete.

---

## Atomic implementation plan for the missing layers

Work in this order. Each layer has prerequisites, deliverables, and a success criterion.

### 0. Prerequisite: PredictiveErrorTensor (3 days)

This is not a layer by itself, but it is required for Layer 3 and is a free byproduct of the existing SSM.

- [ ] Implement `src/bifrost/decomposer/predictive_error.py`
  - `PredictiveErrorTensor` dataclass
  - `PredictiveErrorExtractor` hooked into `ComplexSpectralDecomposer`
- [ ] Add optional `return_error=True` to `ComplexSpectralDecomposer.forward()`
- [ ] Compute `y_pred[t] = C·h[t-1]` and `e[t] = y[t] - y_pred[t]`
- [ ] Compute `surprise_rate[t]` and `semantic_boundary_score[t]`
- [ ] Write `tests/test_predictive_error.py`
- [ ] Benchmark: zero-shot boundary F1 > 0.6 on LibriSpeech forced alignment

### 1. Layer 3 — Granger Causal Graph (1 week, fast mode)

**Purpose**: Add the first directed, asymmetric representation to Bifrost.

- [ ] Create `src/bifrost/causal_graph/` package
- [ ] Implement `CausalGraphTensor` dataclass in `granger_causality.py`
- [ ] Implement `GrangerCausalityExtractor` fast mode
  - Cross-covariance of SSM `B[t]` matrices across bands and lags
  - Asymmetric by construction
- [ ] Implement `CausalAttentionHead` in `causal_attention.py`
  - Score(q_i, k_j) = `GC(i→j)`
- [ ] Wire optional causal return into `BifrostPipeline`
- [ ] Write `tests/test_granger_causality.py`
- [ ] Verify: `GC(i→j) != GC(j→i)` for >60% of band pairs

**Later (Phase 2 exact mode)**: VAR(p) model on SSM hidden states + F-test significance.

### 2. Layer 4 — TDA Persistence (1 week)

**Purpose**: Capture global topology (connected components, loops, voids) of the spectral amplitude landscape.

- [ ] Create `src/bifrost/topology/` package
- [ ] Implement `PersistenceTensor` dataclass in `persistence_tda.py`
- [ ] Implement `TDAPersistenceExtractor`
  - Convert `(B, T, F)` amplitude to 3D point cloud `(t, f, amplitude)`
  - Use `ripser` or `gudhi` for Vietoris-Rips persistence
  - Extract Betti numbers `[β₀, β₁, β₂]`
- [ ] Implement differentiable Wasserstein wrapper in `differentiable_tda.py`
- [ ] Add optional dependency `ripser>=0.6.0` or `gudhi>=3.8.0`
- [ ] Wire `return_persistence=True` into `BifrostPipeline`
- [ ] Write `tests/test_persistence_tda.py`
- [ ] Benchmark: 5 instrument families discriminated by Betti numbers at >80%

### 3. Layer 2 — Hierarchical Multi-Timescale SSM (3 weeks)

**Purpose**: Explicit compositional part-whole structure (phoneme → syllable → word → phrase → discourse).

- [ ] Implement `src/bifrost/decomposer/hierarchical_ssm.py`
  - `HierarchicalSSMConfig`: 5 levels, `timescales_ms=[10, 100, 500, 2000, 10000]`
  - `HierarchicalComplexSSM`: pyramid of `ComplexSpectralDecomposer`s
  - Strided pooling between levels; upsampling for fusion
- [ ] Implement cross-level fusion attention
  - Concatenate per-level outputs and learn attention weights per frame
- [ ] Add `hierarchical=True` mode to `BifrostPipeline`
- [ ] Return `hierarchical_states` list in `BifrostOutput`
- [ ] Write `tests/test_hierarchical_ssm.py`
- [ ] Benchmark: word-boundary F1 > flat SSM baseline on Switchboard

### 4. Layer 6 — SymmetryTensor (2 weeks)

**Purpose**: Detect invariance groups from the signal instead of hardcoding octave invariance.

- [ ] Create `src/bifrost/symmetry/` package
- [ ] Implement `SymmetryTensor` dataclass in `symmetry_detection.py`
- [ ] Implement `SymmetryDetector`
  - Temporal autocorrelation for periodicity
  - Log-frequency autocorrelation for frequency ratios
  - Time-stretch correlation for tempo invariance
  - Angular FFT for image rotational/reflection symmetry
- [ ] Implement `SymmetryGroupEncoder` classifying cyclic/dihedral/trivial/continuous
- [ ] Implement `SymmetryAdaptiveBinding` in `adaptive_harmonic_binding.py`
  - Replace fixed `f, 2f, 3f` grid with detected ratios `r_k·f`
- [ ] Add `adaptive_harmonics=True` to `BifrostPipeline`
- [ ] Write `tests/test_symmetry_detection.py`
- [ ] Benchmark: cyclic vs. dihedral vs. trivial classification >85%

### 5. Layer 7 — DisentangledTensor (4 weeks)

**Purpose**: Factorize representation into statistically independent content, style, and temporal factors.

- [ ] Create `src/bifrost/disentanglement/` package
- [ ] Implement `DisentangledTensor` dataclass in `disentangled_vae.py`
- [ ] Implement `TCVAEEncoder`
  - Outputs: `content_factors`, `style_factors`, `temporal_factors`
  - Loss: reconstruction + β·TC(z) + MI + per-factor KL
- [ ] Implement `MutualInformationEstimator` using MINE in `mi_estimator.py`
- [ ] Add adversarial content-style invariance training
- [ ] Implement decoder for cross-style generation
- [ ] Add `disentangle=True` to `BifrostPipeline`
- [ ] Write `tests/test_disentangled_vae.py`
- [ ] Benchmark: TC score < 1.0; DCI score on dSprites; zero-shot speaker transfer

### 6. Layer 5 — TemporalRelationTensor (2 weeks)

**Purpose**: Explicit qualitative temporal/narrative relations over attractor activations.

- [ ] Create `src/bifrost/temporal/` package
- [ ] Implement `TemporalRelationTensor` dataclass in `allen_algebra.py`
- [ ] Implement `AllenRelationExtractor`
  - Segment attractor activations into intervals
  - Classify 13 Allen relations with tolerance `ε`
- [ ] Implement soft `RelationNet` classifier
- [ ] Implement transitivity closure and narrative inference in `narrative_inference.py`
- [ ] Apply to S3 attractor outputs in `BifrostPipeline`
- [ ] Write `tests/test_allen_algebra.py`
- [ ] Benchmark: 8/13 relations correct >80% on synthetic intervals; narrative recovery on annotated stories

### 7. SevenLayerSemanticScore (1 week)

**Purpose**: Composite evaluation once all layers exist.

- [ ] Implement `SevenLayerSemanticScore` dataclass in `src/bifrost/evaluation.py`
- [ ] Add `compute_seven_layer_score(model, eval_dataset)`
- [ ] Define per-layer metrics and datasets
- [ ] Integrate into validation loop and `BifrostOutput`
- [ ] Write tests and report generation

---

## How to move from token-level to structural-level reasoning

### Why token-level is a bottleneck

Current large language models (and most multimodal models) operate on discrete tokens. This has four fundamental costs:

1. **Lossy front-end**: Audio, image, and sensor data are tokenized into approximations that discard phase, fine time structure, and continuous dynamics.
2. **Context-window fragility**: Token sequences are bounded; long-range structure is compressed or lost.
3. **Sequential bias**: Tokens force reasoning into a linear order, even when the underlying structure is parallel, hierarchical, or cyclic.
4. **Reward-hacking surface**: Learned verifiers can be gamed because they operate on the same token space as the generator.

Bifrost is designed to bypass this dependency by operating on **continuous, structural representations** and only optionally interfacing with tokens at the end.

### How Bifrost already reduces token dependence

| Stage | Token-level alternative | Bifrost structural alternative |
|---|---|---|
| Input | Text tokens / audio mel-tokens / image patch-tokens | `SpectralTensor`: complex spectrum `A·exp(iφ)` |
| Sequence model | Transformer over tokens | Complex SSM over spectra; stateful, sub-linear depth |
| Similarity | Dot-product of token embeddings | Phase coherence across frequency bands |
| Binding | Cross-attention over tokens | Harmonic / spectral binding via phase alignment |
| Memory | KV cache of tokens | Attractor codebook over frequency attractors |
| Reasoning | Next-token prediction | Riemannian geodesics on attractor manifold |
| Verification | Learned reward model | Structural coherence grounded in phase-lock physics |

### The three LLM integration modes

Bifrost does not replace LLMs; it reduces their monopoly on representation. The three modes are already implemented in `src/bifrost/llm_adapter.py`:

**Mode 1: Spectral prefix**
- Bifrost encodes audio, image, or sensor into continuous spectral embeddings.
- These are projected as prefix tokens prepended to text: `[spectral prefix | text tokens]`.
- The LLM is frozen; it receives grounded multimodal context without retraining.
- Token dependence is reduced because the prefix carries continuous signal structure, not tokenized surrogates.

**Mode 2: Parameter-efficient adapter**
- The complex SSM is injected between frozen LLM layers (~5M trainable parameters, ~4% of GPT-2).
- It provides long-context phase coherence tracking and per-token uncertainty beyond the LLM's context window.
- The LLM still emits tokens, but its internal states are shaped by spectral recurrence, not just self-attention over tokens.

**Mode 3: Structural coherence verifier**
- For chain-of-thought reasoning, encode both the problem and each reasoning step through Bifrost.
- Compute: phase-lock score, causal alignment, topological distance, and prediction-error spike.
- Flag or reject structurally incoherent steps.
- This is a verifier grounded in physics (Adler equation, phase coherence, prediction error), not a learned reward model. It is much harder to game by reward hacking because the signal is not in token space.

### How each new layer further reduces token dependence

| Layer | Token-dependency it removes | How |
|---|---|---|
| **L2 Compositional** | Linear token order | Hierarchical SSM captures nested part-whole structure independent of token sequence |
| **L3 Causal** | Correlational reasoning | Granger graph provides directed influence; answers intervention/counterfactual questions without sampling tokens |
| **L4 Topological** | Embedding-space similarity | Betti numbers detect global shape (loops, voids) that dot-product cannot see |
| **L5 Temporal** | Position-based ordering | Allen relations represent event structure without relying on token index positions |
| **L6 Symmetry** | Vocabulary-specific invariance | Detects invariances from raw signal; works for speech, images, and sensors without handcoded token rules |
| **L7 Disentanglement** | Entangled token representations | Separates content/style/temporal factors, enabling controlled manipulation without re-generating token sequences |

### Practical migration path

To reduce token dependence in a real system:

1. **Stop tokenizing multimodal inputs early.** Feed audio/image/sensor directly into Bifrost S0 instead of mel-spectrogram tokens or patch tokens.
2. **Use phase coherence as the primary similarity metric.** Replace token cross-attention with `ResonanceAttention` / `CausalAttentionHead` where the task is structural correspondence.
3. **Use attractors as structural memory.** Store and retrieve frequency attractors instead of token embeddings for long-context or cross-modal retrieval.
4. **Use the structural verifier as a guardrail.** Run Bifrost over LLM chain-of-thought steps and reject low-coherence continuations before emitting tokens.
5. **Train objectives on continuous signals.** Use contrastive phase loss, TDA Wasserstein distance, and TC-VAE losses rather than next-token prediction.
6. **Represent downstream tasks on the manifold.** Treat analogy completion as geodesic interpolation, classification as attractor proximity, and generation as controlled traversal of disentangled factors.

---

## Verification checklist before considering each layer complete

- [ ] Code is located in the expected `src/bifrost/<layer>/` package
- [ ] Unit tests pass (`pytest tests/test_<layer>.py -v`)
- [ ] Dataclass returns meaningful tensors on a synthetic forward pass
- [ ] Integration path into `BifrostPipeline` is documented and working
- [ ] Benchmark metric meets the success criterion above
- [ ] No new dependencies are added without updating `pyproject.toml`
- [ ] `AGENTS.md` is updated if layer-specific conventions change

---

## Environment notes

- `python3` is available (Python 3.9.6), but the project is not installed in this workspace.
- `pytest` and `poetry` are not installed.
- To run anything, first run: `pip install -e ".[dev]"`

---

# Research reference: frequency-level techniques for semantic structure

Phase coherence is the central mechanism in Bifrost, but it is not the only way to extract structure from continuous spectra. The following techniques are complementary and can be added as additional feature channels on top of `SpectralTensor`.

The key principle is that Bifrost stays in **continuous spectral space** through the pipeline, so any frequency-level feature can be concatenated to the amplitude/phase/uncertainty tensors without breaking the architecture.

---

## 1. Spectral envelope and shape features

| Technique | What it captures | Bifrost layer | Use case |
|---|---|---|---|
| **MFCC / PLP** | Mel-warped spectral envelope; compact and robust | L1, L7 | Timbre/speaker/style factor; attractor initialization |
| **Spectral centroid** | Brightness / "where energy lives" | L1 | Timbre, texture classification |
| **Spectral rolloff / bandwidth** | Frequency range of energy | L1 | Distinguish bright vs. dull sounds |
| **Spectral contrast** | Difference between peaks and valleys | L1 | Speech vs. music vs. noise discrimination |
| **Spectral flatness** | Tonal vs. noisy | L1 | Segment harmonic and non-harmonic regions |

**Why semantic:** Vowels, instruments, and noise classes have characteristic envelope shapes. These are low-level but semantically discriminative.

---

## 2. Harmonic and periodic structure features

| Technique | What it captures | Bifrost layer | Use case |
|---|---|---|---|
| **Fundamental frequency (F0)** | Periodic repetition rate | L2, L6 | Bind harmonics to a single source; pitch invariance |
| **Harmonic-to-noise ratio (HNR)** | Tonality vs. noisiness | L1, L6 | Voice quality; voiced vs. unvoiced speech |
| **Inharmonicity** | Deviation from integer harmonic ratios | L6 | Distinguish piano, bells, drums, speech |
| **Log-frequency autocorrelation** | Repeating frequency ratios | L6 | Detect octave, fifth, and sub-octave invariance |
| **Subharmonic summation** | Pitch from sub-harmonics | L2 | Low-frequency pitch recovery |

**Why semantic:** Harmonicity is a core organizing principle. A chord is a set of harmonically related frequencies; speech voicing is a periodic pulse. Detecting these gives primitive objects to compose.

---

## 3. Time-frequency dynamics features

| Technique | What it captures | Bifrost layer | Use case |
|---|---|---|---|
| **Modulation spectrogram** | Slow amplitude modulations across bands | L2, L5 | Speech rhythm, syllable rate, musical meter |
| **Spectral flux / onset envelope** | Rate of spectral change | L2, L5 | Event boundaries for hierarchical and Allen-interval layers |
| **Wavelet scattering transform** | Stable, shift-invariant time-frequency representation | L1, L4 | Robust texture and timbre features |
| **Cochleagram / auditory spectrogram** | Biologically inspired tiling | L1 | Alternative or augmented S0 input |
| **Constant-Q transform (CQT)** | Log-frequency resolution | L1, L6 | Music and tonal analysis |
| **Chroma features** | Pitch-class energy independent of octave | L6 | Harmonic structure collapsed to 12 pitch classes |

**Why semantic:** Language is not a bag of spectral frames; it is structured in time. Modulation spectrograms capture syllable-level rhythm, and spectral flux can detect phoneme and word boundaries.

---

## 4. Cross-frequency coupling and higher-order spectra

| Technique | What it captures | Bifrost layer | Use case |
|---|---|---|---|
| **Cross-frequency coupling (PAC, CFC)** | Phase-amplitude coupling between bands | L3 | Directed influence beyond Granger causality |
| **Bispectrum / bicoherence** | Phase-locked triplets of frequencies | L3, L4 | Nonlinear harmonic interactions; early event detection |
| **Cross-spectral density** | Correlation per frequency band | L1, L3 | Cross-modal coherence and source separation |
| **Magnitude-squared coherence (MSC)** | Stable band-wise correlation | L1 | Cross-signal similarity per band |

**Why semantic:** A word is not just a set of frequencies; it is a coordinated pattern across frequencies. Cross-frequency coupling can reveal which bands drive others, which is richer than pairwise phase coherence.

---

## 5. Decomposition and factor separation

| Technique | What it captures | Bifrost layer | Use case |
|---|---|---|---|
| **Non-negative matrix factorization (NMF)** | Spectral bases + temporal activations | L7 | Source separation; phoneme/note dictionaries |
| **Sparse coding of spectrograms** | Sparse set of spectral atoms | L2, L7 | Discovery of recurrent acoustic events |
| **Source-filter model** | Excitation vs. vocal-tract filter | L7 | Speech content vs. speaker/style separation |
| **Harmonic-percussive separation** | Harmonic vs. transient components | L2, L7 | Texture/structure decomposition |
| **Independent Component Analysis (ICA)** | Statistically independent sources | L7 | Blind source separation |

**Why semantic:** These methods explicitly find parts of a scene. NMF bases often correspond to notes, phonemes, or noise sources. Source-filter separation is essentially a disentanglement of content from style.

---

## 6. Topological and geometric features

| Technique | What it captures | Bifrost layer | Use case |
|---|---|---|---|
| **Persistent homology on spectrograms** | Loops, voids, connected components | L4 | TDA PersistenceTensor |
| **Spectral clustering on frequency bins** | Groups of co-varying bins | L1, L4 | Attractor discovery |
| **Manifold learning on spectral embeddings** | Low-dimensional geometry | L4 | Riemannian metric initialization |
| **UMAP / t-SNE on spectral windows** | Neighborhood structure | L4 | Visualization and attractor validation |

**Why semantic:** A chord and a melody can have the same notes but different topological organization. TDA captures that global organization.

---

## 7. Self-supervised and learned frequency representations

| Technique | What it captures | Bifrost layer | Use case |
|---|---|---|---|
| **Spectral autoencoder / VAE** | Compressed frequency representation | L1, L7 | Pretraining attractor space |
| **Contrastive learning on spectrograms** | Invariant representations | L1 | Augment or replace phase-coherence contrastive loss |
| **Masked frequency modeling** | Predict masked frequency bins | L1 | Frequency-domain pretraining objective |
| **Fourier neural operator (FNO)** | Frequency-domain neural operator | L1 | Direct learning in Fourier space |
| **Spectral transformer** | Attention over frequency bins | L1 | Alternative to ResonanceAttention |

**Why semantic:** Learning can discover structure that hand-crafted features miss. A contrastive loss in frequency space can learn representations where semantically similar sounds are close without ever tokenizing.

---

## How to integrate these into Bifrost

Because Bifrost operates on `SpectralTensor`, the cleanest integration is to augment the tensor with additional feature channels:

```
SpectralTensor(
    amplitude:      (B, T, F)
    phase:          (B, T, F)
    uncertainty:    (B, T, F)
    envelope:       (B, T, n_mfcc)        # spectral envelope features
    harmonic:       (B, T, n_harmonic)     # F0, HNR, inharmonicity
    modulation:     (B, T, n_mod)         # modulation spectrogram
    coupling:       (B, T, n_bands²)      # cross-frequency coupling
    topology:       (B, T, n_betti)       # Betti numbers per frame
)
```

Then the SSM and binding layers can consume these features alongside phase. This is a portfolio approach: phase coherence remains the primary signal, but it is enriched by envelope, harmonic, dynamic, coupling, and topological features.

---

## Layer-specific recommendations

| Layer | Recommended frequency techniques to add |
|---|---|
| **L2 Hierarchical SSM** | Spectral flux, modulation spectrogram, onset envelope as boundary supervision |
| **L3 Granger Causality** | Cross-frequency coupling (PAC/CFC) and bispectrum alongside Granger edges |
| **L4 TDA** | Persistent homology of the spectral amplitude surface; landmark complex for speed |
| **L5 Temporal Relations** | Spectral flux + onset envelope → interval boundaries for Allen algebra |
| **L6 Symmetry** | Log-frequency autocorrelation, harmonic series detection, chroma features |
| **L7 Disentanglement** | Source-filter decomposition, NMF bases, harmonic-percussive separation as inductive biases |

---

## Caution: none of these are automatically semantic

A spectral centroid, a Betti number, or a cross-frequency coupling value is just a number. It only becomes "semantic" when it correlates with human-annotated categories or downstream task performance. The same discipline must apply here as with phase coherence: validate every feature on real data and a real task before treating it as a semantic signal.
