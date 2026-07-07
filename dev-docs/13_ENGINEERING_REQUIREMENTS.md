# 13 — Engineering Requirements to Prove the Bifrost Thesis

**Thesis**: Intelligence is structured resonance, and phase-coherent multi-scale representations capture semantic structure across modalities.

**Status**: Engineering requirements analysis  
**Purpose**: Define exactly what engineering is required to prove (or falsify) this thesis, given what Bifrost already has and what is missing.

---

## The thesis has five load-bearing claims

To prove the thesis, you must demonstrate each claim empirically. Each claim demands a specific type of engineering that conventional ML frameworks do not provide.

| Claim | What you must show | Engineering required |
|---|---|---|
| C1: Phase coherence captures semantic structure | Phase-coherent features > phase-invariant features on semantic tasks | Phase-preservation infrastructure + ablation framework |
| C2: Multi-scale is necessary | Multi-scale coherence > single-scale coherence | Cross-scale coherence aggregation |
| C3: This generalizes across modalities | Same principle works on audio, image, sensor, text | Modality-specific MSC instances + unified coherence metric |
| C4: Cross-modal alignment is possible | Phase coherence patterns align across modalities | Cross-modal phase binding mechanism |
| C5: This enables AGI-level generalization | Systematic compositional generalization from phase structure | Compositional structure extraction + generalization benchmarks |

---

## What Bifrost already has

The codebase audit reveals substantial existing infrastructure:

| Component | Status | File |
|---|---|---|
| SpectralTensor (amplitude, phase, scale, uncertainty) | Complete | `src/bifrost/spectral_tensor.py` |
| Complex-valued SSM (ComplexSelectiveScan) | Complete | `src/bifrost/decomposer/complex_decomposer.py` |
| CBMPC (audio cross-band phase coherence) | Complete | `src/bifrost/cbmpc.py` |
| PhaseCongruencyExtractor (image MSC) | Complete | `src/bifrost/msc_image.py` |
| ResonanceAttention (phase-based attention) | Complete | `src/bifrost/resonance_attention/attention.py` |
| HarmonicBinding | Complete | `src/bifrost/resonance_attention/harmonic_binding.py` |
| PhaseLockBridge | Complete | `src/bifrost/phase_lock_bridge/bridge.py` |
| RiemannianMetricLearner + GeodesicComputer | Complete | `src/bifrost/riemannian_coherence/` |
| MultiModalSpectralPipeline | Complete | `src/bifrost/multimodal_pipeline.py` |
| Complex training (loss, optimizer, distributed) | Complete | `src/bifrost/complex_training.py` |
| LearnableWaveletBank (multi-scale) | Complete | `src/bifrost/decomposer/decomposer.py` |
| Associative scan (O(log n) parallel) | Complete | `src/bifrost/decomposer/associative_scan.py` |

**What works**: The raw machinery for phase-preserving spectral processing exists. Complex-valued SSMs, phase extraction, resonance attention, and modality-specific MSC extractors are all implemented.

**What is missing**: The engineering to *prove the thesis* — the ablation frameworks, cross-scale aggregation, cross-modal binding, compositional structure extraction, and generalization benchmarks.

---

## The five engineering requirements

### Requirement 1: Phase-preservation engineering (proves C1)

**Goal**: Prove that phase coherence captures semantic structure that phase-invariant representations cannot.

**The problem**: Currently, Bifrost has both real-valued and complex-valued decomposers, but there is no systematic ablation framework to compare them. You cannot prove "phase matters" by showing that one architecture happens to work better than another — you need controlled ablations that toggle phase preservation at every layer.

**What needs to be built**:

#### 1a. Phase ablation harness

A testing framework that can selectively destroy phase information at any point in the pipeline and measure the effect on semantic task performance.

```
PhaseAblationHarness
├── phase_randomization (shuffle phase across time)
├── phase_zeroing (set phase to 0 — keep amplitude only)
├── phase_quantization (reduce phase precision)
├── phase_noise (add Gaussian noise to phase)
└── phase_locking_break (perturb cross-band phase relationships)
```

Each ablation runs the full pipeline and measures:
- Classification accuracy delta
- Embedding space structure (silhouette score, cluster purity)
- Phase coherence metrics (PLV, phase entropy)

**The proof**: If phase-coherent features consistently outperform phase-ablated features across tasks and modalities, C1 is supported.

#### 1b. Phase coherence as a first-class metric

Currently, phase coherence is computed inside CBMPC but is not exposed as a general metric. Need:

```python
class PhaseCoherenceMetrics:
    def plv(self, phases_a, phases_b) -> float
    def phase_entropy(self, phases) -> float
    def phase_congruency(self, multi_scale_phases) -> float
    def cross_frequency_coupling(self, phases, freqs_a, freqs_b) -> float
    def phase_stability(self, phases_over_time) -> float
```

These metrics must be computed at every layer of the pipeline, not just the output.

#### 1c. Phase-aware contrastive loss

The current contrastive loss uses dot-product similarity. Need a loss that uses phase coherence as the similarity metric:

```python
class PhaseCoherenceContrastiveLoss:
    """Similar samples should have phase-coherent representations"""
    def forward(self, anchor, positive, negative):
        sim_pos = phase_coherence(anchor, positive)  # PLV-based
        sim_neg = phase_coherence(anchor, negative)
        return -log(sim_pos / (sim_pos + sim_neg))
```

**Existing code to leverage**: `SupervisedSemanticCoherenceLoss` in `semantic_coherence/core.py` already does contrastive learning on coherence features — extend it to use phase coherence directly.

---

### Requirement 2: Cross-scale coherence aggregation (proves C2)

**Goal**: Prove that coherence *across* scales captures more semantic structure than coherence *within* any single scale.

**The problem**: Bifrost's `LearnableWaveletBank` processes multiple scales in parallel, but there is no mechanism to compute coherence *between* scales. Each scale produces features independently. The MSC framework requires cross-scale phase coherence — phase alignment between scale s1 and scale s2 — but this is not computed.

**What needs to be built**:

#### 2a. Cross-scale phase coherence module

```python
class CrossScaleCoherence(nn.Module):
    """Compute phase coherence between different scales of analysis"""
    def forward(self, multi_scale_phases: List[Tensor]) -> Tensor:
        # For each pair of scales (s_i, s_j):
        #   Compute PLV between phase at s_i and phase at s_j
        #   Weight by scale ratio (dyadic: s_j / s_i = 2^k)
        # Aggregate into cross-scale coherence vector
```

This is the generalization of CBMPC from "cross-band" (across frequency bands at one scale) to "cross-scale" (across analysis scales). CBMPC measures phase locking between mel bands at modulation frequencies. Cross-scale coherence measures phase locking between wavelet scales.

#### 2b. Scale-ablation framework

```
ScaleAblationHarness
├── single_scale (use only one wavelet scale)
├── scale_subset (use k of n scales)
├── scale_randomization (shuffle scale assignments)
└── cross_scale_destroy (keep multi-scale but break cross-scale coherence)
```

**The proof**: If cross-scale coherence features outperform single-scale or scale-shuffled features, C2 is supported.

#### 2c. Hierarchical coherence aggregation

The MSC framework's key insight is that semantic structure exists at multiple scales simultaneously. Need a mechanism that aggregates coherence from fine to coarse scales:

```
Scale 1 (finest): features, edges, samples
    ↓ coherence aggregation
Scale 2: parts, phonemes, events
    ↓ coherence aggregation
Scale 3 (coarsest): objects, words, patterns
    ↓ coherence aggregation
Global coherence vector
```

This is not pooling — it is coherence *computation* at each level, then aggregation of coherence patterns.

---

### Requirement 3: Modality-specific MSC instances + unified metric (proves C3)

**Goal**: Prove that the same principle — phase-coherent multi-scale structure — captures semantic structure across audio, image, sensor, and text.

**The problem**: Bifrost has CBMPC (audio) and PhaseCongruencyExtractor (image), but they produce different types of features with different dimensions. There is no unified coherence metric that allows comparison across modalities. The sensor (wavelet coherence) and text (graph spectral coherence) instances are not implemented.

**What needs to be built**:

#### 3a. Wavelet coherence extractor (sensor modality)

```python
class WaveletCoherenceExtractor(nn.Module):
    """Sensor MSC: Cross-channel wavelet coherence at multiple time scales"""
    def forward(self, multi_channel_signal) -> Tensor:
        # For each pair of channels (c_i, c_j):
        #   Compute CWT of each channel
        #   Compute wavelet coherence R^2(a, t) = |S(s^-1 W_i W_j*)|^2 / (S|W_i|^2 * S|W_j|^2)
        #   Extract phase angle arctan(Im/Re)
        # Aggregate coherence across channel pairs and scales
```

**Reference**: Grinsted, Moore & Jevrejeva (2004). MATLAB toolbox available at https://github.com/grinsted/wavelet-coherence.

#### 3b. Graph spectral coherence extractor (text modality)

```python
class GraphSpectralCoherenceExtractor(nn.Module):
    """Text MSC: Cross-role phase locking in dependency parse tree"""
    def forward(self, token_embeddings, dependency_graph) -> Tensor:
        # 1. Build graph Laplacian L from dependency parse
        # 2. Compute graph Fourier transform: f_hat = U^T f
        # 3. Compute graph wavelets: ψ_t = g(tL) δ
        # 4. For each pair of syntactic roles (subject, verb, object):
        #    Compute phase coherence across graph wavelet scales
        # Aggregate into graph spectral coherence vector
```

**Reference**: Hammond, Vandergheynst & Gribonval (2011).

#### 3c. Unified coherence metric

All four MSC instances must produce features in a shared "coherence space" that allows cross-modal comparison:

```python
class UnifiedCoherenceMetric:
    """Maps modality-specific coherence to a shared coherence space"""
    def __init__(self, target_dim=256):
        self.audio_proj = nn.Linear(cbmpc_dim, target_dim)
        self.image_proj = nn.Linear(pc_dim, target_dim)
        self.sensor_proj = nn.Linear(wc_dim, target_dim)
        self.text_proj = nn.Linear(gsc_dim, target_dim)
    
    def forward(self, coherence_features, modality) -> Tensor:
        # Project to shared coherence space
        # Normalize to unit hypersphere
        return F.normalize(proj(coherence_features), dim=-1)
```

**The proof**: If the same coherence metric captures semantic structure across all four modalities (above-chance classification in each), C3 is supported.

---

### Requirement 4: Cross-modal phase binding (proves C4)

**Goal**: Prove that phase coherence patterns from different modalities can be aligned — that "rhythmic" audio and "rhythmic" video have more similar coherence profiles than "rhythmic" audio and "arrhythmic" video.

**The problem**: Bifrost's `PhaseLockBridge` is designed for cross-domain knowledge transfer (transferring learned attractors between domains), not for cross-modal semantic alignment. There is no mechanism to align phase coherence patterns from different modalities into a shared semantic space.

**What needs to be built**:

#### 4a. Cross-modal phase alignment module

```python
class CrossModalPhaseBinder(nn.Module):
    """Aligns phase coherence patterns across modalities"""
    def forward(self, coherence_a: Tensor, coherence_b: Tensor) -> Tensor:
        # 1. Project both to shared coherence space
        # 2. Compute cross-modal phase alignment score
        # 3. Learn a linear mapping that maximizes alignment
        #    for semantically corresponding pairs
        return alignment_score
```

#### 4b. Cross-modal contrastive loss

```python
class CrossModalCoherenceLoss(nn.Module):
    """Corresponding pairs should have aligned coherence patterns"""
    def forward(self, audio_coherence, image_coherence, labels):
        # For each (audio, image) pair with same label:
        #   Maximize phase coherence alignment
        # For pairs with different labels:
        #   Minimize alignment
```

#### 4c. Cross-modal retrieval benchmark

```
Input: Audio sample
Query: Find image with most similar phase coherence pattern
Metric: Recall@K, MAP
Baseline: Random retrieval, CLIP, amplitude-only features
```

**Existing code to leverage**: The `RiemannianMetricLearner` and `GeodesicComputer` in `riemannian_coherence/` provide the geometric infrastructure for measuring semantic similarity. Extend them to operate across modalities.

**The proof**: If cross-modal retrieval via phase coherence significantly outperforms amplitude-only retrieval, C4 is supported.

---

### Requirement 5: Compositional structure extraction + generalization benchmarks (proves C5)

**Goal**: Prove that phase-coherent multi-scale representations enable systematic compositional generalization — the key AGI capability.

**The problem**: Bifrost's seven-layer framework (L2-L7) is specified but not implemented. There is no mechanism to extract compositional structure from phase coherence patterns, and no benchmarks to test compositional generalization.

**What needs to be built**:

#### 5a. Compositional structure extractor (L2)

```python
class CompositionalStructureExtractor(nn.Module):
    """Extracts hierarchical compositional structure from phase coherence"""
    def forward(self, multi_scale_coherence: List[Tensor]) -> Dict:
        # 1. Identify phase-locked groups at each scale
        #    (groups of frequency bands with high PLV)
        # 2. Detect hierarchical nesting:
        #    Scale 1 groups → parts of Scale 2 groups → parts of Scale 3 groups
        # 3. Extract compositional tree
        return {
            'parts': List[List[int]],  # phase-locked groups at each scale
            'tree': Tree,              # hierarchical composition structure
            'binding_scores': Tensor,  # strength of each compositional binding
        }
```

This is the mechanism that turns phase coherence patterns into explicit compositional structure — the "parts" and "whole" relationships that enable systematic generalization.

#### 5b. Phase-based slot attention

Combine Bifrost's phase coherence with object-centric representation learning:

```python
class PhaseCoherenceSlotAttention(nn.Module):
    """Slot attention where slots are phase-coherent groups"""
    def forward(self, spectral_features):
        # 1. Compute phase coherence between all spectral locations
        # 2. Initialize slots as phase-coherent groups
        # 3. Iteratively refine slots via phase-based attention
        # 4. Output: object-centric spectral representations
        return slots
```

**Reference**: Locatello et al. (2020) Slot Attention, extended with phase coherence.

#### 5c. Compositional generalization benchmark

```
Train: Compositions A, B, C (e.g., "red square", "blue circle", "green triangle")
Test: Novel compositions D, E, F (e.g., "red circle", "blue triangle", "green square")

Metric: Accuracy on novel compositions
Baseline: Standard neural network (fails on novel compositions)
Hypothesis: Phase-coherent representations generalize compositionally
```

For audio:
```
Train: Phoneme combinations (ba, di, ku)
Test: Novel combinations (bi, du, ka)
```

For images:
```
Train: Object-color combinations (red car, blue house)
Test: Novel combinations (blue car, red house)
```

#### 5d. Systematic generalization metrics

```python
class SystematicGeneralizationMetrics:
    def compositional_accuracy(self, model, train_compositions, test_compositions) -> float
    def systematicity_score(self, model, held_out_combinations) -> float
    def productivity_score(self, model, longer_compositions) -> float
    def substitutivity_score(self, model, swapped_components) -> float
```

**The proof**: If phase-coherent representations achieve significantly higher compositional generalization than baseline representations, C5 is supported.

---

## The engineering stack

Putting it all together, the engineering required to prove the Bifrost thesis forms a stack:

```
Layer 5: Compositional generalization benchmarks
         (proves C5: AGI-level generalization)
              ↑
Layer 4: Cross-modal phase binding
         (proves C4: cross-modal alignment)
              ↑
Layer 3: Unified coherence metric + modality instances
         (proves C3: cross-modal generalization)
              ↑
Layer 2: Cross-scale coherence aggregation
         (proves C2: multi-scale necessity)
              ↑
Layer 1: Phase-preservation infrastructure + ablation framework
         (proves C1: phase coherence captures structure)
              ↑
Layer 0: Bifrost core (EXISTS)
         SpectralTensor, ComplexSSM, CBMPC, PhaseCongruency,
         ResonanceAttention, RiemannianCoherence
```

Each layer builds on the one below. You cannot prove C5 without first proving C1-C4. You cannot prove cross-modal generalization without first proving within-modal phase coherence.

---

## What type of engineering is this?

This is not conventional software engineering. It is **hypothesis-driven engineering** — each component is designed to test a specific scientific claim. The engineering is the experiment.

### The five engineering disciplines required:

#### 1. Complex-valued deep learning engineering
- Complex-valued tensors, convolutions, batch norm, attention
- Complex backpropagation (Wirtinger calculus)
- Complex SSM kernels (CUDA/Triton)
- Phase-aware loss functions
- **Bifrost status**: Mostly complete. `ComplexLinear`, `ComplexSelectiveScan`, `ComplexBifrostTrainer` exist.

#### 2. Signal processing engineering
- Phase extraction, Hilbert transform, analytic signals
- Wavelet transforms (CWT, DWT, wavelet packets)
- Phase locking value, wavelet coherence
- Phase congruency (log-Gabor filters)
- Graph wavelets
- **Bifrost status**: Partially complete. CBMPC and PhaseCongruency exist. Wavelet coherence and graph wavelets are missing.

#### 3. Ablation and validation engineering
- Phase ablation harness (randomize, zero, quantize, noise phase)
- Scale ablation harness (single-scale, scale-subset, cross-scale-destroy)
- Modality ablation (within-modal vs cross-modal)
- Compositional generalization benchmarks
- Statistical testing (paired t-tests, Bonferroni correction, effect sizes)
- **Bifrost status**: Missing. This is the biggest gap. The experiments exist as scripts but the ablation framework does not.

#### 4. Geometric/manifold engineering
- Riemannian metric learning on coherence space
- Geodesic computation for semantic similarity
- Cross-modal alignment in coherence space
- Phase coherence as a metric tensor
- **Bifrost status**: Partially complete. `RiemannianMetricLearner` and `GeodesicComputer` exist but are not adapted for cross-modal use.

#### 5. Compositional structure engineering
- Phase-locked group detection
- Hierarchical composition tree extraction
- Phase-based slot attention
- Compositional generalization benchmarks
- **Bifrost status**: Missing. This is the most novel and most difficult engineering.

---

## Priority order

Based on the dependency stack and what already exists:

| Priority | Engineering task | Proves | Effort | Status |
|---|---|---|---|---|
| 1 | Phase ablation harness | C1 | Medium | Missing |
| 2 | Phase coherence metrics at every layer | C1 | Medium | Missing |
| 3 | Cross-scale coherence module | C2 | Medium | Missing |
| 4 | Scale ablation framework | C2 | Low | Missing |
| 5 | Wavelet coherence extractor (sensor) | C3 | Medium | Missing |
| 6 | CIFAR-10 validation (image MSC) | C3 | Low | Blocked (deps) |
| 7 | Unified coherence metric | C3 | Medium | Missing |
| 8 | Cross-modal phase binding | C4 | High | Missing |
| 9 | Cross-modal retrieval benchmark | C4 | Medium | Missing |
| 10 | Compositional structure extractor | C5 | High | Missing |
| 11 | Phase-based slot attention | C5 | High | Missing |
| 12 | Compositional generalization benchmark | C5 | Medium | Missing |

---

## The minimal viable proof

To prove the thesis with the least engineering, you need:

### Step 1: Prove phase matters (C1)
- Build phase ablation harness
- Run CBMPC (audio) with and without phase
- Run PhaseCongruency (image) with and without phase
- Show phase-coherent features > phase-ablated features (p < 0.05)

### Step 2: Prove multi-scale matters (C2)
- Build cross-scale coherence module
- Run with single-scale vs multi-scale vs cross-scale
- Show cross-scale > multi-scale > single-scale

### Step 3: Prove cross-modal generalization (C3)
- Implement wavelet coherence (sensor)
- Run MSC on audio, image, sensor
- Show all three achieve above-baseline accuracy

### Step 4: Prove cross-modal alignment (C4)
- Build cross-modal phase binder
- Run cross-modal retrieval (audio → image)
- Show phase coherence retrieval > amplitude-only retrieval

### Step 5: Prove compositional generalization (C5)
- Build compositional structure extractor
- Run compositional generalization benchmark
- Show phase-coherent representations generalize compositionally

**Each step is a publishable result.** Steps 1-3 are achievable with current infrastructure plus moderate engineering. Steps 4-5 require novel engineering.

---

## What makes this engineering different from conventional ML

Conventional ML engineering asks: "How do I build a system that performs well on task X?"

Bifrost engineering asks: "How do I build a system whose *structure* proves a *theory of intelligence*?"

The difference is that every engineering decision in Bifrost is a scientific decision. The choice to use complex-valued SSMs is not an architecture choice — it is a hypothesis test (does phase matter?). The choice to compute cross-scale coherence is not a feature engineering choice — it is a hypothesis test (does multi-scale structure matter?).

This means the engineering must be:
1. **Ablation-first**: Every component must be toggleable, so you can run with and without it
2. **Metric-driven**: Every component must expose phase coherence metrics, not just accuracy
3. **Cross-modal**: Every component must work across modalities, not just audio
4. **Compositional**: Every component must be evaluated on compositional generalization, not just classification

The engineering IS the experiment. The framework IS the proof.
