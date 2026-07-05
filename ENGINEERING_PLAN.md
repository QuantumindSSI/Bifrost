# Bifrost Engineering Plan

**Document type:** Living technical roadmap  
**Scope:** Phases 1–4 (current through production deployment)  
**Last updated:** June 2026

---

## Executive Summary

Bifrost targets **seven distinct structural layers of semantic understanding**, each requiring different mathematics. Existing AI addresses Layer 1 (distributional statistics) well; Bifrost adds Layers 2–7. The engineering plan is divided into four phases:

| Phase | Focus | Duration | Priority Signal |
|---|---|---|---|
| 1 | Core infrastructure + Phase 1 training | 8 weeks | Phase coherence, attractor stability |
| 2 | Causal structure + TDA + hierarchical SSM | 10 weeks | Causal graph, predictive error, topology |
| 3 | Symmetry + disentanglement + Allen algebra | 10 weeks | Symmetry, factor independence, narrative |
| 4 | LLM integration + production + evaluation | 8 weeks | Full 7-layer semantic score |

---

## Semantic Layer Coverage Map

This table tracks which engineering component addresses which semantic layer, at what phase, and with what mathematical primitive.

| Semantic Layer | What it captures | Primary component | Phase | Mathematical primitive |
|---|---|---|---|---|
| L1 Distributional | Statistical co-occurrence | SpectralBinding (spectral covariance) | 1 | Phase coherence spectrum |
| L2 Compositional | Part-whole recursive structure | HierarchicalSSM (multi-timescale) | 2 | Pyramidal SSM state |
| L3 Causal-world-model | Directed influence, counterfactuals | CausalGraphTensor (Granger) | 2 | Granger causality GC(i→j) |
| L4 Topological/Geometric | Curved manifold of concepts | RiemannianMetricLearner + TDA | 1+2 | Cholesky metric G = LLᵀ + Betti numbers |
| L5 Temporal/Narrative | Discourse arcs, event sequences | ComplexSSM + TemporalRelationTensor | 1+3 | Complex recurrence + Allen algebra |
| L6 Invariance/Symmetry | What transforms leave meaning fixed | SymmetryTensor | 3 | Autocorrelation spectrum + stabiliser group |
| L7 Disentanglement | Independent generative factors | DisentangledTensor (β-VAE) | 3 | TCVAE loss + MI matrix |

---

## Phase 1: Foundation (In Progress — 75% Complete)

### Goal
Complete core infrastructure: distributed training, real data ingestion, cross-modal alignment, empirical validation of phase coherence claims.

### 1.1 Training Stabilization [COMPLETE]

Architecture components all implemented and tested:
- `ComplexSpectralDecomposer`: complex diagonal A matrix, parallel Blelloch scan
- `SpectralBinding`: collapse-proof, canonical phase bypass
- `HarmonicBinding`: explicit harmonic grid with trainable weights
- `ResonanceAttention`: multi-band phase coherence attention
- `RiemannianMetricLearner`: Cholesky metric tensor, geodesic computer
- `AttractorLearningModule`: VQ-VAE 65K codebook, temporal stability predictor
- `TruePhaseLockDetector`: Adler equation, temporal consistency, coupling strength
- `BifrostEnhancedLLM`: three adapter modes (prefix, intermediate, verifier)

Six collapse-prevention mechanisms active (ratio loss, crossover negatives, gradient clipping, mixed precision stability, warm-up scheduling, curriculum learning).

### 1.2 Data Ingestion [IN PROGRESS]

**Remaining tasks:**

```
Task 1.2.1: Credential management
    • AWS S3 / GCS credential integration
    • HuggingFace API authentication
    • Secrets manager integration (no hardcoded keys)
    • Timeline: 1 week, 1 engineer

Task 1.2.2: Download pipelines
    • Audio: LibriLight (60K hours), AudioSet (2M clips), FMA (100K tracks)
    • Images: LAION-400M filtered, COCO (330K), ImageNet (14M)
    • Text: Books3, CommonCrawl filtered
    • Video: YouTube-8M clips, Kinetics-700
    • Sensors: industrial vibration, IMU, automotive CAN
    • Timeline: 2 weeks, 2 engineers, ~3 PB total

Task 1.2.3: Storage + deduplication
    • Compression: Zstandard for spectral tensors
    • Near-duplicate detection: MinHash LSH for audio/text
    • Metadata index: SQLite or Parquet per modality
    • Timeline: 1 week, 1 engineer

Task 1.2.4: Cross-modal alignment
    • Caption-image pairs: COCO annotation extraction
    • Audio-text alignment: Whisper transcription → text
    • Video-audio sync: ffmpeg scene detection + STFT alignment
    • Music-lyrics: MusicXML + lyrics source alignment
    • Timeline: 2 weeks, 2 engineers
```

**Quality filters (all implemented in `multimodal_curator.py`):**
- Audio: SNR > 20dB, 0.5s–30s duration, MD5 deduplication
- Images: 256×256 min, 4:1 max aspect ratio, blur detection
- Text: perplexity < threshold (KenLM 3-gram), English detection (langdetect)
- Video: min 10 frames, scene boundary detection, resolution > 360p

### 1.3 Empirical Validation [COMPLETE — validation suite built]

The `empirical_validation.py` module validates five core claims with quantitative thresholds:

| Claim | Test | Success threshold |
|---|---|---|
| Anti-phase discrimination | Phase-coherent vs anti-phase: Δcoherence | Δ > 0.5 |
| Harmonic binding | Harmonic stack vs inharmonic: attention diff | effect > 0.3 |
| Cross-modal retrieval | Audio↔image retrieval accuracy | > 0.7 |
| Attractor stability | Centroid variance across windows | < 0.2 |
| Phase-lock across domains | Phase-lock score: matched vs random | > 0.6 |

**Run validation:**
```bash
python -m bifrost.validation.empirical_validation --full
```

### 1.4 Training Launch Checklist

```
[ ] 1. Single GPU smoke test: 32 batch, 10 epochs, d_model=256
[ ] 2. Multi-GPU test: 2x GPU, 64 batch, validate DDP sync
[ ] 3. Coherence metrics stable (no collapse) for 1000 steps
[ ] 4. Checkpoint save/restore cycle verified
[ ] 5. First 100K audio hours ingested and validated
[ ] 6. Launch 8x A100 training: 256 batch, 100 epochs, d_model=1024
[ ] 7. Coherence metric improves over baseline for 10K steps
[ ] 8. Cross-modal alignment > 0.6 audio-video after 50K steps
```

**Phase 1 success criteria:**
- Coherence metrics improve monotonically during training
- Phase-lock bridge achieves > 0.7 precision cross-modal
- Distributed training runs without DDP hangs on 8x A100
- All five empirical claims validated on real held-out data

---

## Phase 2: Causal Structure, Topology, and Hierarchical SSM

**Duration:** 10 weeks  
**Semantic layers addressed:** L2 (Compositional), L3 (Causal), L4 (Topological)  
**New representations:** PredictiveErrorTensor, CausalGraphTensor, PersistenceTensor, HierarchicalSSM

---

### 2.1 Predictive Error Tensor

**Motivation:** The complex SSM already computes implicit prediction error internally. Surfacing this signal makes it explicit, self-supervised, and usable as a semantic boundary detector without additional training.

**Implementation: `src/bifrost/decomposer/predictive_error.py`**

```python
@dataclass
class PredictiveErrorTensor:
    prediction: torch.Tensor      # (B, T, F) SSM expected output at each t
    error: torch.Tensor           # (B, T, F) actual[t] - predicted[t]
    error_magnitude: torch.Tensor # (B, T, F) |error| per frequency band
    error_phase: torch.Tensor     # (B, T, F) angle(error) — direction of surprise
    cumulative_surprise: torch.Tensor  # (B, T) integral of |error|
    surprise_rate: torch.Tensor   # (B, T, F) d|error|/dt — surprise change rate
    semantic_boundary_score: torch.Tensor  # (B, T) high = likely boundary

class PredictiveErrorExtractor(nn.Module):
    """
    Hooks into ComplexSpectralDecomposer to extract per-step prediction error.
    
    The SSM recurrence is:
        h[t] = exp(−Δ·A) · h[t−1] + Δ·B[t] · x[t]
        y[t] = C · h[t] + D · x[t]

    Prediction:
        y_pred[t] = C · h[t−1]  (using previous state, before x[t] update)
    Error:
        e[t] = y[t] − y_pred[t]
    """
    def extract(
        self,
        ssm: ComplexSpectralDecomposer,
        spectral_tensor: SpectralTensor,
    ) -> PredictiveErrorTensor:
        ...
```

**Semantic boundary detection:**
```
surprise_rate[t] > threshold → probable semantic boundary at t
```
This is a zero-shot boundary detector: semantic units (phonemes, words, sentences) produce characteristic prediction-error spikes at their boundaries.

**Integration with training:** Add `PredictiveErrorLoss`:
```
L_error = MSE(surprise_rate, boundary_labels)
```
where `boundary_labels` can be derived from forced alignment (for text/audio) or left unsupervised.

**Deliverables:**
- `predictive_error.py` — extractor + dataclass
- `tests/test_predictive_error.py` — unit tests
- Integration into `ComplexSpectralDecomposer.forward()` as optional flag
- Benchmark: boundary F1 on LibriSpeech forced-alignment labels

---

### 2.2 Granger Causal Graph

**Motivation:** All current Bifrost representations are symmetric (phase coherence is a symmetric relation). The CausalGraphTensor is the first **directed** (asymmetric) representation in the pipeline. It answers: "does frequency band A predict frequency band B?" which is qualitatively different from "do A and B co-occur?"

**Granger causality definition:**
```
GC(i→j) = log( Var[e_j^restricted] / Var[e_j^full] )
```
where:
- `Var[e_j^restricted]` = variance of prediction error for band j without band i's history
- `Var[e_j^full]` = variance of prediction error for band j with band i's history

GC(i→j) > 0 means band i Granger-causes band j.

**Implementation: `src/bifrost/causal_graph/granger_causality.py`**

```python
@dataclass
class CausalGraphTensor:
    adjacency: torch.Tensor       # (B, n_bands, n_bands) directed weights GC(i→j)
    causal_strength: torch.Tensor # (B, n_bands, n_bands) signed GC values
    lag_structure: torch.Tensor   # (B, n_bands, n_bands, max_lag) lag-dependent GC
    p_values: torch.Tensor        # (B, n_bands, n_bands) significance of each edge
    cycle_count: int              # number of feedback cycles (causal loops)
    feedback_strength: float      # average cycle weight (high = highly recurrent)

class GrangerCausalityExtractor(nn.Module):
    """
    Extracts Granger causality graph from SSM transition matrices.
    
    Key insight: the learned A matrices in the complex SSM encode
    spectral-band interaction dynamics. Their off-diagonal elements
    (in a full, non-diagonal A) are proxies for Granger causal influence.
    
    For a diagonal A (current Bifrost), we use the learned B[t] input matrices
    as conditional predictors across bands.
    
    Two computation modes:
    1. FAST (approximate): use SSM A matrix structure directly
    2. EXACT (expensive): VAR model fit on extracted SSM hidden states
    """
    
    def __init__(
        self,
        n_bands: int,
        max_lag: int = 10,
        significance_level: float = 0.05,
        mode: str = "fast",  # "fast" | "exact"
    ): ...
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, D) SSM hidden states
        band_projector: nn.Linear,    # project D → n_bands
    ) -> CausalGraphTensor: ...
```

**VAR-based computation (mode="exact"):**
```
1. Project hidden states h[t] → band activations x[t] via band_projector
2. Fit VAR(p) model: x[t] = Σ_{k=1}^{p} A_k x[t−k] + e[t]
3. For each pair (i,j): fit restricted model excluding x_i lags
4. Compute GC(i→j) = log(Var[e_j^restricted] / Var[e_j^full])
5. F-test for significance
```

**Fast computation (mode="fast"):**
```
1. Extract SSM B matrix sequence: B[t] ∈ ℝ^{D×F}
2. Compute band cross-covariance: Cov[B_i, B_j at lag k]
3. Normalised Granger proxy: GC_proxy(i→j) = Σ_k Cov[B_i[t−k], B_j[t]]
   (asymmetric by construction since lag is directional)
```

**Causal attention head:**
```python
class CausalAttentionHead(nn.Module):
    """
    Attends using directed causal graph instead of dot product.
    Score(q_i, k_j) = GC(i→j)  (influence from j's past predicting i's future)
    """
```

**Deliverables:**
- `src/bifrost/causal_graph/` — new module
- `granger_causality.py` — GrangerCausalityExtractor + CausalGraphTensor
- `causal_attention.py` — CausalAttentionHead
- `tests/test_granger_causality.py`
- Integration: CausalGraphTensor returned optionally from S1 forward pass

---

### 2.3 TDA Persistence Tensor

**Motivation:** Topology (connectedness, holes, voids) is a global property of the spectral amplitude landscape that phase coherence cannot detect. A chord (several frequencies simultaneously present) produces a different topological signature than a melody (sequential frequencies); the Betti numbers distinguish them.

**Topological data analysis (TDA) background:**
- Build Vietoris-Rips filtration on point cloud formed by (frequency, amplitude, time) tuples
- Track connected components (β₀), loops (β₁), voids (β₂) as the filtration parameter increases
- Persistence = birth radius to death radius; long bars = topologically significant features

**Implementation: `src/bifrost/topology/persistence_tda.py`**

```python
@dataclass
class PersistenceTensor:
    diagram: list                    # [(birth, death, dim), ...] sorted by persistence
    barcode_lengths: torch.Tensor    # (n_features,) = death − birth
    betti_numbers: torch.Tensor      # (3,) = [β₀, β₁, β₂]
    persistent_generators: list      # indices of most persistent topological features
    wasserstein_distance: float      # W₂ distance to reference diagram
    bottleneck_distance: float       # W_∞ distance (more robust to outliers)
    topological_entropy: float       # H = Σ (persistence_i / total_persistence)

class TDAPersistenceExtractor(nn.Module):
    """
    Computes persistence diagrams from spectral amplitude data.
    
    Input: SpectralTensor.amplitude  shape (B, T, F)
    
    Pipeline:
        1. For each batch item: treat (t, f, amplitude[t,f]) as 3D point cloud
        2. Vietoris-Rips filtration via gudhi or ripser
        3. Persistence diagram extraction (dim 0, 1, 2)
        4. Wasserstein/bottleneck distance to reference diagram
    
    This is parameter-free: no learnable weights.
    The persistence diagram is a topological fingerprint of the signal.
    """
    
    def __init__(
        self,
        max_dimension: int = 2,
        max_edge_length: float = 2.0,
        backend: str = "ripser",  # "ripser" | "gudhi"
        reference_diagram: Optional[list] = None,
    ): ...
```

**Topological signatures by signal type:**
```
Speech phoneme /a/: β₀=1 (one connected cluster), β₁=0
Speech phoneme /s/: β₀=3+ (spread across frequencies), β₁=1+
Piano chord C-E-G: β₀=3, β₁=1 (loop connecting harmonics)
White noise: β₀=N (many isolated clusters), β₁≈0
```

**Differentiable TDA (for learning):**
Use `TopologicalSignatureDistance` from `pytorch-topological` (differentiable Wasserstein):
```
L_tda = W₂(diag_output, diag_target)
```
This enables learning to produce specific topological signatures.

**Deliverables:**
- `src/bifrost/topology/persistence_tda.py`
- `src/bifrost/topology/differentiable_tda.py` (differentiable wrapper)
- `tests/test_persistence_tda.py` — phoneme discrimination test
- Optional dependency: `gudhi>=3.8.0` or `ripser>=0.6.0`
- Benchmark: Betti number discrimination accuracy on instrument classification

---

### 2.4 Hierarchical Multi-Timescale SSM

**Motivation:** A single SSM timescale captures one level of temporal structure. Language has at least five simultaneously relevant timescales: phoneme (~30ms), syllable (~100ms), word (~300ms), phrase (~1s), sentence (~3s), paragraph (~10s). Current Bifrost uses a single SSM; this misses compositional structure (Layer 2).

**Architecture: `src/bifrost/decomposer/hierarchical_ssm.py`**

```python
@dataclass
class HierarchicalSSMConfig:
    n_levels: int = 5
    timescales_ms: list = [10, 100, 500, 2000, 10000]  # L0→L4
    d_model_per_level: list = [64, 128, 256, 256, 128]  # narrower at extremes
    compression_factor: int = 4  # time compression between levels
    fusion_mode: str = "attention"  # "attention" | "concat" | "gate"

class HierarchicalComplexSSM(nn.Module):
    """
    Pyramid of ComplexSpectralDecomposers operating at different timescales.
    
    Architecture:
    
    Input: x[t]  (T timesteps)
    
    Level 0 (10ms, raw frame rate):
        SSM_0(x[t]) → h_0[t], y_0[t]   (all T frames)
    
    Level 1 (100ms, 10x slower):
        Strided pool: x_1[t] = pool(y_0, stride=10)
        SSM_1(x_1[t]) → h_1[t], y_1[t]  (T/10 frames)
    
    Level 2 (500ms):
        x_2[t] = pool(y_1, stride=5)
        SSM_2(x_2[t]) → h_2[t], y_2[t]  (T/50 frames)
    
    Level 3 (2s):
        x_3[t] = pool(y_2, stride=4)
        SSM_3(x_3[t]) → h_3[t], y_3[t]  (T/200 frames)
    
    Level 4 (10s):
        x_4[t] = pool(y_3, stride=5)
        SSM_4(x_4[t]) → h_4[t], y_4[t]  (T/1000 frames)
    
    Fusion (upsample back to T, then fuse):
        y_fused[t] = Attention([y_0[t], up(y_1)[t], up(y_2)[t], up(y_3)[t], up(y_4)[t]])
    
    The attention over levels learns WHICH timescale is relevant for each token.
    """
    
    def __init__(self, config: HierarchicalSSMConfig): ...
    
    def forward(
        self,
        spectral_tensor: SpectralTensor,
        return_per_level: bool = False,
    ) -> Tuple[SpectralTensor, Optional[list]]: ...
```

**Level interaction (compositional binding):**
```
Level 0 detects: phoneme-level phase patterns (formants)
Level 1 detects: syllable-level rhythmic patterns
Level 2 detects: word-level prosodic patterns
Level 3 detects: phrase-level intonation
Level 4 detects: discourse-level structure
```

The cross-level attention weights learn WHEN a higher-level abstraction is active for a given frame, realising Layer 2 (compositional structure).

**Deliverables:**
- `src/bifrost/decomposer/hierarchical_ssm.py`
- `tests/test_hierarchical_ssm.py`
- Integration: `BifrostPipeline` optional `hierarchical=True` mode
- Benchmark: word boundary F1 using level-1 SSM state vs. single-level baseline

---

### 2.5 Phase 2 Dependencies and Timeline

```
Week 1–2:  PredictiveErrorTensor (2.1)
           − Implement extractor, hook into SSM
           − Unit tests, semantic boundary validation on LibriSpeech

Week 3–4:  GrangerCausalityExtractor (2.2) — fast mode
           − Implement B matrix cross-covariance proxy
           − Unit tests, verify asymmetry property
           − CausalAttentionHead (drop-in for ResonanceAttention)

Week 5:    GrangerCausalityExtractor — exact mode (VAR model)
           − Implement VAR(p) on SSM hidden states
           − F-test significance, cycle count

Week 6–7:  TDA Persistence (2.3)
           − ripser integration for persistence diagrams
           − Differentiable Wasserstein via pytorch-topological
           − Phoneme discrimination benchmark

Week 8–10: Hierarchical SSM (2.4)
           − 5-level pyramid implementation
           − Cross-level attention fusion
           − Word boundary benchmark
           − Integration with full pipeline

Week 10:   Integration + Phase 2 eval
           − Full pipeline: S0→S1(hierarchical)→S2→S3→S4+causal+tda
           − Phase 2 report: semantic layer coverage scores
```

**Phase 2 success criteria:**
- PredictiveErrorTensor boundary F1 > 0.6 on LibriSpeech (zero-shot)
- GC(i→j) ≠ GC(j→i) for at least 60% of band pairs (asymmetry verified)
- TDA Betti numbers discriminate 5 instrument families at > 80% accuracy
- Hierarchical SSM word boundary F1 > flat SSM baseline

---

## Phase 3: Symmetry, Disentanglement, and Temporal Relations

**Duration:** 10 weeks  
**Semantic layers addressed:** L5 (Temporal narrative — extension), L6 (Invariance/Symmetry), L7 (Disentanglement)  
**New representations:** SymmetryTensor, DisentangledTensor, TemporalRelationTensor

---

### 3.1 Symmetry Tensor

**Motivation:** The current HarmonicBinding module assumes octave invariance by hardcoding the overtone grid (f, 2f, 3f, ...). This is correct for pitched audio but wrong for speech (formants don't follow harmonic series), images (rotational symmetry), and sensors (periodicity at non-integer multiples). The SymmetryTensor detects whatever invariance group the signal actually obeys.

**Mathematical framework:**

A transformation group G acts on signal space. The signal x is G-invariant if:
```
|x − T_g(x)| < ε   for all g ∈ G
```
We detect: which elements g have this property, forming the stabiliser group Stab(x).

**Implementation: `src/bifrost/symmetry/symmetry_detection.py`**

```python
@dataclass
class SymmetryTensor:
    # Temporal periodicities
    period_spectrum: torch.Tensor    # (B, n_periods) detected periods (in samples)
    period_strengths: torch.Tensor   # (B, n_periods) autocorrelation at each period
    
    # Frequency symmetries
    octave_invariance: torch.Tensor  # (B,) energy ratio under f → 2f (octave symmetry)
    fifth_invariance: torch.Tensor   # (B,) energy ratio under f → 1.5f (perfect fifth)
    sub_octave_pattern: torch.Tensor # (B, n_bins) detected frequency ratios
    
    # Temporal scaling
    tempo_invariance: torch.Tensor   # (B,) correlation under time-stretch [0.8, 1.25]
    
    # Spatial (images only)
    rotational_order: torch.Tensor   # (B,) detected rotational symmetry order
    reflection_axes: torch.Tensor    # (B, n_axes) detected reflection axes
    
    # Group structure
    stabiliser_generators: list      # generators of the detected stabiliser subgroup
    group_type: str                  # "cyclic_n" | "dihedral_n" | "trivial" | "continuous"

class SymmetryDetector(nn.Module):
    """
    Parameter-free symmetry detection via autocorrelation and group orbit analysis.
    
    Temporal periodicity: autocorrelation in time domain
    Frequency symmetry: log-frequency autocorrelation (periodic in log-freq ↔ harmonic)
    Spatial symmetry: angular Fourier transform of image magnitude
    
    Optional learnable component: SymmetryGroupEncoder (learns which group type
    each signal belongs to via classification loss on known-symmetry signals)
    """
```

**Generalised harmonic binding:**
```
current HarmonicBinding: Σ_k Energy(k·f)  [fixed octave-harmonic]
SymmetryAdaptiveBinding: Σ_k Energy(r_k·f) where r_k = detected frequency ratios
```

**Deliverables:**
- `src/bifrost/symmetry/symmetry_detection.py`
- `src/bifrost/symmetry/adaptive_harmonic_binding.py` (replaces fixed grid)
- `tests/test_symmetry_detection.py`
- Benchmark: symmetry group classification accuracy (cyclic vs dihedral vs trivial)

---

### 3.2 Disentangled Tensor

**Motivation:** The VQ-VAE codebook in S3 currently encodes a mixture of content, style, and noise. The DisentangledTensor factorises the representation into statistically independent components — content invariant to style, style invariant to content — enabling controlled generation and more robust classification.

**Mathematical framework:**

Total correlation (TC) as disentanglement measure:
```
TC(z) = KL( q(z) || Π_i q(z_i) ) = Σ_{i≠j} I(z_i; z_j)
```
where I(z_i; z_j) is mutual information between factor i and factor j. TC = 0 ↔ perfect disentanglement.

**Implementation: `src/bifrost/disentanglement/disentangled_vae.py`**

```python
@dataclass
class DisentangledTensor:
    content_factors: torch.Tensor    # (B, d_content) semantic content
    style_factors: torch.Tensor      # (B, d_style) speaker/instrument/environment
    temporal_factors: torch.Tensor   # (B, T, d_temporal) time-varying factors
    mi_matrix: torch.Tensor          # (n_factors, n_factors) I(z_i; z_j) estimates
    total_correlation: float         # sum of off-diagonal MI (lower = better)
    
class TCVAEEncoder(nn.Module):
    """
    Total Correlation VAE (TC-VAE / β-TC-VAE).
    Decomposes AttractorLearningModule features into disentangled factors.
    
    Loss:
        L = reconstruction_loss
            + β · TC(z)              [total correlation penalty]
            + mutual_info_term        [maximise I(z; x)]
            + dimension_wise_KL      [per-factor regularisation]
    
    where TC is estimated using the minibatch stratified estimator
    (Chen et al., 2018) without requiring a discriminator.
    
    Factor types:
        content_factors: invariant to style (speaker, instrument, environment)
            − Implemented via style adversarial training
        style_factors: invariant to content
            − Implemented via content adversarial training
        temporal_factors: captures time-varying aspects
            − Extracted from hierarchical SSM level-1 states
    """
    
    def __init__(
        self,
        d_input: int,
        d_content: int = 64,
        d_style: int = 32,
        d_temporal: int = 64,
        beta: float = 6.0,  # TC penalty weight (higher = more disentangled)
    ): ...
```

**MI matrix estimation:**
```python
class MutualInformationEstimator(nn.Module):
    """
    MINE-based MI estimator: I(z_i; z_j) via neural network f_θ.
    I(X; Y) ≥ E[f_θ(x,y)] − log E[exp(f_θ(x,y'))]  (MINE lower bound)
    Evaluated for all pairs (i,j) to build MI matrix.
    """
```

**Content-style interpolation (generation):**
```python
# Keep content, change style
z_new = DisentangledTensor(
    content_factors=z_speech.content_factors,  # speaker A's content
    style_factors=z_target.style_factors,       # speaker B's style
    ...
)
x_new = decoder(z_new)  # speech in speaker B's voice
```

**Deliverables:**
- `src/bifrost/disentanglement/disentangled_vae.py`
- `src/bifrost/disentanglement/mi_estimator.py` (MINE)
- `tests/test_disentangled_vae.py`
- Benchmark: disentanglement score (DCI-Disentanglement, 3DShapes or dSprites)
- Benchmark: zero-shot style transfer quality (speaker similarity metric)

---

### 3.3 Temporal Relation Tensor (Allen Algebra)

**Motivation:** The complex SSM captures temporal ordering via phase lag. The TemporalRelationTensor makes this explicit as a qualitative relation over attractor activations, implementing James Allen's 13 temporal interval relations.

**Allen's interval relations:**
```
before(A, B):   A ends before B starts           A---  B---
meets(A, B):    A ends exactly when B starts      A---B---
overlaps(A, B): A starts before B, they overlap   A---
                                                     B---
starts(A, B):   A and B start together, A shorter A--
                                                   B------
during(A, B):   A entirely within B                 A--
                                                   B------
finishes(A, B): A and B end together, A shorter      A--
                                                   B------
equals(A, B):   A and B coincide exactly           A------
                                                   B------
+ 6 inverses + their mirrors = 13 relations total
```

**Implementation: `src/bifrost/temporal/allen_algebra.py`**

```python
@dataclass
class TemporalRelationTensor:
    relations: torch.Tensor           # (B, n_attractors, n_attractors) Allen relation IDs
    # IDs: 0=before, 1=meets, 2=overlaps, 3=starts, 4=during, 5=finishes,
    #       6=equals, 7=after, 8=met-by, 9=overlapped-by,
    #       10=started-by, 11=contains, 12=finished-by
    
    relation_confidence: torch.Tensor # (B, n_attractors, n_attractors) confidence
    duration_matrix: torch.Tensor     # (B, n_attractors) activation duration in frames
    onset_matrix: torch.Tensor        # (B, n_attractors) onset timestep
    offset_matrix: torch.Tensor       # (B, n_attractors) offset timestep
    narrative_structure: dict         # inferred high-level structure

class AllenRelationExtractor(nn.Module):
    """
    Extracts Allen temporal relations from attractor activations.
    
    Input: AttractorActivations (B, T, n_attractors) — from S3
    
    Step 1: Segment each attractor's activation into intervals
            (onset, offset) per continuous activation above threshold
    
    Step 2: For each pair (A, B): compute Allen relation
            based on onset_A, offset_A, onset_B, offset_B
            with tolerance ε for boundary cases (meets, starts, finishes, equals)
    
    Step 3: Optionally train a soft Allen classifier:
            RelationNet(onset_A, offset_A, onset_B, offset_B) → P(relation | A, B)
            Trained on synthetic interval pairs with known relations.
    """
    
    def __init__(
        self,
        tolerance_frames: int = 3,    # ε for boundary relation classification
        learnable: bool = True,        # use soft classifier or hard rules
    ): ...
```

**Narrative structure inference:**
```
Given TemporalRelationTensor over n attractors:
    - Build transitivity closure (Allen algebra is a constraint algebra)
    - Identify narrative arc: intro → development → climax → resolution
    - Detect cyclic patterns: A before B before C before A (loops)
    - Compute "temporal density": how many relations per attractor
```

**Deliverables:**
- `src/bifrost/temporal/allen_algebra.py`
- `src/bifrost/temporal/narrative_inference.py`
- `tests/test_allen_algebra.py`
- Integration: AllenRelationExtractor applied to S3 attractor outputs
- Benchmark: narrative structure recovery on annotated story datasets

---

### 3.4 Phase 3 Timeline

```
Week 1–3:  SymmetryTensor (3.1)
           − Autocorrelation-based periodicity detection
           − Log-frequency symmetry detection
           − Adaptive harmonic binding
           − Instrument classification benchmark

Week 4–7:  DisentangledTensor (3.2)
           − TC-VAE encoder/decoder
           − MINE MI estimator
           − Style transfer validation
           − DCI disentanglement score

Week 8–10: TemporalRelationTensor (3.3)
           − Allen relation extractor (hard rules)
           − Soft RelationNet classifier
           − Narrative inference module
           − Integration with S3 attractor outputs
```

**Phase 3 success criteria:**
- SymmetryTensor correctly classifies cyclic vs dihedral vs trivial at > 85% on held-out set
- DisentangledTensor TC score < 1.0 (vs unregularised baseline > 3.0)
- Allen relations correctly identify 8/13 relation types at > 80% on synthetic test
- Full 7-layer semantic coverage score computable for any input

---

## Phase 4: LLM Integration and Production

**Duration:** 8 weeks  
**Goal:** Integrate all layers into LLM adapters, deploy production system, establish evaluation suite

---

### 4.1 Refined LLM Adapter

The `BifrostEnhancedLLM` is already implemented with three modes. Phase 4 refines these with new structural signals.

**Mode 1: Spectral Prefix with Causal Context**
```
[S0→S1(hier)→S2→S3→S4] → [PredictiveErrorTensor, CausalGraphTensor, PersistenceTensor]
    → concat structural embeddings → project → LLM prefix tokens
```
The prefix now carries:
- Temporal structure (which timescale SSM is most active)
- Causal structure (which bands predict which)
- Topological structure (Betti numbers)
- Prediction-error profile (where boundaries are)

**Mode 2: Structural Coherence Verifier (production-ready)**
```python
class StructuralCoherenceVerifier:
    """
    Model-agnostic reasoning quality signal.
    
    For each reasoning step step_i in a chain-of-thought:
    
    1. Encode problem_context via Bifrost → SpectralTensor P
    2. Encode step_i via Bifrost → SpectralTensor S_i
    3. Compute:
       - phase_lock_score(P, S_i): structural consistency
       - causal_alignment(P, S_i): causal direction consistency
       - topological_distance(P, S_i): manifold distance
       - prediction_error_spike(S_i): surprise at step boundary
    
    4. Coherence score = f(phase_lock, causal, topological, error)
    5. Flag step_i if score < threshold → regenerate
    
    This is grounded in structural physics, not a learned reward model.
    It cannot be gamed by reward hacking.
    """
```

**Mode 3: Parameter-Efficient Adapter**
```
Adapter parameters: SpectralProjector + ComplexSSM (injection at layer 6 of LLM)
Total trainable params: ~5M (vs GPT-2: 124M → 4% overhead)
Target: perplexity reduction on structured text (proofs, code, narrative)
```

### 4.2 Evaluation Framework: Seven-Layer Semantic Score

```python
@dataclass
class SevenLayerSemanticScore:
    """
    Composite score quantifying coverage of all 7 semantic layers.
    Range: [0, 1] for each layer; composite = geometric mean.
    """
    L1_distributional: float       # spectral co-occurrence entropy
    L2_compositional: float        # cross-level hierarchical SSM consistency
    L3_causal: float               # Granger graph edge precision vs ground truth
    L4_topological: float          # Betti number stability across paraphrase pairs
    L5_temporal: float             # Allen relation accuracy on ordered pairs
    L6_symmetry: float             # symmetry group classification accuracy
    L7_disentanglement: float      # DCI disentanglement score (normalised)
    
    composite: float               # geometric mean of [L1..L7]
    
def compute_seven_layer_score(
    model: BifrostPipeline,
    eval_dataset: MultimodalEvalDataset,
) -> SevenLayerSemanticScore: ...
```

**Evaluation datasets per layer:**
| Layer | Dataset | Metric |
|---|---|---|
| L1 | AudioSet, COCO | Spectral retrieval precision@10 |
| L2 | Switchboard word boundaries | Boundary F1 (per-level SSM) |
| L3 | EEG causal ground truth | GC edge precision |
| L4 | VGGSound paraphrase pairs | Betti stability across pairs |
| L5 | TimeBank temporal relations | Allen relation accuracy |
| L6 | TUT Sound Events (periodic) | Symmetry group accuracy |
| L7 | dSprites factors | DCI disentanglement |

### 4.3 Production Deployment

**Inference optimization:**
- Quantize complex SSM weights: INT8 for amplitude paths, FP16 for phase paths
- Cache attractor codebook in GPU SRAM (65K × d_state = ~10MB)
- Streaming inference: stateful SSM with h_0 = previous chunk output
- TDA: precompute persistence diagrams for reference set (no online computation)

**Latency targets (per 1 second of audio, A100 GPU):**
```
S0 canonicalization:      < 5ms
S1 hierarchical SSM:      < 15ms (5 levels, parallelised)
S2 spectral binding:      < 10ms
S3 attractor learning:    < 20ms (VQ lookup dominates)
S4 Riemannian coherence:  < 10ms (pre-built geodesic graph)
S5 causal graph (fast):   < 5ms
S5 TDA (cached):          < 5ms
Total pipeline:           < 70ms  (> 14x real-time on 1 audio sec)
```

**API endpoints:**
```
POST /encode          → SpectralTensor + all structural tensors
POST /coherence       → phase-lock score for a pair
POST /boundary        → predictive error boundary scores
POST /causal          → causal graph for signal
POST /verify_reasoning → coherence score for reasoning step
GET  /health          → pipeline health + GPU utilisation
```

---

## Data Architecture

### SpectralTensor (current, unchanged)
```python
@dataclass
class SpectralTensor:
    amplitude: torch.Tensor   # (B, T, F) magnitude of spectral components
    phase: torch.Tensor       # (B, T, F) phase angle of spectral components
    scale: float              # normalisation factor
    uncertainty: torch.Tensor # (B, T, F) per-element uncertainty (calibrated)
    metadata: dict            # modality, sample_rate, domain, attractor_count, ...
```

### Extended Pipeline Return Type (Phase 4)
```python
@dataclass
class BifrostOutput:
    # Core spectral
    spectral: SpectralTensor
    coherence: torch.Tensor             # (B, T) phase coherence per timestep
    
    # Phase 1
    attractors: list[FrequencyAttractor]
    riemannian_distance: torch.Tensor   # (B, n_attractors) geodesic distances
    
    # Phase 2
    predictive_error: PredictiveErrorTensor
    causal_graph: CausalGraphTensor
    persistence: PersistenceTensor
    hierarchical_states: list[torch.Tensor]  # one per SSM level
    
    # Phase 3
    symmetry: SymmetryTensor
    disentangled: DisentangledTensor
    temporal_relations: TemporalRelationTensor
    
    # Scores
    seven_layer_score: Optional[SevenLayerSemanticScore]
```

---

## Known Limitations and Mitigations

| Limitation | Impact | Mitigation | Phase |
|---|---|---|---|
| Phase grounding breaks at VQ-VAE discrete bottleneck | Riemannian geometry operates on semantically arbitrary tokens | Add PredictiveError signal before VQ; Fisher metric baseline | 2 |
| Diagonal complex A matrix misses cross-frequency interaction | Granger causality requires off-diagonal structure | Fast mode via B matrix; optional full A in mode="exact" | 2 |
| TDA is O(n³) for n points; expensive for long sequences | Latency budget exceeded for real-time use | Landmark TDA (subsample + Witness complex) | 2 |
| TC-VAE requires many samples to estimate TC accurately | Small batch → noisy MI estimates | Minibatch stratified estimator (Chen et al., 2018) | 3 |
| Allen algebra discretises continuous activations | Soft activations produce ambiguous intervals | Learnable soft classifier with tolerance ε | 3 |
| No ground truth for Granger structure in audio | Cannot evaluate GC precision without ground truth labels | Use EEG datasets with known causal structure as proxy | 4 |

---

## Risk Register

| Risk | Probability | Severity | Mitigation |
|---|---|---|---|
| Phase coherence signal too weak on text tokens | Medium | High | Validate on CharacterBERT / byte-level LM |
| VQ-VAE codebook collapse (few active codes) | Medium | High | EMA updates + commitment loss + temperature annealing |
| TDA too slow for 3PB training corpus | Low | Medium | Offline TDA during preprocessing; cache diagrams |
| Granger causality degenerate (all zeros) | Low | Medium | Add minimum variance regulariser; check B matrix norms |
| Hierarchical SSM memory explosion (5 levels × batch) | Medium | Medium | Gradient checkpointing per level |

---

## Compute Budget

| Phase | Compute | Storage | Engineers | Duration |
|---|---|---|---|---|
| Phase 1 | $50K (8x A100 × 8 weeks) | 3PB | 3 | 8 weeks |
| Phase 2 | $40K (8x A100 × 10 weeks) | +500TB (cached TDA) | 3 | 10 weeks |
| Phase 3 | $60K (TC-VAE needs more data) | +200TB | 4 | 10 weeks |
| Phase 4 | $30K (fine-tuning + eval) | minimal | 3 | 8 weeks |
| **Total** | **$180K** | **~4PB** | **4 peak** | **36 weeks** |

---

## Implementation Priority Queue

Ordered by impact/effort ratio:

1. **PredictiveErrorTensor** (3 days) — zero-shot boundary detection, free from existing SSM
2. **TDA Persistence** (1 week) — parameter-free, validates L4 coverage claim
3. **GrangerCausality fast mode** (1 week) — first directed signal in pipeline
4. **HierarchicalSSM** (3 weeks) — addresses L2, composes existing SSM modules
5. **SymmetryTensor** (2 weeks) — generalises HarmonicBinding, no training needed
6. **DisentangledTensor** (4 weeks) — complex but high-value for generation and transfer
7. **TemporalRelationTensor** (2 weeks) — Allen hard rules first; soft classifier second
8. **SevenLayerSemanticScore** (1 week) — composite evaluation once all layers exist

---

## Appendix: Mathematical Reference

**Complex SSM recurrence:**
```
h[t] ∈ ℂ^{d_state},  A ∈ ℂ^{d_state×d_state} (diagonal)
h[t] = exp(−Δ·A)·h[t−1] + Δ·B[t]·x[t]
y[t] = Re(C·h[t]) + D·x[t]
```

**Phase coherence (multi-band):**
```
C(i,j) = Σ_{b=1}^{B} w_b · (1/F) Σ_{f} cos(φ_q[i,f,b] − φ_k[j,f,b])
```

**Riemannian metric (Cholesky):**
```
G(x) = L(x)·L(x)ᵀ,   L lower triangular, det(G) > 0
geodesic distance ≈ Dijkstra on k-NN graph with edge weight √(xᵀG(m)x)
```

**Granger causality:**
```
GC(i→j) = log( Var[ê_j^{(−i)}] / Var[ê_j] )
ê_j = residual of AR model for band j
ê_j^{(−i)} = residual of restricted AR model excluding band i's history
```

**TDA (Vietoris-Rips):**
```
VR(ε) = simplicial complex with σ ∈ VR(ε) ↔ diam(σ) ≤ ε
β_k(ε) = k-th Betti number of VR(ε)
persistence of feature = ε_death − ε_birth
```

**Total correlation:**
```
TC(z) = KL( q(z) || Π_i q(z_i) )
       = Σ_{i=1}^{d} H(z_i) − H(z)
```

**Allen relation (before):**
```
before(A, B) ↔ offset(A) + ε < onset(B)
meets(A, B) ↔ |offset(A) − onset(B)| ≤ ε
```

**Adler equation (phase-lock):**
```
dφ/dt = Δω + K·sin(φ_target − φ)
phase-locked ↔ |Δω| < K
```
