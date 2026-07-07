# 09 — Research Paths Compendium

**Status**: Complete catalog of all research directions  
**Purpose**: Document every research path considered for the Bifrost project, with status, evidence, and assessment.

---

## Overview

The Bifrost project aims to establish **structural intelligence across data modalities** for AI, AGI, and ASI. This document catalogs every research path that has been considered, surveyed, or proposed, organized by status and category.

---

## A. Validated paths (empirical support exists)

### A1. CBMPC — Cross-Band Modulation Phase Coherence (audio/speech)

**What**: Extracts phase coherence of temporal modulations across mel frequency bands. Measures how synchronized the temporal dynamics of different frequency regions are at each modulation rate.

**Evidence**: SpeechCommands 10-class, 200 samples/class, 5-fold CV. CBMPC-STFT: 0.41 ± 0.04 accuracy. STFT baseline: 0.27 ± 0.01. Delta = +13.65 pp, p = 0.0033 (Bonferroni-corrected). Cohen's d ≈ 3.5.

**Limitation**: Speech-specific. Failed on ESC-50 environmental sounds (0.12 vs 0.16 STFT baseline).

**Files**: `src/bifrost/cbmpc.py`, `research_dir/experiment_cbmpc_comparison.py`, `research_dir/results/cbmpc_baseline_comparison.json`

---

## B. Investigated paths (implementation exists, results mixed)

### B1. Pre-SSM CBMPC integration

**What**: Extract CBMPC features from raw audio before the SSM, combine with SSM embedding via concatenation.

**Evidence**: 5-class pilot, 3-fold CV. CBMPC+SSM: 0.47 ± 0.04. CBMPC-only: 0.44 ± 0.08. Delta = +3.2 pp, p = 0.66 (not significant).

**Conclusion**: SSM does not destroy CBMPC in parallel, but adds no significant value for speech classification.

**Files**: `src/bifrost/pipeline.py` (use_cbmpc parameter), `research_dir/experiment_cbmpc_pre_ssm.py`

### B2. Modulation-preserving SSM architectures

**What**: Four candidate SSM architectures designed to preserve modulation structure:
- **Architecture C (Residual SSM)**: SSM output + α × mel projection. Implemented.
- **Architecture A (Band-wise SSM)**: Separate SSM per mel band. Implemented.
- **Architecture B (Modulation-domain SSM)**: SSM in modulation domain. Specified only.
- **Architecture D (Dual-path SSM)**: Parallel temporal + modulation paths. Specified only.

**Evidence**: Architectures C and A implemented but not validated. The pre-SSM pilot suggests the SSM adds no value on top of CBMPC.

**Files**: `src/bifrost/decomposer/modulation_preserving_ssm.py`

---

## C. Specified but unimplemented paths

### C1. MSC framework — image instance (phase congruency)

**What**: Detect image features (edges, corners) by measuring phase alignment across spatial frequency scales and orientations using log-Gabor filters. The image analog of CBMPC.

**Evidence**: Phase congruency is the most well-established phase-based image feature (Kovesi 1999, 2000+ citations). Validated extensively in computer vision literature but not in the Bifrost context.

**Status**: Implementation started (`src/bifrost/msc_image.py`), validation experiment written (`research_dir/experiment_msc_image_cifar10.py`), but CIFAR-10 download stalled and experiment not completed.

**Priority**: Highest — clearest test of cross-modal generalization.

### C2. MSC framework — sensor instance (wavelet coherence)

**What**: Measure cross-sensor wavelet coherence at multiple time scales for activity recognition. Captures coordinated motion across sensor channels.

**Evidence**: Wavelet coherence is a mature technique (Grinsted et al. 2004). Applied extensively in geophysics, neuroscience, and sensor fusion but not in the Bifrost context.

**Status**: Specified in `dev-docs/07_MSC_MODALITY_INSTANCES.md`. Not implemented.

**Priority**: Medium — validates MSC on a third modality.

### C3. MSC framework — text instance (graph spectral coherence)

**What**: Measure cross-role phase locking at different compositional depths in a dependency parse tree using graph spectral wavelets.

**Evidence**: Graph wavelets exist (Hammond et al. 2011) but the specific application to syntactic structure is novel and speculative.

**Status**: Specified only. Not implemented.

**Priority**: Lower — most speculative, requires NLP parsing infrastructure.

### C4. Cross-modal retrieval and alignment

**What**: Test whether MSC embeddings from different modalities can be aligned for cross-modal retrieval (audio → image, image → text, etc.).

**Evidence**: Pre-registered in `dev-docs/08_CROSS_MODAL_VALIDATION_PROTOCOL.md` but not tested. Requires at least 2 validated MSC instances.

**Priority**: After ≥ 2 modalities are validated.

### C5. Cross-modal structural similarity

**What**: Test whether inputs from different modalities with shared structural properties (e.g., rhythmic audio + rhythmic video) have more similar MSC coherence profiles than inputs without shared structure.

**Evidence**: Pre-registered but not tested. Requires controlled datasets with known cross-modal correspondence.

**Priority**: After cross-modal retrieval.

---

## D. The Seven-Layer Semantic Framework (L2–L7)

The original Bifrost vision proposed seven layers of semantic structure. Only L1 was (partially) implemented and it failed. The remaining six layers are planned with pre-registered hypotheses but zero implementation.

### D1. L2 — Compositional structure (Hierarchical SSM)

**What**: Hierarchical SSMs at multiple timescales (phoneme → syllable → word → phrase → discourse) capture compositional structure.

**Hypothesis**: Hierarchical SSM improves word boundary detection vs. flat SSM.

**Status**: Not implemented. Proposed in `SELF_DRIVING_RESEARCH_LOOP.md`.

### D2. L3 — Causal structure (Granger causality)

**What**: Neural Granger causality on spectral representations to recover cause-effect relationships in time series.

**Hypothesis**: Granger causality on Bifrost embeddings recovers known EEG causal edges vs. random graph.

**Status**: Not implemented. `PredictiveErrorTensor` and Granger fast mode mentioned in synthesis.

### D3. L4 — Topological structure (TDA)

**What**: Persistent homology on spectrograms/spectral embeddings to capture topological invariants (Betti numbers, persistence diagrams).

**Hypothesis**: TDA Betti numbers distinguish instrument families vs. random features.

**Status**: Not implemented. TDA surveyed in `LITERATURE_SURVEY.md`.

**Assessment**: TDA is the most modality-agnostic technique surveyed. It captures shape and connectivity, not phase coherence. Could complement the MSC framework.

### D4. L5 — Temporal structure (Allen relations)

**What**: Allen interval algebra (before, during, overlaps, contains) applied to temporal patterns in spectral representations.

**Hypothesis**: Allen relations recover temporal order vs. random relation.

**Status**: Not implemented. `TemporalRelationTensor` mentioned in synthesis.

### D5. L6 — Symmetry structure (SymmetryTensor)

**What**: Detect invariance under transformation (octave invariance in music, rotation invariance in images, translation invariance in time series).

**Hypothesis**: SymmetryTensor detects octave vs. non-octave invariance vs. fixed harmonic grid.

**Status**: Not implemented. EquiSym, EquiAV surveyed in literature.

### D6. L7 — Disentanglement (TC-VAE)

**What**: Separate independent factors of variation (speaker identity vs. content, timbre vs. pitch) using β-TC-VAE.

**Hypothesis**: TC-VAE separates speaker from content vs. standard VQ-VAE.

**Status**: Not implemented. β-VAE, β-TC-VAE surveyed in literature.

---

## E. Surveyed techniques (not implemented, not specified)

### E1. Frequency-level audio techniques

| Technique | Category | What it captures |
|---|---|---|
| MFCC / PLP | Spectral envelope | Cepstral coefficients, decorrelated features |
| Spectral centroid/rolloff/contrast/flatness | Spectral shape | Brightness, bandwidth, energy distribution |
| F0 / HNR / Inharmonicity | Harmonic structure | Pitch, voicing, harmonic deviation |
| Wavelet scattering transform | Time-frequency dynamics | Rotation/translation-invariant features via cascaded wavelets |
| Spectral flux / onset envelope | Time-frequency dynamics | Onset detection, temporal changes |
| Cochleagram / auditory spectrogram | Auditory model | Detailed cochlear simulation |
| CQT | Auditory model | Log-spaced frequency, variable Q |
| Chroma features | Harmonic structure | Pitch class distribution |
| Cross-frequency coupling (PAC, CFC) | Higher-order spectra | Phase-amplitude coupling across frequencies |
| Bispectrum / bicoherence | Higher-order spectra | Quadratic phase coupling |
| NMF | Decomposition | Spectrogram basis decomposition |
| Sparse coding | Decomposition | Sparse spectral representations |
| Source-filter model | Decomposition | Source/filter separation |
| Harmonic-percussive separation | Decomposition | Harmonic vs. percussive components |
| ICA | Decomposition | Independent components |
| Persistent homology on spectrograms | Topological | Topological invariants of spectral data |
| Spectral clustering | Topological | Cluster structure in frequency |
| Manifold learning | Topological | Low-dimensional spectral manifolds |
| Spectral autoencoder / VAE | Learned | Learned spectral representations |
| Contrastive spectrogram learning | Learned | Self-supervised spectral embeddings |
| Masked frequency modeling | Learned | Self-supervised frequency prediction |
| Fourier neural operator (FNO) | Learned | Learn spectral transformations |
| Spectral transformer | Learned | Attention in frequency domain |

### E2. Complex-valued neural networks

| Technique | Citation | What it does |
|---|---|---|
| Deep Complex Networks | Trabelsi et al. 2018 | Complex-valued CNNs, batch norm, activations |
| Phase-aware deep learning | Pham et al. 2025 | Complex-valued CNNs for phase-aware processing |
| Hybrid real-complex networks | Paul & Nelson 2023 | Mixed real/complex architecture |
| Complex Transformer | Muqiaoyu et al. 2019 | Complex-valued attention |

### E3. State space model variants

| Technique | Citation | What it does |
|---|---|---|
| S4 | Gu, Goel & Ré 2022 | Structured state spaces for long sequences |
| Mamba | Gu & Dao 2023 | Selective SSMs (partially integrated) |
| S5 | Smith et al. 2023 | Simplified structured state spaces |

### E4. Topological data analysis

| Technique | Citation | What it does |
|---|---|---|
| Homological persistence | Bennet et al. 2020 | Topological persistence in time series |
| Time delay embeddings for timbre | Zan et al. 2026 | TDA on time-delay embeddings |
| Topological fingerprint of music | Bergomi et al. 2016 | Persistent homology for music style |
| Topological fingerprints for audio | Chen et al. 2023 | Audio identification via topology |

### E5. Symmetry detection

| Technique | Citation | What it does |
|---|---|---|
| EquiSym | Seo et al. 2022 | Equivariant symmetry detection |
| EquiAV | Devillers & Lefort 2024 | Equivariant audio-visual learning |
| Group invariants | Zimmermann et al. 2022 | Structuring representations via group invariants |
| E3Sym | Li et al. 2023 | 3D Euclidean symmetry |
| Group-convolutional networks | Dieleman et al. 2014 | Rotation-equivariant convolutions |

### E6. Disentanglement

| Technique | Citation | What it does |
|---|---|---|
| β-VAE | Higgins et al. 2017 | Disentangled VAE |
| β-TC-VAE | Chen et al. 2018 | Total correlation VAE |
| Disentangled speech | Wu et al. 2022 | Speaker/content separation |
| Self-supervised multi-view | Li et al. 2024 | Multi-view music disentanglement |
| MERIT | Kumar et al. 2024 | Multi-view disentanglement |

### E7. Cross-modal learning

| Technique | Citation | What it does |
|---|---|---|
| CLIP | Radford et al. 2021 | Contrastive language-image pretraining |
| ImageBind | Girdhar et al. 2023 | Binds 6 modalities into one embedding |
| CoAVT | Xu et al. 2024 | Coarse-to-fine audio-visual transformer |
| WAVE | Tang et al. 2025 | Web-scale audio-visual evaluation |
| OmniRetriever | Li et al. 2024 | Any-to-any retrieval |

### E8. Multimodal LLM grounding

| Technique | Citation | What it does |
|---|---|---|
| CoRGI | Liu et al. 2024 | Counterfactual reasoning for grounding |
| PostAlign | Wang et al. 2024 | Post-hoc alignment for LLMs |
| Grounded CoT | Chen et al. 2024 | Grounded chain-of-thought |
| Grounded verification | Whitehouse et al. 2024 | Multimodal judgment verification |
| VBackChecker | Zhang et al. 2024 | Visual back-checking for LLMs |

---

## F. Abandoned paths (with reasons)

| Direction | Reason for abandonment |
|---|---|
| Raw spectral phase coherence (L1) | Never tested (phase discarded in embedding); theoretically misdirected — modulation phase carries semantic structure, not spectral phase |
| Bifrost amplitude-only embedding | Fundamentally flawed — discarded phase, collapsed temporal dimension |
| CBMPC-Bifrost (post-SSM) | SSM destroys modulation structure (accuracy drops to chance 0.10) |
| CBMPC for environmental sounds | Speech-specific — failed on ESC-50 (0.12 vs 0.16 STFT baseline) |
| Token-based approaches | Explicitly rejected in favor of continuous spectral representations |
| Mean+std temporal pooling | Destroys temporal dynamics that carry semantic structure |

---

## G. Applications beyond audio

| Application | Modality | Description | Status |
|---|---|---|---|
| EEG/ECG analysis | Sensor | Process neural/medical signals in frequency domain | Mentioned |
| Industrial vibration | Sensor | Anomaly detection in machine sensor data | Mentioned |
| Sonar/radar | Sensor | Active sensing signal processing | Mentioned |
| IMU activity recognition | Sensor | Human activity recognition from IMU | Specified (MSC sensor) |
| Cross-modal retrieval | Multi | Audio ↔ image ↔ text retrieval | Pre-registered |
| LLM grounding | Text | Structural verification of LLM outputs | Proposed (L7) |
| Music analysis | Audio | Instrument classification, timbre | Surveyed |
| Speech disentanglement | Audio | Speaker/content separation | Proposed (L7) |
| Analogical reasoning | Multi | Riemannian manifold geodesic as analogy | Mentioned |
| Image classification | Image | CIFAR-10, ImageNet via phase congruency | Specified (MSC image) |

---

## H. Priority assessment

Based on the evidence and theoretical grounding:

| Priority | Path | Rationale |
|---|---|---|
| 1 | MSC image (phase congruency) on CIFAR-10 | Clearest cross-modal test; strongest literature support |
| 2 | MSC sensor (wavelet coherence) on UCI HAR | Validates MSC on third modality; mature technique |
| 3 | TDA (L4) on audio | Most modality-agnostic; captures complementary structure |
| 4 | Frequency-level data model investigation | Structure may be inherent in the representation itself |
| 5 | Cross-modal retrieval | Tests MSC unification; requires ≥ 2 validated instances |
| 6 | Disentanglement (L7) | Separating factors of variation for AGI |
| 7 | Compositional SSM (L2) | Hierarchical structure for AGI |
| 8 | Symmetry detection (L6) | Invariance for generalization |
| 9 | Causal structure (L3) | Causal understanding for AGI |
| 10 | Text MSC (graph spectral) | Most speculative; requires NLP infrastructure |
