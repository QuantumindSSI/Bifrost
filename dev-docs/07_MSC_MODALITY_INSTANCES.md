# 07 — Modality-Specific MSC Instances

**Status**: Implementation specifications  
**Goal**: Define the concrete MSC instance for each modality (audio, image, sensor, text).

---

## Overview

Each modality has a specific MSC instance that implements the general formula:

```
MSC(x, s) = |Σ_b A_b(s) · exp(i · φ_b(s))| / Σ_b A_b(s)
```

The instances differ in:
- What constitutes a "band" (b)
- What constitutes a "scale" (s)
- How the amplitude A_b(s) and phase φ_b(s) are extracted
- What the coherence measure captures semantically

---

## Instance 1: Audio — CBMPC (VALIDATED)

**Status**: Implemented and validated on SpeechCommands.

### Bands (b)
Mel frequency bands (n_mels = 64). Each band corresponds to a frequency region of the cochlea.

### Scales (s)
Modulation frequencies: [0.5, 1, 2, 4, 8, 16, 32] Hz. These correspond to the temporal rates at which spectral energy changes (syllable rate, onset rate, etc.).

### Amplitude and phase extraction
1. Compute STFT of the audio signal.
2. Apply mel filterbank to get mel spectrogram S(t, f).
3. Log-compress: L(t, f) = log(|S(t, f)| + ε).
4. For each mel band f, compute temporal FFT: L̃(ω, f) = FFT_t{L(t, f)}.
5. A_b(s) = |L̃(s, b)|, φ_b(s) = ∠L̃(s, b).

### Coherence measure
Cross-band phase locking value: C(s) = |(1/B) Σ_b exp(i · φ_b(s))|.

### Feature vector
[per-band modulation amplitudes A_b(s), PLV values C(s), mean amplitudes] — 462 dimensions in "rich" mode.

### Semantic meaning
- Low modulation frequencies (0.5-4 Hz): syllable rate, vowel structure.
- Mid modulation frequencies (4-16 Hz): onset patterns, consonant transitions.
- High modulation frequencies (16-32 Hz): pitch, fine temporal structure.
- Cross-band coherence: formant transitions, manner of articulation.

### Validated results
- SpeechCommands (10 classes): 0.41 ± 0.04 accuracy, +13.65 pp over STFT baseline (p = 0.0033).
- ESC-50 (50 classes): 0.12 ± 0.02 accuracy, NOT supported (speech-specific).

### Implementation
`src/bifrost/cbmpc.py` — `CBMPCExtractor` class.

---

## Instance 2: Image — Phase Congruency (TO IMPLEMENT)

**Status**: Specification only. Not yet implemented.

### Bands (b)
Orientation bands (e.g., 6 orientations: 0°, 30°, 60°, 90°, 120°, 150°). Each band corresponds to a preferred edge orientation.

### Scales (s)
Spatial frequency scales (e.g., 5 octaves: σ = 1, 2, 4, 8, 16 pixels). Each scale corresponds to a level of detail.

### Amplitude and phase extraction
1. Compute a steerable pyramid or log-Gabor wavelet decomposition of the image.
2. For each scale s and orientation b, extract the filter response: R_b(s, x, y).
3. A_b(s, x, y) = |R_b(s, x, y)|, φ_b(s, x, y) = ∠R_b(s, x, y).

### Coherence measure
Phase congruency at each pixel:
```
PC(x, y) = |Σ_{s,b} W_{s,b} · A_{s,b}(x,y) · cos(φ_{s,b}(x,y) - φ̄(x,y))| / Σ_{s,b} A_{s,b}(x,y)
```
where φ̄(x,y) is the weighted mean phase angle at pixel (x,y).

### Feature vector
- Global: [mean PC, std PC, PC histogram (16 bins), per-scale mean PC (5 values), per-orientation mean PC (6 values)] — ~30 dimensions.
- Local: PC map (downsampled to 8×8 = 64 dimensions).
- Total: ~94 dimensions.

### Semantic meaning
- Phase congruency detects edges, corners, and junctions — features where Fourier components are maximally in phase.
- Per-scale PC captures structure at different levels of detail (fine edges vs. coarse boundaries).
- Per-orientation PC captures the distribution of edge orientations (texture, shape).
- Illumination and contrast invariant — captures intrinsic structure.

### Datasets for validation
- CIFAR-10 (10 classes, 32×32 images) — simple classification.
- ImageNet-100 (100 classes, 224×224 images) — harder classification.
- MNIST (10 digit classes, 28×28) — simple baseline.

### Hypothesis
Phase congruency features will outperform raw pixel and raw FFT magnitude baselines on image classification, particularly for datasets where edge structure is discriminative (shapes, textures).

### Implementation plan
`src/bifrost/msc_image.py` — `PhaseCongruencyExtractor` class using log-Gabor filters.

---

## Instance 3: Sensor — Wavelet Coherence (TO IMPLEMENT)

**Status**: Specification only. Not yet implemented.

### Bands (b)
Sensor channels (e.g., for IMU: accelerometer x/y/z, gyroscope x/y/z = 6 channels). Each band is a sensor channel.

### Scales (s)
Wavelet scales (e.g., 7 scales corresponding to periods of 0.5, 1, 2, 4, 8, 16, 32 seconds). Each scale captures structure at a different temporal granularity.

### Amplitude and phase extraction
1. For each sensor channel b, compute the continuous wavelet transform: W_b(s, t) = ∫ x_b(τ) ψ*((τ-t)/s) dτ.
2. A_b(s, t) = |W_b(s, t)|, φ_b(s, t) = ∠W_b(s, t).

### Coherence measure
Cross-sensor wavelet coherence at each scale:
```
C(s) = |(1/B) Σ_b exp(i · φ_b(s, t_mean))|
```
where t_mean is the temporal average (or the coherence can be computed at each time step and then averaged).

Additionally, pairwise wavelet coherence between sensor pairs:
```
R^2_{b1,b2}(s) = |S(s^{-1} W_{b1}(s) W*_{b2}(s))|^2 / (S(s^{-1}|W_{b1}|^2) S(s^{-1}|W_{b2}|^2))
```

### Feature vector
- Per-scale cross-sensor PLV: 7 values.
- Per-scale per-sensor wavelet power: 7 × 6 = 42 values.
- Pairwise coherence summary (mean and std across pairs): 7 × 2 = 14 values.
- Total: ~63 dimensions.

### Semantic meaning
- Low scales (fast periods): transient events, impacts, sudden changes.
- Mid scales: rhythmic patterns, gait cycles, vibration modes.
- High scales (slow periods): trends, drift, long-term dynamics.
- Cross-sensor coherence: coordinated motion (e.g., walking = coherent accelerometer and gyroscope), vs. uncoordinated noise.

### Datasets for validation
- UCI HAR (Human Activity Recognition, 6 activities, smartphone IMU) — standard benchmark.
- PAMAP2 (Physical Activity Monitoring, 18 activities, multiple IMUs) — harder benchmark.

### Hypothesis
Wavelet coherence features will outperform raw statistical features (mean, std, etc.) on activity recognition, particularly for activities with characteristic temporal patterns (walking, cycling) vs. static activities (sitting, standing).

### Implementation plan
`src/bifrost/msc_sensor.py` — `WaveletCoherenceExtractor` class using PyWavelets.

---

## Instance 4: Text — Graph Spectral Coherence (TO IMPLEMENT)

**Status**: Specification only. Not yet implemented. Most speculative instance.

### Bands (b)
Syntactic roles in a dependency graph (e.g., subject, object, verb, modifier, root = 5 roles). Each band corresponds to a syntactic function.

### Scales (s)
Compositional depth in the dependency tree (e.g., depth 1 = immediate dependencies, depth 2 = grandparent dependencies, depth 3 = great-grandparent, etc. = 4 scales).

### Amplitude and phase extraction
1. Parse the text into a dependency graph.
2. Construct a graph Laplacian: L = D - A.
3. Compute spectral graph wavelets at each scale: ψ_s = U · diag(g(s·λ_i)) · U^T, where U are eigenvectors of L and g is a wavelet kernel.
4. For each syntactic role b and scale s, compute the wavelet response on the nodes with that role: R_b(s) = ψ_s · f_b, where f_b is the embedding vector for nodes with role b.
5. A_b(s) = |R_b(s)|, φ_b(s) = ∠R_b(s) (in the complex wavelet case).

### Coherence measure
Cross-role phase locking at each compositional depth:
```
C(s) = |(1/B) Σ_b exp(i · φ_b(s))|
```

### Feature vector
- Per-scale cross-role PLV: 4 values.
- Per-scale per-role wavelet power: 4 × 5 = 20 values.
- Graph spectral properties (Fiedler value, spectral gap, clustering coefficient): 3 values.
- Total: ~27 dimensions.

### Semantic meaning
- Low compositional depth: local word-level structure (subject-verb agreement, noun-adjective modification).
- Mid depth: phrase-level structure (verb phrase, noun phrase coherence).
- High depth: sentence-level structure (clause relationships, long-range dependencies).
- Cross-role coherence: how well the syntactic components fit together compositionally.

### Datasets for validation
- SST-2 (Stanford Sentiment Treebank, binary sentiment) — simple classification.
- TREC (question type classification, 6 types) — medium difficulty.
- MNLI (natural language inference, 3 classes) — hard, requires compositional understanding.

### Hypothesis
Graph spectral coherence features will outperform bag-of-words and mean word embedding baselines on tasks that require compositional understanding (NLI), but may not improve on tasks that can be solved with lexical cues (sentiment).

### Implementation plan
`src/bifrost/msc_text.py` — `GraphSpectralCoherenceExtractor` class using spaCy for parsing and PyGSP or custom implementation for graph wavelets.

---

## Cross-modal alignment

Once all four instances are implemented and validated, the MSC embeddings from different modalities can be aligned because they share the same mathematical structure:

1. **Common structure**: All MSC embeddings contain per-scale coherence values and per-scale per-band amplitudes.
2. **Scale alignment**: The scale axes can be normalized (e.g., log-spaced scales mapped to a common [0, 1] range).
3. **Coherence as universal similarity**: Two inputs from different modalities with similar coherence profiles (e.g., high coherence at low scales, low coherence at high scales) may share structural properties.

### Cross-modal retrieval task
Given a query from modality A (e.g., an audio clip), retrieve the most similar item from modality B (e.g., an image) based on MSC embedding similarity.

### Pre-registered hypothesis
MSC embeddings from different modalities will produce above-chance cross-modal retrieval on datasets with known audio-visual correspondence (e.g., AudioSet-ImageNet pairs, Flickr8k).

---

## Implementation priority

| Instance | Priority | Rationale |
|---|---|---|
| Audio (CBMPC) | Done | Already validated |
| Image (Phase Congruency) | 1 | Strongest literature support (Kovesi), clear mathematical parallel to CBMPC |
| Sensor (Wavelet Coherence) | 2 | Well-established technique (Grinsted), clear practical applications |
| Text (Graph Spectral) | 3 | Most speculative, requires NLP parsing infrastructure |

The image instance is the highest priority because:
1. Phase congruency is the most well-established phase-based structural feature in computer vision.
2. The Bifrost pipeline already has 2D FFT support.
3. Image classification benchmarks (CIFAR-10, ImageNet) are well-established.
4. The mathematical parallel between CBMPC (audio) and phase congruency (image) is the clearest cross-modal connection.
