# 06 — Multi-Scale Structural Coherence (MSC) Framework

**Status**: Theoretical framework (unifying principle)  
**Goal**: Define the general principle that unifies structural intelligence across data modalities.

---

## The problem

The Bifrost project's driving goal is to establish **structural intelligence across different data modalities** — audio, images, text, sensor data. The CBMPC technique validated a specific instance of this for speech audio (modulation phase coherence across frequency bands), but:

1. CBMPC is **speech-specific** — it failed on environmental sounds (ESC-50).
2. CBMPC is **audio-specific** — it cannot be applied to images, text, or sensor data.
3. The Bifrost pipeline has multimodal infrastructure (4 modalities declared, 2D FFT, cross-modal bridge) but it is **unvalidated** and **audio-centric**.

The question is: **what is the general principle that CBMPC is a specific instance of?**

## The answer: Multi-Scale Structural Coherence (MSC)

A literature survey of 10 techniques and 30+ peer-reviewed papers reveals that the unifying principle is **Multi-Scale Phase Coherence** — the alignment (coherence) of oscillatory components across multiple scales of analysis encodes semantic meaning across all modalities.

### Core principle

**Semantic meaning across modalities is encoded in the coherence of oscillatory components across multiple scales of analysis.**

This principle manifests differently in each modality:

| Modality | Oscillatory component | Scale axis | Coherence measure | Instance |
|---|---|---|---|---|
| Audio (speech) | Temporal modulations of spectral energy | Modulation frequency | Cross-band phase locking value | **CBMPC** (validated) |
| Images | Spatial frequency components | Spatial frequency scale | Phase congruency across scales | **Phase Congruency** (Kovesi 1999) |
| Time series/sensors | Wavelet components | Time scale | Wavelet coherence across sensors | **Wavelet Coherence** (Grinsted 2004) |
| Text/graphs | Graph spectral components | Graph scale | Spectral graph wavelet coherence | **Graph Wavelet Coherence** (Hammond 2011) |

### Mathematical formulation

The general MSC measure at scale s is:

```
MSC(x, s) = |Σ_b A_b(s) · exp(i · φ_b(s))| / Σ_b A_b(s)
```

where:
- `b` indexes the bands/channels at scale `s`
- `A_b(s)` is the amplitude of the oscillatory component in band `b` at scale `s`
- `φ_b(s)` is the phase of the oscillatory component in band `b` at scale `s`

This is a **weighted phase locking value** across bands at each scale. It ranges from 0 (completely incoherent) to 1 (perfectly phase-locked).

**CBMPC is the audio instance**: `b` = mel frequency bands, `s` = modulation frequency, `A_b(s)` = modulation amplitude, `φ_b(s)` = modulation phase.

**Phase congruency is the image instance**: `b` = orientation bands, `s` = spatial frequency scale, `A_b(s)` = filter response amplitude, `φ_b(s)` = filter response phase.

**Wavelet coherence is the sensor instance**: `b` = sensor channels, `s` = wavelet scale, `A_b(s)` = wavelet power, `φ_b(s)` = wavelet phase.

### Why this is the right unifying principle

1. **Phase carries structural information** (Oppenheim & Lim, 1981). Phase alone can reconstruct the structure of a signal; amplitude alone cannot. This is true for audio, images, and any signal with oscillatory structure.

2. **Multi-scale analysis is essential** (Koenderink, 1984; Lindeberg, 1998). Structure exists at specific scales. A single scale cannot capture the full structural richness of any modality.

3. **Coherence across scales/bands detects meaningful features** (Kovesi, 1999; Grinsted, 2004). Features (edges in images, syllables in speech, events in sensor data) correspond to moments where phase is coherent across scales.

4. **This principle generalizes across modalities**. Audio, images, text, and sensor data all have oscillatory/spectral structure that can be decomposed into multi-scale components with amplitude and phase.

5. **The Bifrost pipeline already implements this principle** in its audio path (complex spectra → SSM → phase-lock binding). The framework extends this to other modalities.

### What MSC is not

- MSC is not a single algorithm. It is a **framework** with modality-specific instances.
- MSC is not raw spectral phase. It is **modulation/structural phase** — the phase of oscillatory components at specific scales, not the instantaneous phase of individual frequency components.
- MSC is not a replacement for the Bifrost pipeline. It is a **feature extraction layer** that provides the structural coherence features the pipeline should operate on.
- MSC is not limited to 1D signals. The framework extends to 2D (images), graph-structured (text), and multi-channel (sensor) data.

## The MSC hierarchy

The MSC framework defines a hierarchy of structural coherence:

### Level 0: Raw signal
The raw data in each modality (audio waveform, image pixels, text tokens, sensor readings).

### Level 1: Spectral decomposition
Decompose the raw signal into oscillatory components at multiple scales:
- Audio: STFT → mel spectrogram → modulation spectrum
- Images: Steerable pyramid / wavelet decomposition → multi-scale oriented coefficients
- Sensors: Wavelet transform → multi-scale wavelet coefficients
- Text: Graph construction → spectral graph wavelet decomposition

### Level 2: Phase and amplitude extraction
For each scale and band, extract the amplitude A_b(s) and phase φ_b(s) of the oscillatory component.

### Level 3: Cross-band coherence
Compute the MSC measure: the weighted phase locking value across bands at each scale.

### Level 4: Semantic embedding
The MSC features (coherence values at each scale, plus per-band amplitudes) form a semantic embedding that captures the structural intelligence of the input.

### Level 5: Cross-modal alignment
MSC embeddings from different modalities can be aligned because they share the same mathematical structure (coherence values at multiple scales). This enables cross-modal retrieval, transfer, and fusion.

## Relationship to the Bifrost pipeline

The Bifrost pipeline already implements parts of this framework:

| MSC level | Bifrost component | Status |
|---|---|---|
| Level 0: Raw signal | Input signal | Implemented (audio, image, text, tensor) |
| Level 1: Spectral decomposition | SpectralCanonicalizer | Implemented (1D FFT, 2D FFT) |
| Level 2: Phase and amplitude | SpectralTensor (amplitude + phase) | Implemented |
| Level 3: Cross-band coherence | ComplexSpectralDecomposer (SSM) | **Broken** — SSM destroys modulation structure |
| Level 4: Semantic embedding | Pipeline output | **Flawed** — amplitude-only embedding discards phase |
| Level 5: Cross-modal alignment | PhaseLockBridge | **Unvalidated** |

The CBMPC technique fixes Levels 3-4 for audio by computing cross-band coherence directly from the modulation spectrum, bypassing the broken SSM path. The MSC framework extends this fix to all modalities.

## The path forward

1. **Implement MSC instances for each modality** (audio: CBMPC done; images: phase congruency; sensors: wavelet coherence; text: graph wavelet coherence).
2. **Validate each instance** with pre-registered protocols on modality-specific benchmarks.
3. **Test cross-modal alignment** — can MSC embeddings from different modalities be aligned for cross-modal retrieval?
4. **Integrate MSC into the Bifrost pipeline** as the Level 3-4 feature extraction layer, replacing the broken SSM path.
5. **Rebuild the cross-modal bridge** on MSC embeddings instead of raw attractor phase-locking.

## Theoretical grounding

| Reference | Contribution to MSC |
|---|---|
| Oppenheim & Lim (1981) | Phase carries structural information in signals |
| Morrone & Burr (1988) | Phase-dependent energy model for feature detection in vision |
| Kovesi (1999) | Phase congruency as image feature detector (image instance of MSC) |
| Grinsted et al. (2004) | Wavelet coherence for cross-scale phase relationships (sensor instance) |
| Chi et al. (1999) | Spectrotemporal modulation transfer functions (audio instance) |
| Greenberg & Arai (2001) | Complex modulation spectrum and speech intelligibility |
| Elhilali et al. (2009) | Temporal coherence across frequency bands organizes auditory perception |
| Freeman & Adelson (1991) | Steerable filters for multi-scale oriented decomposition |
| Hammond et al. (2011) | Wavelets on graphs (text/graph instance of MSC) |
| Chung (1997) | Spectral graph theory (mathematical foundation for graph instance) |
| Koenderink (1984) | Scale-space theory (multi-scale structure) |
| Lachaux et al. (1999) | Phase locking value as a measure of synchrony |
