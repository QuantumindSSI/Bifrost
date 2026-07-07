# 10 — Frequency-Level Data Models and Inherent Structure

**Status**: Research investigation  
**Goal**: Determine whether semantic structure is inherent in the choice of frequency representation itself, not just in the data being represented.

---

## The hypothesis

**The choice of frequency representation imposes structural priors that make certain kinds of semantic structure visible while hiding others. The data model is not neutral — it is a form of structural intelligence.**

If this is true, then the Bifrost project's choice of frequency representation is itself a research decision that determines what semantic structure the system can discover. The question is: which frequency representation's structural prior is most aligned with semantic structure across modalities?

---

## The survey

A comprehensive survey of 8 categories of frequency-level data models was conducted, covering 30+ specific representations. For each, we analyzed:
- The mathematical structure (basis/decomposition)
- The structural prior it imposes (what structure is made visible)
- What structure it hides or destroys
- The time-frequency tiling pattern
- Whether it preserves phase
- Whether it is invertible
- What invariances it has

### Category 1: Linear time-frequency representations

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| STFT | Uniform | Yes | Yes | Stationary components within window |
| Gabor | Uniform (optimal) | Yes | Yes | Optimal time-frequency concentration |
| Windowed Fourier | Uniform | Yes | Yes | Window-dependent trade-offs |

**Key insight**: Uniform time-frequency tiling makes stationary structure visible but hides non-stationary and multi-scale structure. The uncertainty principle (Δt · Δf ≥ 1/4π) is a fundamental limit.

### Category 2: Multiresolution / wavelet representations

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| DWT | Dyadic | Depends | Yes | Multi-scale, sparse, singularity detection |
| CWT | Log (constant-Q) | Yes (complex) | Yes | Scale-invariant structure |
| Wavelet packets | Adaptive | Depends | Yes | Data-driven adaptive tiling |
| Stationary WT | Dyadic (redundant) | Depends | Yes | Shift-invariance |
| Dual-tree CWT | Dyadic (complex) | Yes | Yes | Directional selectivity, shift-invariance |

**Key insight**: Dyadic/log tiling makes multi-scale structure visible — structure that exists at specific scales (edges in images, syllables in speech, events in sensors). This is the structural prior that aligns with hierarchical semantic structure.

### Category 3: Auditory / perceptual representations

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| Mel spectrogram | Log-spaced | No | No | Perceptual frequency resolution |
| Gammatone | ERB-spaced | Yes (complex) | No | Cochlear model, biological plausibility |
| CQT | Log (constant-Q) | Yes | Yes | Musical pitch structure |
| Cochleagram | ERB-spaced | Yes | No | Most biologically accurate auditory model |
| MFCC | Log → decorrelated | No | No | Compact, decorrelated spectral envelope |

**Key insight**: Perceptual representations match biological processing but typically discard phase. The CQT is unique in preserving phase while matching perceptual scale — it may be the optimal audio representation for semantic structure.

### Category 4: Non-linear and higher-order representations

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| Wigner-Ville | Continuous | Implicit | No | Maximum concentration, cross-terms |
| Choi-Williams | Continuous | Implicit | No | Reduced cross-terms |
| Bispectrum | Bifrequency | Yes | No | Quadratic phase coupling |
| Polyspectra | N-dimensional | Yes | No | Higher-order nonlinearities |
| EMD | Adaptive | Yes | Yes | Data-driven, non-stationary |
| Hilbert-Huang | Adaptive | Yes | Partial | Time-varying frequency, non-stationary |

**Key insight**: Higher-order spectra (bispectrum, polyspectra) capture phase coupling — the nonlinear relationships between frequency components. This is directly relevant to semantic structure: phase coupling is the mechanism by which different frequency bands coordinate to encode meaning.

### Category 5: Learned / neural frequency representations

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| Fourier Neural Operator | Uniform | Yes | No | Global correlations, PDE operators |
| Spectral CNN | Uniform | Depends | No | Efficient global convolutions |
| SincNet | Learned | Depends | No | Data-adaptive frequency analysis |
| Complex-valued NN | Varies | Yes | No | Phase-aware processing |
| Spectral autoencoder | Varies | Depends | Partial | Learned spectral features |

**Key insight**: Learned representations can adapt their structural prior to the data, but risk overfitting. Complex-valued neural networks are unique in preserving phase as a first-class citizen.

### Category 6: Graph spectral representations

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| Graph Fourier | Eigenvalues | Yes | Yes | Graph structure, smoothness |
| Graph wavelets | Multi-scale | Depends | Yes | Multi-scale graph structure |
| Spectral GCN | Polynomial | Typically no | No | Localized graph operations |
| Graph spectral pooling | Hierarchical | N/A | No | Hierarchical graph structure |

**Key insight**: Graph spectral representations capture relational structure — the structure of connections between entities. This is the natural representation for text (dependency graphs), social networks, and knowledge graphs. Graph wavelets extend this to multi-scale analysis.

### Category 7: Physics-based frequency representations

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| Normal modes | Discrete (system) | Yes | Yes | Resonant structure of physical systems |
| Eigenfunction expansions | Eigenvalue spectrum | Yes | Yes | PDE solution structure |
| Spherical harmonics | Angular frequency | Yes | Yes | Rotational structure on sphere |
| Lorentzian | Continuous | Yes | Partial | Resonance, damping, linewidth |

**Key insight**: Physics-based representations capture the natural resonant structure of systems. Normal modes are the frequencies at which a system "wants" to oscillate — this is a form of inherent structure that exists independent of any observer.

### Category 8: Information-theoretic frequency models

| Representation | Tiling | Phase? | Invertible? | Structural prior |
|---|---|---|---|---|
| Spectral entropy | Depends | No | No | Information content per frequency |
| Spectral mutual information | Bifrequency | Yes | No | Dependencies between frequencies |
| Spectral information bottleneck | Depends | Depends | No | Relevant frequency information |

**Key insight**: Information-theoretic models measure the statistical dependencies between frequency components. Spectral mutual information captures cross-frequency coupling — the same mechanism that CBMPC measures via phase locking.

---

## The key finding: structure is inherent in the representation

**Every frequency representation imposes a structural prior.** The choice of representation determines what semantic structure is visible:

| Structure type | Best representation | Why |
|---|---|---|
| Stationary spectral content | STFT, Gabor | Uniform tiling reveals stationary components |
| Multi-scale / hierarchical | Wavelets (DWT, CWT) | Dyadic tiling reveals structure at multiple scales |
| Perceptual / biological | Mel, Gammatone, CQT | Log-spaced tiling matches biological processing |
| Phase coupling / nonlinear | Bispectrum, polyspectra | Higher-order spectra reveal phase relationships |
| Relational / graph | Graph Fourier, graph wavelets | Eigenvalue structure reveals graph topology |
| Resonant / physical | Normal modes, eigenfunctions | Natural frequencies reveal system structure |
| Information-theoretic | Spectral MI, spectral entropy | Statistical dependencies reveal coupling |

**No single representation is optimal for all structure types.** The optimal representation depends on what kind of semantic structure is present in the data.

---

## The synthesis: hybrid phase-coherent multi-scale representations

The survey concludes that **phase-coherent multi-scale representations** come closest to revealing semantic structure across modalities. These representations:

1. **Preserve phase information** (critical for temporal coherence and causal structure)
2. **Use multi-scale / adaptive tiling** (matches hierarchical semantic structure)
3. **Are invertible** (preserves all information for reconstruction)
4. **Include uncertainty quantification** (handles noise and missing data)

**Bifrost's `SpectralTensor`** already combines these elements:
- Complex spectrum (amplitude + phase)
- Scale metadata (multi-resolution)
- Uncertainty quantification
- Invertible via iFFT

**CBMPC** extends this by computing modulation spectra and measuring cross-band phase coherence — capturing the phase coupling that is the mechanism of semantic structure.

---

## Implications for the Bifrost project

### 1. The representation is the inductive bias

The frequency representation chosen by Bifrost is not a neutral preprocessing step — it is the primary inductive bias that determines what semantic structure the system can discover. This aligns with the finding from the AGI literature that **representation matters more than algorithm** (arXiv:2402.06590, 2024).

### 2. Spectral bias is fundamental

Neural networks have an intrinsic spectral bias toward low-frequency functions (arXiv:1806.08734, 2018). This bias is key to generalization. Bifrost's frequency-domain approach can leverage this bias directly — by operating in the frequency domain, the system can control which frequencies it learns, enabling systematic generalization.

### 3. Phase coupling is the mechanism of semantic structure

The bispectrum and polyspectra capture phase coupling — nonlinear relationships between frequency components. CBMPC captures cross-band phase locking. The biological literature shows that phase synchronization is the mechanism for cross-modal binding and semantic processing. **Phase coupling is the common thread.**

### 4. Multi-scale tiling matches hierarchical semantics

Wavelet representations with dyadic/log tiling make multi-scale structure visible. This matches the hierarchical nature of semantic structure: phonemes → syllables → words → phrases → sentences in speech; edges → textures → objects → scenes in images; samples → events → patterns → trends in sensors.

### 5. The optimal representation is modality-dependent but the principle is universal

While the optimal frequency representation varies by modality (CQT for music, gammatone for speech, wavelets for sensors, graph spectral for text), the underlying principle is universal: **phase-coherent multi-scale representations reveal semantic structure.**

---

## Recommendations for Bifrost

1. **Add Constant-Q Transform (CQT) as an alternative canonicalizer** for audio and music. CQT preserves phase while matching perceptual scale — it may outperform STFT for music and tonal audio.

2. **Implement wavelet packet decomposition** as an alternative to the current learnable wavelet bank. Wavelet packets provide adaptive time-frequency tiling that can match signal characteristics.

3. **Add bispectrum / bicoherence features** to capture quadratic phase coupling. This is a direct measure of nonlinear frequency relationships that may carry semantic structure not captured by CBMPC.

4. **Explore graph spectral methods** for text and relational data. Graph wavelets extend the MSC framework to non-Euclidean data.

5. **Investigate spectral information bottleneck** for task-relevant frequency selection. This could enable the system to learn which frequencies are semantically relevant for each task.

6. **Maintain phase as a first-class citizen** in all representations. The survey confirms that phase preservation is essential for semantic structure.

7. **Implement spectral mutual information** as a complementary coherence measure to CBMPC's phase locking value. Spectral MI captures statistical dependencies that PLV may miss.

---

## Theoretical grounding

| Reference | Contribution |
|---|---|
| Gabor (1946) | Time-frequency uncertainty principle |
| Mallat (1989) | Multiresolution signal decomposition |
| Daubechies (1992) | Wavelet theory |
| Kovesi (1999) | Phase congruency as image feature |
| Grinsted et al. (2004) | Wavelet coherence for cross-scale phase |
| Hammond et al. (2011) | Graph wavelets via spectral graph theory |
| Li et al. (2020) | Fourier Neural Operator |
| Rahaman et al. (2018) | Spectral bias of neural networks |
| Oppenheim & Lim (1981) | Phase carries structural information |
| Huang et al. (1998) | Empirical Mode Decomposition |
| Patterson et al. (1988) | Gammatone filterbank |
| Brown (1992) | Constant-Q transform |
| Coifman & Wickerhauser (1992) | Wavelet packet best basis |
| Kingsbury (1999) | Dual-tree complex wavelet transform |
