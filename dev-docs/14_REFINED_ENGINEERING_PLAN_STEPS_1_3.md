# 14 — Refined Engineering Plan: Steps 1-3

**Goal**: Prove the first three claims of the Structured Resonance Thesis with minimal engineering, using existing Bifrost infrastructure.

**Thesis**: Intelligence is structured resonance, and phase-coherent multi-scale representations capture semantic structure across modalities.

---

## Step 1: Prove phase coherence captures semantic structure (C1)

### Hypothesis

Phase-coherent features outperform phase-invariant features on semantic classification tasks. If phase is destroyed, semantic structure is lost.

### What exists

- `CBMPCExtractor` (`src/bifrost/cbmpc.py`) — computes cross-band modulation phase locking value for audio. Validated on SpeechCommands (+13.65 pp, p = 0.0033).
- `PhaseCongruencyExtractor` (`src/bifrost/msc_image.py`) — computes phase congruency across log-Gabor scales for images. Implemented but not validated.
- `SpectralCanonicalizer` (`src/bifrost/canonicalizer/canonicalizer.py`) — extracts amplitude and phase from FFT.

### What needs to be built

#### 1.1 Phase ablation harness

**File**: `src/bifrost/validation/phase_ablation.py` (new)

```python
class PhaseAblationHarness:
    """Systematically destroys phase information to measure its contribution
    to semantic task performance."""

    def phase_randomize(self, spectral: SpectralTensor) -> SpectralTensor:
        """Shuffle phase across the time axis. Preserves amplitude distribution
        but destroys temporal phase relationships."""
        # phase_shuffled = phase[:, torch.randperm(phase.shape[1]), :]
        # Keep amplitude, replace phase with shuffled version

    def phase_zero(self, spectral: SpectralTensor) -> SpectralTensor:
        """Set all phase to zero. Equivalent to magnitude-only spectrogram.
        This is what most audio ML systems actually use."""

    def phase_noise(self, spectral: SpectralTensor, sigma: float) -> SpectralTensor:
        """Add Gaussian noise to phase: phi' = phi + N(0, sigma).
        Degrades phase gradually to measure phase precision sensitivity."""

    def phase_quantize(self, spectral: SpectralTensor, n_levels: int) -> SpectralTensor:
        """Quantize phase to n_levels. Measures how much phase precision
        is needed to preserve semantic structure."""

    def cross_band_phase_scramble(self, spectral: SpectralTensor) -> SpectralTensor:
        """Scramble phase relationships between frequency bands while
        preserving within-band phase. Directly tests CBMPC's hypothesis:
        if cross-band phase relationships carry semantic structure,
        scrambling them should destroy it."""
```

**Design principle**: Each ablation targets a specific aspect of phase. `phase_zero` tests whether phase matters at all. `phase_randomize` tests whether temporal phase structure matters. `cross_band_phase_scramble` directly tests the CBMPC hypothesis — that cross-band phase relationships carry semantic structure.

#### 1.2 Phase coherence metrics module

**File**: `src/bifrost/validation/phase_metrics.py` (new)

```python
class PhaseCoherenceMetrics:
    """Phase coherence metrics computable at any pipeline layer."""

    def phase_locking_value(self, phases_a: Tensor, phases_b: Tensor) -> float:
        """PLV = |mean(exp(i * (phase_a - phase_b)))|.
        Range: [0, 1]. 1 = perfect phase locking, 0 = no locking."""

    def phase_entropy(self, phases: Tensor) -> float:
        """Shannon entropy of phase distribution.
        Low entropy = concentrated phases = high coherence.
        High entropy = dispersed phases = low coherence."""

    def phase_congruency(self, multi_scale_phases: List[Tensor],
                         amplitudes: List[Tensor]) -> Tensor:
        """PC(x) = |sum_s A_s cos(phi_s - phi_bar)| / sum_s A_s
        Kovesi (1999) phase congruency across scales.
        High PC = phase aligned across scales = semantic feature present."""

    def cross_frequency_coupling(self, phases: Tensor,
                                  low_freq_idx: List[int],
                                  high_freq_idx: List[int]) -> float:
        """Phase-amplitude coupling between low and high frequency bands.
        Measures whether high-frequency phase is locked to low-frequency phase.
        This is the theta-gamma coupling mechanism from neuroscience."""

    def phase_stability(self, phases_over_time: Tensor) -> float:
        """Temporal stability of phase: 1 - var(phase) / (2*pi).
        Stable phase = consistent semantic structure over time."""
```

**Design principle**: These metrics are not learned — they are pure signal processing. They can be computed at any layer of the pipeline to measure how much phase coherence exists before, during, and after processing.

#### 1.3 Phase-aware contrastive loss

**File**: `src/bifrost/training.py` (extend existing)

```python
class PhaseCoherenceContrastiveLoss(nn.Module):
    """Contrastive loss where similarity is phase coherence, not dot product.

    Hypothesis: samples with similar semantics have phase-coherent
    representations. Pulling same-class samples together in phase coherence
    space should produce more semantically meaningful embeddings than
    pulling them together in dot-product space.
    """

    def forward(self, anchor: SpectralTensor,
                positive: SpectralTensor,
                negative: SpectralTensor) -> Tensor:
        sim_pos = self.plv(anchor.phase, positive.phase)
        sim_neg = self.plv(anchor.phase, negative.phase)
        return -torch.log(sim_pos / (sim_pos + sim_neg + 1e-8))
```

### Experiments

#### Experiment 1A: Phase ablation on audio (SpeechCommands)

**Dataset**: SpeechCommands (10 classes, 200 samples/class, 5-fold CV)
**Feature extractor**: CBMPCExtractor
**Conditions**:
1. CBMPC with full phase (baseline — already validated)
2. CBMPC with phase_zero (magnitude-only modulation spectrum)
3. CBMPC with phase_randomize (temporal phase shuffled)
4. CBMPC with cross_band_phase_scramble (cross-band relationships destroyed)
5. STFT amplitude-only baseline (no phase at any stage)

**Metrics**: Accuracy, PLV, phase entropy, paired t-test (Bonferroni-corrected)

**Success criterion**: Condition 1 > Conditions 2-4 with p < 0.05. Condition 1 > Condition 5 by at least 10 pp (already shown: +13.65 pp).

**What this proves**: If cross_band_phase_scramble destroys performance but phase_randomize does not, it confirms that *cross-band* phase relationships (not just temporal phase) carry semantic structure.

#### Experiment 1B: Phase ablation on images (CIFAR-10)

**Dataset**: CIFAR-10 (10 classes, 500 samples/class, 5-fold CV)
**Feature extractor**: PhaseCongruencyExtractor
**Conditions**:
1. Phase congruency with full phase (baseline)
2. Phase congruency with phase_zero (amplitude-only)
3. Phase congruency with phase_randomize (spatial phase shuffled)
4. Phase congruency with cross_scale_phase_scramble (cross-scale relationships destroyed)
5. Raw pixel baseline (no spectral processing)
6. HOG baseline (gradient-based features)

**Metrics**: Accuracy, phase congruency map quality, paired t-test

**Success criterion**: Condition 1 > Conditions 2-4 with p < 0.05. Condition 1 > Condition 5 by at least 5 pp.

**What this proves**: If the same pattern holds on images as on audio — phase matters, and cross-scale phase relationships matter specifically — then C1 is supported across modalities.

#### Experiment 1C: Phase coherence as a predictor

**Method**: For each sample in SpeechCommands and CIFAR-10, compute phase coherence metrics (PLV, phase entropy, phase congruency). Correlate these metrics with classification confidence.

**Success criterion**: Phase coherence metrics correlate with classification confidence (r > 0.3, p < 0.05).

**What this proves**: Phase coherence is not just useful for classification — it *predicts* classification confidence. This supports the stronger claim that phase coherence IS semantic structure, not just a useful feature.

### Deliverables

| Deliverable | File | Status |
|---|---|---|
| Phase ablation harness | `src/bifrost/validation/phase_ablation.py` | To build |
| Phase coherence metrics | `src/bifrost/validation/phase_metrics.py` | To build |
| Phase-aware contrastive loss | `src/bifrost/training.py` (extend) | To build |
| Experiment 1A script | `research_dir/experiment_phase_ablation_audio.py` | To write |
| Experiment 1B script | `research_dir/experiment_phase_ablation_image.py` | To write |
| Experiment 1C script | `research_dir/experiment_phase_coherence_predictor.py` | To write |

---

## Step 2: Prove multi-scale coherence is necessary (C2)

### Hypothesis

Coherence *across* scales captures more semantic structure than coherence *within* any single scale. The multi-scale structure is not just "more features" — it is the relationships between scales that carry semantic structure.

### What exists

- `LearnableWaveletBank` (`src/bifrost/decomposer/decomposer.py`) — processes n_scales in parallel with different dilations. But scales are processed independently — no cross-scale coherence is computed.
- `PhaseCongruencyExtractor` (`src/bifrost/msc_image.py`) — computes phase congruency across log-Gabor scales. This IS cross-scale phase coherence for images, but it is not generalized to a framework concept.

### What needs to be built

#### 2.1 Cross-scale phase coherence module

**File**: `src/bifrost/cross_scale_coherence.py` (new)

```python
class CrossScaleCoherence(nn.Module):
    """Computes phase coherence between different analysis scales.

    This is the generalization of CBMPC from cross-band (across frequency
    bands at one scale) to cross-scale (across analysis scales).

    For audio: measures whether modulation phase at scale s1 (e.g., 2Hz)
    is coherent with modulation phase at scale s2 (e.g., 4Hz).
    For images: measures whether phase congruency at fine scale is
    coherent with phase congruency at coarse scale.
    """

    def __init__(self, n_scales: int = 6):
        self.n_scales = n_scales
        # Scale pairs: (s1, s2) where s2/s1 = 2^k (dyadic)
        self.scale_pairs = [(i, j) for i in range(n_scales)
                           for j in range(i+1, n_scales)]

    def forward(self, multi_scale_phases: List[Tensor],
                multi_scale_amplitudes: List[Tensor]) -> Tensor:
        """
        Args:
            multi_scale_phases: [phase_s1, phase_s2, ..., phase_sn]
            multi_scale_amplitudes: [amp_s1, amp_s2, ..., amp_sn]

        Returns:
            cross_scale_coherence: Tensor of shape [n_pairs, ...]
            containing PLV between each scale pair
        """
        coherence_features = []
        for (i, j) in self.scale_pairs:
            # Weight by amplitude — high-amplitude phases matter more
            weight = multi_scale_amplitudes[i] * multi_scale_amplitudes[j]
            plv = self._weighted_plv(
                multi_scale_phases[i],
                multi_scale_phases[j],
                weight
            )
            coherence_features.append(plv)

        # Also compute scale ratio coherence:
        # If s2 = 2*s1, phase at s2 should be 2x phase at s1
        # (harmonic relationship). Measure deviation from this.
        for (i, j) in self.scale_pairs:
            ratio = 2 ** (j - i)  # dyadic ratio
            expected_phase = multi_scale_phases[i] * ratio
            phase_deviation = torch.angle(
                torch.exp(1j * (multi_scale_phases[j] - expected_phase))
            )
            coherence_features.append(phase_deviation.abs().mean())

        return torch.stack(coherence_features, dim=-1)

    def _weighted_plv(self, phase_a, phase_b, weight):
        diff = phase_a - phase_b
        return (weight * torch.exp(1j * diff)).sum() / (weight.sum() + 1e-8)
```

**Design principle**: Cross-scale coherence has two components:
1. **PLV between scales** — are phases at different scales locked?
2. **Harmonic deviation** — if scale s2 = 2 * scale s1, does phase at s2 follow the harmonic relationship? This is the wavelet analog of harmonic binding in audio.

#### 2.2 Scale ablation framework

**File**: `src/bifrost/validation/scale_ablation.py` (new)

```python
class ScaleAblationHarness:
    """Tests whether cross-scale coherence matters beyond single-scale."""

    def single_scale(self, multi_scale_phases, scale_idx: int):
        """Use only one scale. Discard all others."""

    def scale_subset(self, multi_scale_phases, k: int):
        """Use k randomly chosen scales. Tests whether more scales = better."""

    def scale_shuffle(self, multi_scale_phases):
        """Shuffle scale assignments. Same scales, wrong scale labels.
        Tests whether the scale ordering matters."""

    def cross_scale_destroy(self, multi_scale_phases):
        """Keep all scales but break cross-scale phase relationships.
        Add independent random phase offset to each scale.
        Tests whether cross-scale relationships specifically matter,
        not just having multiple scales."""
```

**Design principle**: `cross_scale_destroy` is the critical ablation. If performance drops when cross-scale relationships are broken but individual scales are preserved, it proves that the *relationships between scales* carry semantic structure — not just the features at each scale.

### Experiments

#### Experiment 2A: Cross-scale coherence on audio

**Dataset**: SpeechCommands (10 classes, 200 samples/class, 5-fold CV)
**Pipeline**: CBMPC → CrossScaleCoherence → classifier

**Conditions**:
1. CBMPC + cross-scale coherence (full)
2. CBMPC + single-scale (best single scale)
3. CBMPC + multi-scale (all scales, no cross-scale coherence)
4. CBMPC + cross-scale-destroyed (scales preserved, relationships broken)
5. CBMPC only (baseline, no cross-scale)

**Metrics**: Accuracy, cross-scale PLV, paired t-test

**Success criterion**: Condition 1 > Conditions 2-4 with p < 0.05. Specifically:
- Condition 1 > Condition 2: multi-scale > single-scale
- Condition 1 > Condition 3: cross-scale coherence > multi-scale without coherence
- Condition 1 > Condition 4: cross-scale relationships matter, not just having scales

**What this proves**: If Condition 1 > Condition 4, it proves that cross-scale *relationships* carry semantic structure. If Condition 1 > Condition 3, it proves that computing coherence across scales adds value beyond just having multi-scale features.

#### Experiment 2B: Cross-scale coherence on images

**Dataset**: CIFAR-10 (10 classes, 500 samples/class, 5-fold CV)
**Pipeline**: PhaseCongruencyExtractor → CrossScaleCoherence → classifier

**Conditions**: Same as 2A, adapted for image phase congruency.

**Success criterion**: Same pattern as 2A.

**What this proves**: Cross-scale coherence matters across modalities, not just audio.

#### Experiment 2C: Scale hierarchy analysis

**Method**: For each sample, measure which scale pairs have the highest cross-scale PLV. Test whether semantically similar samples have similar cross-scale coherence profiles.

**Metric**: Correlation between cross-scale coherence profile similarity and semantic similarity (same class vs different class).

**Success criterion**: Same-class samples have more similar cross-scale coherence profiles than different-class samples (p < 0.05).

### Deliverables

| Deliverable | File | Status |
|---|---|---|
| Cross-scale coherence module | `src/bifrost/cross_scale_coherence.py` | To build |
| Scale ablation harness | `src/bifrost/validation/scale_ablation.py` | To build |
| Experiment 2A script | `research_dir/experiment_cross_scale_audio.py` | To write |
| Experiment 2B script | `research_dir/experiment_cross_scale_image.py` | To write |
| Experiment 2C script | `research_dir/experiment_scale_hierarchy.py` | To write |

---

## Step 3: Prove cross-modal generalization (C3)

### Hypothesis

The same principle — phase-coherent multi-scale structure — captures semantic structure across audio, image, and sensor modalities. A unified coherence metric works across all three.

### What exists

- `CBMPCExtractor` (audio) — validated
- `PhaseCongruencyExtractor` (image) — implemented, not validated
- `MultiModalSpectralPipeline` — routes by modality but does not unify coherence

### What needs to be built

#### 3.1 Wavelet coherence extractor (sensor modality)

**File**: `src/bifrost/msc_sensor.py` (new)

```python
class WaveletCoherenceExtractor(nn.Module):
    """Sensor MSC: Cross-channel wavelet coherence at multiple time scales.

    Implements the Grinsted, Moore & Jevrejeva (2004) wavelet coherence
    method for measuring cross-channel phase relationships in sensor data.

    For each pair of sensor channels (c_i, c_j):
    1. Compute CWT of each channel
    2. Compute cross-wavelet transform: W_ij = W_i * conj(W_j)
    3. Compute wavelet coherence: R^2(a,t) = |S(s^-1 W_ij)|^2 / (S(s^-1|W_i|^2) * S(s^-1|W_j|^2))
    4. Extract phase angle: arctan(Im(S(s^-1 W_ij)) / Re(S(s^-1 W_ij)))
    5. Aggregate coherence across channel pairs and scales
    """

    def __init__(self, n_scales: int = 12, n_channels: int = 6):
        self.n_scales = n_scales
        self.n_channels = n_channels
        # Morlet wavelet parameters
        self.scales = torch.tensor([2 ** (i/2) for i in range(n_scales)])

    def cwt(self, signal: Tensor) -> Tensor:
        """Continuous wavelet transform using Morlet wavelet.
        Returns [n_scales, signal_length] complex tensor."""

    def wavelet_coherence(self, cwt_a: Tensor, cwt_b: Tensor) -> Tensor:
        """Compute R^2(a,t) wavelet coherence between two CWTs.
        Smooths the cross-wavelet spectrum before normalizing."""

    def forward(self, multi_channel_signal: Tensor) -> Tensor:
        """Input: [n_channels, signal_length]
        Output: coherence feature vector

        For each channel pair (i,j):
            Compute wavelet coherence R^2(a,t)
            Extract mean coherence per scale
            Extract phase angle distribution per scale

        Aggregate into unified coherence vector.
        """
```

**Reference**: Grinsted, Moore & Jevrejeva (2004). "Application of the cross wavelet transform and wavelet coherence to geophysical time series." Nonlinear Processes in Geophysics, 11, 561-566.

**Dataset**: UCI Human Activity Recognition (6 activities: walking, walking upstairs, walking downstairs, sitting, standing, laying). 6 sensor channels (accelerometer x/y/z, gyroscope x/y/z) at 50 Hz.

#### 3.2 Unified coherence metric

**File**: `src/bifrost/unified_coherence.py` (new)

```python
class UnifiedCoherenceMetric(nn.Module):
    """Maps modality-specific coherence features to a shared coherence space.

    All MSC instances (CBMPC, PhaseCongruency, WaveletCoherence) produce
    coherence features in different dimensions. This module projects them
    to a shared space where cross-modal comparison is possible.

    The projection is learned via contrastive learning:
    - Same-semantic samples from different modalities should map nearby
    - Different-semantic samples should map far apart
    """

    def __init__(self, audio_dim: int, image_dim: int, sensor_dim: int,
                 target_dim: int = 256):
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, target_dim * 2),
            nn.ReLU(),
            nn.Linear(target_dim * 2, target_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, target_dim * 2),
            nn.ReLU(),
            nn.Linear(target_dim * 2, target_dim)
        )
        self.sensor_proj = nn.Sequential(
            nn.Linear(sensor_dim, target_dim * 2),
            nn.ReLU(),
            nn.Linear(target_dim * 2, target_dim)
        )

    def forward(self, coherence_features: Tensor,
                modality: str) -> Tensor:
        """Project to shared coherence space and normalize."""
        if modality == "audio":
            projected = self.audio_proj(coherence_features)
        elif modality == "image":
            projected = self.image_proj(coherence_features)
        elif modality == "sensor":
            projected = self.sensor_proj(coherence_features)
        return F.normalize(projected, dim=-1)

    def coherence_similarity(self, feat_a: Tensor, feat_b: Tensor) -> float:
        """Cosine similarity in unified coherence space."""
        return F.cosine_similarity(feat_a, feat_b, dim=-1).mean()
```

**Design principle**: The unified coherence space is where the thesis is tested. If coherence features from different modalities cluster by semantic category (not by modality), the thesis is supported.

#### 3.3 Cross-modal coherence dataset

**Problem**: Existing datasets are single-modal. To test cross-modal generalization, we need paired data where the same semantic concept appears in multiple modalities.

**Approach**: Use existing paired datasets:

| Dataset | Modalities | Semantic classes | Size |
|---|---|---|---|
| AudioSet-Strong | Audio + video thumbnails | 527 event classes | ~2M clips |
| VGGSound | Audio + video frames | 309 sound classes | 200K videos |
| ESC-50 + ImageNet subset | Audio (environmental) + Image (matching categories) | 10 matched classes | 2K audio + 13K images |
| UCI HAR | Sensor (IMU) | 6 activities | 10K samples |

**For the minimal proof**: Use ESC-10 (environmental sounds) paired with ImageNet images of the same categories (dog, cat, rooster, rain, sea waves, crackling fire, crickets, chirping birds, water droplets, wind). This gives us audio-image pairs with shared semantic categories.

For sensor: use UCI HAR independently — it's a within-modal validation (does wavelet coherence work for sensor data?).

### Experiments

#### Experiment 3A: Validate image MSC (CIFAR-10)

**Dataset**: CIFAR-10 (10 classes, 500 samples/class, 5-fold CV)
**Feature extractor**: PhaseCongruencyExtractor
**Baselines**: Raw pixels, HOG, SIFT, ResNet-18 (frozen features)
**Metrics**: Accuracy, F1, confusion matrix

**Success criterion**: PhaseCongruency > Raw pixels by at least 10 pp. PhaseCongruency competitive with HOG.

**What this proves**: The image MSC instance works — phase congruency captures semantic structure in images.

#### Experiment 3B: Validate sensor MSC (UCI HAR)

**Dataset**: UCI HAR (6 classes, 500 samples/class, 5-fold CV)
**Feature extractor**: WaveletCoherenceExtractor
**Baselines**: Raw time series, FFT magnitude, statistical features (mean, std, skew, kurtosis per channel)
**Metrics**: Accuracy, F1

**Success criterion**: WaveletCoherence > FFT magnitude by at least 5 pp. WaveletCoherence > statistical features by at least 10 pp.

**What this proves**: The sensor MSC instance works — wavelet coherence captures semantic structure in sensor data.

#### Experiment 3C: Cross-modal coherence alignment

**Dataset**: ESC-10 + ImageNet matched subset (10 shared categories)
**Method**:
1. Extract audio coherence features (CBMPC) from ESC-10
2. Extract image coherence features (PhaseCongruency) from ImageNet
3. Train UnifiedCoherenceMetric with contrastive loss:
   - Same-category audio-image pairs → pull together
   - Different-category pairs → push apart
4. Evaluate: can coherence features classify samples across modalities?

**Conditions**:
1. Train classifier on audio coherence, test on image coherence (cross-modal transfer)
2. Train classifier on image coherence, test on audio coherence (reverse transfer)
3. Train on both, test on held-out pairs (joint classification)
4. Baseline: same with amplitude-only features (no phase)

**Metrics**: Cross-modal transfer accuracy, joint classification accuracy, coherence space silhouette score

**Success criterion**: Cross-modal transfer accuracy > chance (10%) by at least 5 pp. Phase coherence transfer > amplitude-only transfer.

**What this proves**: The same coherence principle captures semantic structure across audio and image. Coherence features from one modality can classify samples in another — meaning the semantic structure is in the coherence pattern, not in the modality-specific features.

#### Experiment 3D: Coherence space visualization

**Method**: Project all coherence features (audio, image, sensor) to 2D via t-SNE. Color by semantic category. Check whether samples cluster by category (not by modality).

**Success criterion**: t-SNE visualization shows clustering by semantic category, not by modality. Quantify with silhouette score (category clustering > modality clustering).

### Deliverables

| Deliverable | File | Status |
|---|---|---|
| Wavelet coherence extractor | `src/bifrost/msc_sensor.py` | To build |
| Unified coherence metric | `src/bifrost/unified_coherence.py` | To build |
| Cross-modal dataset loader | `src/bifrost/validation/cross_modal_dataset.py` | To build |
| Experiment 3A script | `research_dir/experiment_msc_image_cifar10.py` | Exists (fix deps) |
| Experiment 3B script | `research_dir/experiment_msc_sensor_har.py` | To write |
| Experiment 3C script | `research_dir/experiment_cross_modal_coherence.py` | To write |
| Experiment 3D script | `research_dir/experiment_coherence_visualization.py` | To write |

---

## Summary: The minimal viable proof

| Step | Claim | Experiment | Dataset | Success criterion |
|---|---|---|---|---|
| 1A | Phase matters (audio) | Phase ablation on CBMPC | SpeechCommands | Full phase > ablated (p < 0.05) |
| 1B | Phase matters (image) | Phase ablation on PhaseCongruency | CIFAR-10 | Full phase > ablated (p < 0.05) |
| 1C | Phase predicts semantics | Coherence-confidence correlation | Both | r > 0.3, p < 0.05 |
| 2A | Cross-scale matters (audio) | Scale ablation | SpeechCommands | Cross-scale > single/destroyed (p < 0.05) |
| 2B | Cross-scale matters (image) | Scale ablation | CIFAR-10 | Same pattern |
| 2C | Scale hierarchy is semantic | Coherence profile similarity | Both | Same-class > different-class (p < 0.05) |
| 3A | Image MSC works | PhaseCongruency classification | CIFAR-10 | > raw pixels by 10 pp |
| 3B | Sensor MSC works | WaveletCoherence classification | UCI HAR | > FFT by 5 pp |
| 3C | Cross-modal alignment | Cross-modal transfer | ESC-10 + ImageNet | Transfer > chance by 5 pp |
| 3D | Coherence space is semantic | t-SNE visualization | All three | Category clustering > modality clustering |

**Total experiments**: 10
**Total new modules**: 6 (phase ablation, phase metrics, cross-scale coherence, scale ablation, wavelet coherence, unified coherence)
**Total new experiment scripts**: 8 (2 existing scripts can be reused)

Each experiment is independently publishable. Together, they prove the first three claims of the Structured Resonance Thesis: phase coherence captures semantic structure (C1), multi-scale coherence is necessary (C2), and the principle generalizes across modalities (C3).
