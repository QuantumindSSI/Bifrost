# 08 — Cross-Modal Validation Protocol

**Status**: Pre-registration  
**Goal**: Validate that the MSC framework establishes structural intelligence across multiple data modalities.

---

## Pre-registered hypotheses

### H1: Each MSC instance beats its modality-specific baseline
For each modality, the MSC feature extractor will achieve significantly higher classification accuracy than a standard baseline on a modality-specific benchmark.

- **Audio**: CBMPC > STFT magnitude baseline on SpeechCommands. **Already validated** (0.41 vs 0.27, p = 0.0033).
- **Image**: Phase congruency > raw pixel / FFT magnitude baseline on CIFAR-10.
- **Sensor**: Wavelet coherence > statistical features baseline on UCI HAR.
- **Text**: Graph spectral coherence > mean word embedding baseline on SST-2.

**Success criterion**: Each MSC instance must exceed its baseline by ≥ 5 absolute percentage points, p < 0.05 (paired t-test across CV folds).

### H2: MSC embeddings enable above-chance cross-modal retrieval
MSC embeddings from different modalities will produce above-chance cross-modal retrieval on datasets with known cross-modal correspondence.

**Success criterion**: Retrieval accuracy (Recall@1) must exceed chance by a statistically significant margin (p < 0.05, binomial test).

### H3: MSC coherence profiles correlate across modalities for structurally similar content
Inputs from different modalities that share structural properties (e.g., rhythmic audio + rhythmic video) will have more similar MSC coherence profiles than inputs without shared structure.

**Success criterion**: Correlation between MSC coherence profiles of structurally matched pairs must be significantly higher than random pairs (p < 0.05, permutation test).

---

## Modality-specific protocols

### Image: Phase Congruency on CIFAR-10

**Dataset**: CIFAR-10, 10 classes, 6000 images/class, 5-fold CV.
**Models**:
1. Phase congruency features → linear classifier.
2. Raw pixel mean-pool → linear classifier (baseline).
3. Raw FFT magnitude mean-pool → linear classifier (baseline).
4. HOG features → linear classifier (strong baseline).

**Pre-registration**: Phase congruency must beat both raw pixel and raw FFT baselines by ≥ 5 pp.

### Sensor: Wavelet Coherence on UCI HAR

**Dataset**: UCI HAR, 6 activities, 30 subjects, pre-defined train/test split.
**Models**:
1. Wavelet coherence features → linear classifier.
2. Statistical features (mean, std, mad, max, min, etc.) → linear classifier (baseline).
3. Raw signal → MLP (strong baseline).

**Pre-registration**: Wavelet coherence must beat statistical features baseline by ≥ 5 pp.

### Text: Graph Spectral Coherence on SST-2

**Dataset**: SST-2, binary sentiment, 67k train / 872 dev, standard split.
**Models**:
1. Graph spectral coherence features → linear classifier.
2. Mean word embedding (GloVe) → linear classifier (baseline).
3. Bag-of-words → linear classifier (baseline).

**Pre-registration**: Graph spectral coherence must beat mean word embedding baseline by ≥ 3 pp (lower threshold because text classification is harder to improve with structural features).

---

## Cross-modal protocol

### Cross-modal retrieval: Audio-Visual

**Dataset**: We will construct audio-visual pairs from a dataset with known correspondence. Options:
1. AudioSet-ImageNet pairs (if available).
2. Flickr8k (audio captions + images).
3. VGGSound (video frames + audio).

**Protocol**:
1. Extract MSC embeddings for all audio clips and all images.
2. For each audio clip, retrieve the top-k most similar images by cosine similarity of MSC embeddings.
3. Measure Recall@1, Recall@5, Recall@10.
4. Compare to chance (1/N where N is the number of images).

**Pre-registration**: Recall@1 must exceed chance by a statistically significant margin.

### Cross-modal structural similarity: Rhythmic patterns

**Dataset**: We will construct a controlled dataset of:
- Audio clips with rhythmic structure (music with clear beats at 1, 2, 4 Hz).
- Video clips with visual rhythms at matching frequencies (flashing lights, moving patterns).
- Non-rhythmic audio and video as controls.

**Protocol**:
1. Extract MSC embeddings for all clips.
2. Compute the coherence profile (coherence values at each scale) for each clip.
3. For each rhythmic audio clip, compute the correlation of its coherence profile with each video clip's coherence profile.
4. Test whether matched-frequency pairs (audio rhythm at 2 Hz + video rhythm at 2 Hz) have higher coherence profile correlation than mismatched pairs.

**Pre-registration**: Matched pairs must have significantly higher coherence profile correlation than mismatched pairs (p < 0.05, permutation test).

---

## Statistical analysis

- **Paired t-tests** for within-modality comparisons (MSC vs baseline across CV folds).
- **Binomial tests** for cross-modal retrieval (Recall@1 vs chance).
- **Permutation tests** for cross-modal structural similarity (matched vs mismatched pairs).
- **Bonferroni correction** for multiple comparisons within each experiment family.
- **Effect sizes** (Cohen's d) reported for all comparisons.

---

## Failure modes and interpretations

### If H1 fails for a modality
The MSC instance for that modality does not capture semantic structure as well as the baseline. This could mean:
- The chosen bands/scales are not appropriate for that modality.
- The coherence measure is not the right structural feature.
- The baseline is too strong for linear classification to show a difference.

**Action**: Investigate alternative band/scale choices, nonlinear classifiers, or different coherence measures before abandoning the instance.

### If H2 fails (cross-modal retrieval at chance)
MSC embeddings from different modalities cannot be aligned by simple cosine similarity. This could mean:
- The scale axes are not comparable across modalities.
- The coherence values have different distributions across modalities.
- Cross-modal alignment requires learning (not just cosine similarity).

**Action**: Try learned alignment (CCA, Procrustes) instead of cosine similarity. Investigate scale normalization.

### If H3 fails (no structural similarity correlation)
MSC coherence profiles do not capture structural similarity across modalities. This would be a significant negative result for the MSC framework's cross-modal unification claim.

**Action**: Investigate whether the structural similarity is captured at a different level (e.g., topological features, information-theoretic measures) rather than phase coherence.

---

## Timeline

| Phase | Task | Priority |
|---|---|---|
| 1 | Implement image MSC instance (phase congruency) | Highest |
| 2 | Validate image MSC on CIFAR-10 | High |
| 3 | Implement sensor MSC instance (wavelet coherence) | Medium |
| 4 | Validate sensor MSC on UCI HAR | Medium |
| 5 | Implement text MSC instance (graph spectral coherence) | Lower |
| 6 | Validate text MSC on SST-2 | Lower |
| 7 | Cross-modal retrieval experiment | After ≥ 2 modalities validated |
| 8 | Cross-modal structural similarity experiment | After ≥ 2 modalities validated |

The image instance is the highest priority because:
1. Phase congruency is the most well-established phase-based structural feature.
2. The mathematical parallel to CBMPC is the clearest.
3. The Bifrost pipeline already has 2D FFT support.
4. CIFAR-10 is a well-established benchmark with clear baselines.
