# Type Ω Epistemic Audit: Phase Coherence vs. Semantic Similarity

**Date**: July 2026  
**Auditor**: Oluwaferanmi Oluwagbamila (Type Ω Epistemic Intelligence)  
**Scope**: First experimental validation loop of the Bifrost hypothesis  
**Status**: Exploratory → Protocol correction required before confirmatory claims

---

## 1. The Hypothesis as Currently Stated

**Original claim**: Bifrost phase coherence correlates with semantic category similarity on real audio.

**Operational form tested**: Given a Bifrost spectral embedding for a short audio clip, pairwise embedding cosine similarity should correlate with a binary same/different category indicator.

**Target thresholds**: Pearson r > 0.3, p < 0.05, classification accuracy > 0.6.

---

## 2. Bayesian Update on the Hypothesis

| Experiment | Prior odds | Evidence | Posterior assessment |
|---|---|---|---|
| Untrained synthetic (5 classes, n=200) | Weak positive plausible | r = 0.14, p < 0.001, but below threshold | Slightly favors null; not decisive |
| Fine-tuned synthetic (5 classes, n=300) | Moderate positive | r = 0.92, acc = 0.98, on trivially separable synthetic tones | Strongly favors separability of synthetic task, not general claim |
| Frozen SpeechCommands (5 classes, n=150) | Weak positive | acc = 0.13–0.22, r ≈ 0.01 | Strongly favors null for untrained representations on real data |
| Fine-tuned SpeechCommands (5 classes, n=150) | Moderate positive | train acc = 1.00, test acc = 0.33, r_test = 0.007 | Strong evidence of overfitting; no generalization |
| Fine-tuned SpeechCommands (10 classes, n=500) | Moderate positive | train acc = 0.99, test acc = 0.16, r_test = 0.07 | Confirmed overfitting at larger scale; no generalization |

**Calibrated posterior**: The phase-coherence → semantic-similarity hypothesis is **not supported** on real audio at the current model scale and training regime. The synthetic results are expected because the classes are defined by non-overlapping frequency structures. The real-data results show either (a) the Bifrost representation is too weak without substantial training data, or (b) the current pipeline architecture is not a natural feature extractor for spoken-word semantics.

**Confidence**: Low-to-moderate. The evidence is internally consistent but the experimental design has multiple flaws that prevent a decisive test.

---

## 3. Methodological Flaws Identified (Adversarial Self-Review)

### 3.1 HARKing / post-hoc protocol modification

The initial experiment was untrained and produced r = 0.14. After observing this, the protocol was changed to add end-to-end training, a classification head, and a different dataset. This is **exploratory**, not confirmatory. The strong synthetic result was obtained on a task designed after seeing the weak initial result.

**Correction**: All future experiments must be pre-registered before data is examined. The pre-registration must specify the model, dataset, splits, metrics, and stopping criteria.

### 3.2 No baseline comparison

There is no control condition comparing Bifrost to:
- Raw STFT magnitude + linear classifier
- Mel-spectrogram + linear classifier
- A standard pre-trained audio embedding (e.g., CLAP, wav2vec 2.0, YAMNet)

Without these, it is impossible to attribute any success or failure to Bifrost's phase-coherence mechanism rather than to generic spectral features or to a poorly calibrated classifier.

**Correction**: Every Bifrost experiment must include a minimal baseline on the same data and split.

### 3.3 Single train/test split

Results are reported on one random split. This makes the estimate unstable and prevents proper confidence intervals.

**Correction**: Use k-fold cross-validation (k ≥ 5) or repeated stratified splits with confidence intervals on the mean metric.

### 3.4 No regularization trajectory

The first real-data run used no dropout, no weight decay, and trained all parameters of a deep pipeline on 120 samples. Severe overfitting is the predictable outcome. The subsequent run added regularization but did not systematically vary it.

**Correction**: Pre-register a regularization sweep or at least a default regime validated on a held-out validation set, not the test set.

### 3.5 Dataset and sample size not justified by power analysis

No power analysis was performed. The choice of 5 classes, 30 per class was arbitrary. With 120 training samples and a 5-class problem, the classifier is underdetermined.

**Correction**: Pre-specify the smallest effect size of interest and compute the required sample size. For a 5-class accuracy baseline of 0.35 vs. null 0.20, with α = 0.05 and power = 0.80, roughly n ≈ 60–100 per class is needed (binomial exact test).

### 3.6 Embedding choice not pre-registered

The experiment used mean-pooled amplitude + std as the embedding. This was chosen post-hoc and not compared to alternatives (e.g., SSM hidden state, coherence matrix, learned projection).

**Correction**: Pre-register the embedding extraction method and justify it from the model's design principles.

### 3.7 No data augmentation

Audio clips were used as-is. Small datasets benefit from augmentation (time shift, pitch shift, noise, time masking).

**Correction**: Include a pre-registered augmentation pipeline or use a larger dataset.

---

## 4. What the Results Actually Show

### 4.1 Synthetic data

The fine-tuned Bifrost classifier can separate synthetic tones, harmonic stacks, and noise. This is a **sanity check**, not evidence for the semantic hypothesis. The classes are frequency-defined; any spectral model with a linear head should solve this.

### 4.2 Real data (SpeechCommands)

- **Untrained Bifrost**: no discriminative structure beyond chance for spoken-word categories.
- **Fine-tuned Bifrost**: memorizes the training set, fails to generalize.

**Interpretation**: The current Bifrost pipeline, trained from scratch on small speech datasets, does not naturally learn semantically meaningful representations. This could be due to:
1. Insufficient data for the model capacity.
2. The phase-coherence objective not being aligned with semantic category structure.
3. The embedding pooling strategy discarding useful temporal/phonetic information.
4. The classifier head being too shallow to exploit the representation.

These are testable alternatives, not conclusions.

---

## 5. Pre-Registered Protocol for the Next Iteration

### 5.1 Primary hypothesis (revised)

H1: A Bifrost pipeline trained with a cross-entropy objective on a real audio classification dataset achieves test accuracy significantly higher than both (a) an untrained Bifrost pipeline and (b) a raw STFT-magnitude baseline on the same dataset, when evaluated with stratified k-fold cross-validation.

H0: There is no difference in test accuracy among the three conditions.

### 5.2 Dataset

Google SpeechCommands v0.02. Use the 12 core command classes ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one") with at least 1,000 training examples per class. This avoids the underpowered small-sample problem.

### 5.3 Models

1. **Bifrost fine-tuned**: Full pipeline + temporal pooling + linear head. Dropout = 0.5, weight decay = 1e-4, d_model = 64, n_fft = 1024, duration = 1.0 s.
2. **Bifrost frozen**: Same pipeline frozen; only linear head trained.
3. **STFT baseline**: Log STFT magnitude (n_fft=1024) → mean-pool over time → linear head.
4. **Mel baseline**: 64-bin mel-spectrogram → mean-pool → linear head.

### 5.4 Evaluation

- Stratified 5-fold cross-validation.
- Metric: macro-averaged test accuracy and F1.
- Pairwise comparisons via paired t-test across folds (α = 0.05, Bonferroni-corrected for 3 comparisons).
- Report mean ± std, not a single point estimate.
- Stop training at the epoch with lowest validation loss on a validation fold; never use the test fold for model selection.

### 5.5 Success criteria

Bifrost fine-tuned must exceed the STFT baseline by at least 5 absolute percentage points in mean test accuracy, and the difference must be statistically significant after correction.

### 5.6 Failure modes and interpretations

- If Bifrost fine-tuned < STFT baseline: the current Bifrost architecture is not advantageously exploiting phase coherence for this task.
- If Bifrost frozen < STFT baseline: the untrained Bifrost representation is not a useful feature extractor.
- If no model exceeds chance: the task is too hard or the features are insufficient.

---

## 6. Immediate Actions Required

1. Implement the STFT and mel baselines.
2. Implement stratified k-fold evaluation.
3. Run the pre-registered protocol on SpeechCommands with ≥ 12 classes and sufficient samples per class.
4. Do not report any result as supporting the original hypothesis until the pre-registered success criteria are met.
5. Update the paper outline and hypothesis registry to reflect this null/uncertain state.

---

## 7. Meta-Epistemic Note

This audit itself is subject to the same standards. The flaws identified are common in early-stage exploratory research. The goal is not to discard the Bifrost framework but to prevent the strong synthetic result from being mistaken for evidence of the broader semantic claim. The hypothesis remains open; it is simply not yet supported by rigorous, generalizable evidence.

**Next deliverable**: Pre-registered protocol implemented as a runnable script with baseline comparisons and cross-validation.
