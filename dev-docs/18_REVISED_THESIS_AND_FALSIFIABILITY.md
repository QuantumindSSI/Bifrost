# 18 — Revised Thesis and Falsifiability

**Date**: Following real-data experiments (document 17)

This document revises the Structured Resonance Thesis to match the actual evidence,
makes it falsifiable, and honestly addresses what is and isn't supported.

---

## The Original Thesis (OVERCLAIMED)

> "Intelligence is structured resonance. Semantic structure is encoded in the phase
> coherence of oscillatory components across multiple scales, and this principle
> generalizes across all modalities and all levels of intelligence."

**Status: NOT SUPPORTED by the evidence.**

This claim makes three assertions:
1. Phase coherence encodes semantic structure (C1)
2. Multi-scale structure is necessary (C2)
3. This generalizes across all modalities (C3)

The real-data experiments show:
- C1 is NOT supported for speech, marginal for environmental audio, supported for sensors
- C2 has literature support but was not tested on real data in this project
- C3 is NOT supported — coherence features have modality-dependent value

---

## The Revised Thesis (EVIDENCE-BACKED)

> **Multi-scale coherence features capture structural information that amplitude-only
> features miss, but this effect is modality-dependent: significant for sensor data
> with strong temporal structure, marginal for environmental audio, and absent for
> speech where spectral envelope dominates.**

This is a much weaker claim, but it is:
1. **Supported by real data** (UCI HAR: +13.83pp, p=0.002)
2. **Honest about limitations** (SpeechCommands: no benefit, ESC-50: marginal)
3. **Falsifiable** (see below)
4. **Consistent with the literature** (wavelet methods show modality-dependent value)

---

## Falsifiability

A scientific theory must be falsifiable. Here are the conditions that would falsify
each part of the revised thesis:

### Claim 1: "Multi-scale coherence captures structure that amplitude misses"

**Falsified if**: On a dataset where temporal/multi-scale structure is known to matter
(e.g., sensor data with periodic patterns), adding coherence features to amplitude
features does NOT improve classification beyond a trivial margin (<2pp) in any
well-powered experiment.

**NOT falsified if**: Coherence features add significant value even in one modality.
(Current status: SUPPORTED by UCI HAR, p=0.002)

### Claim 2: "The effect is modality-dependent"

**Falsified if**: Coherence features show the SAME magnitude of benefit across all
tested modalities (suggesting it's a universal principle, not modality-dependent).

**NOT falsified if**: Different modalities show different magnitudes of benefit.
(Current status: SUPPORTED — sensors +13.83pp vs speech +1.70pp)

### Claim 3: "Coherence features work when temporal structure is the signal"

**Falsified if**: Coherence features fail on data with strong temporal structure
(e.g., periodic motion data) OR succeed on data without temporal structure
(e.g., static images where spatial structure dominates).

**NOT falsified if**: Coherence features succeed specifically on temporally-structured data.
(Current status: PARTIALLY SUPPORTED — sensors work, speech doesn't, ESC-50 marginal)

---

## What We Can and Cannot Claim

### CAN claim (with evidence):
1. Wavelet coherence features add significant value for sensor activity recognition
   (UCI HAR: +13.83pp over FFT magnitude, p=0.002)
2. The value of coherence features is modality-dependent (sensors >> speech)
3. Phase coherence alone (without amplitude) is insufficient for real audio classification
4. On speech, amplitude features (spectrogram) capture most semantic information
5. The synthetic experiments were circular — they proved the data generator, not the thesis

### CANNOT claim (no evidence):
1. "Intelligence is structured resonance" — no evidence on intelligence tasks
2. "Phase coherence generalizes across all modalities" — explicitly contradicted
3. "Phase captures semantic structure" — only true for sensors, not audio
4. "Cross-modal alignment via coherence" — Experiment 3D failed
5. "The thesis applies to language models" — untested, and the foundation is weak

### MUST acknowledge:
1. The original thesis was overclaimed
2. The synthetic experiments were circular
3. The real-data results are mostly negative for the phase coherence claim
4. The only strong positive result is wavelet coherence on sensor data
5. The LM integration plan was premature

---

## Addressing the 3D Failure (C3)

Experiment 3D showed:
- Silhouette by modality: 0.873 (strong clustering by modality)
- Silhouette by category: -0.019 (clustering by category is WORSE than random)

This means the UnifiedCoherenceMetric completely fails to align modalities. The
modality gap (documented in CLIP by Liang et al., NeurIPS 2022) is a fundamental
geometric problem that our simple linear UCM cannot overcome.

**Honest assessment**: C3 (cross-modal generalization) is NOT supported. The coherence
features do not create a shared semantic space across modalities. This is consistent
with the modality-dependent results — if coherence features work differently in each
modality, they cannot be expected to align across modalities.

**What would be needed to support C3**:
1. Shared encoder weights across modalities (to close the modality gap)
2. Much larger datasets (ImageBind uses millions of pairs)
3. Nonlinear projections with sufficient capacity
4. Or: a fundamentally different approach to cross-modal binding

**Current status**: C3 is unsupported and should be dropped from the thesis claims
until these requirements are met.

---

## The Path Forward

### Short-term (honest research):
1. **Understand the sensor success.** Why does wavelet coherence work for UCI HAR?
   Is it the periodic motion structure? The multi-scale temporal patterns? This is
   the most interesting positive result and deserves investigation.

2. **Test on more sensor datasets.** Does the coherence advantage replicate on other
   sensor data (e.g., EEG, EMG, accelerometer for fall detection)?

3. **Test CIFAR-10 (when download completes).** Does phase congruency add value for
   real images, or is it also modality-dependent?

4. **Test the multi-scale claim (C2) on real data.** The literature supports
   multi-scale methods (Wavelet GPT), but we haven't tested C2 on real data.

### Medium-term (practical applications):
5. **Focus on sensor applications.** If coherence features work for sensors, build
   practical sensor classification systems.

6. **Spectral analysis of existing LMs.** This is independent of the thesis —
   the literature (Spectral Geometry of Thought) shows spectral structure in LMs
   regardless of whether phase coherence is the mechanism.

### Long-term (if warranted):
7. **Revisit cross-modal alignment** only if a fundamentally different approach
   is developed (shared encoders, large data, nonlinear projections).

8. **Revisit LM integration** only if the coherence features are shown to add
   value in at least one real-world modality beyond sensors.
