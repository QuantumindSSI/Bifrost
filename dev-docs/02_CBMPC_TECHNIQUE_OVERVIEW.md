# 02 — CBMPC Technique Overview

**Date**: July 2026  
**Full specification**: `research_dir/CBMPC_TECHNIQUE_PROPOSAL.md`  
**Implementation**: `src/bifrost/cbmpc.py`

---

## What CBMPC is

Cross-Band Modulation Phase Coherence (CBMPC) is a frequency-level semantic structure extractor that measures the phase coherence of temporal modulations across spectral bands.

## Why it was developed

The original Bifrost hypothesis conflated two different notions of phase:

1. **Spectral phase**: the phase of individual frequency components at a given time. This is what the Bifrost pipeline preserves.
2. **Modulation phase**: the phase of temporal modulations of spectral energy. This is what carries semantic structure in speech.

The literature is clear that (2) carries semantic structure and (1) does not:
- Greenberg & Arai (2001): speech intelligibility correlates with the complex modulation spectrum, not spectral amplitude alone.
- Drullman et al. (1994): smearing temporal modulations below 4 Hz destroys intelligibility.
- Chi et al. (1999): spectrotemporal modulation transfer functions predict speech intelligibility.
- Elhilali et al. (2009): temporal coherence across frequency bands organizes auditory perception.

CBMPC operationalizes (2) as a computable feature vector.

## Mathematical formulation

Given a spectrogram S(t, f):

1. **Log-compress**: L(t, f) = log(|S(t, f)| + ε)
2. **Modulation spectrum**: For each frequency band f, compute FFT along time: L̃(ω_t, f) = FFT_t{L(t, f)}
3. **Extract modulation amplitude and phase**: A(ω_t, f) = |L̃(ω_t, f)|, φ(ω_t, f) = ∠L̃(ω_t, f)
4. **Cross-band phase locking value**: C(ω_t) = |(1/F) Σ_f exp(i·φ(ω_t, f))|
5. **Feature vector**: [per-band modulation amplitudes, PLV values, mean amplitudes]

The PLV ranges from 0 (completely incoherent) to 1 (perfectly phase-locked across bands).

## Why it captures semantic structure

1. **Speech categories have characteristic modulation rates**: stop consonants produce rapid onsets (20–40 Hz), vowels produce slow modulations (2–8 Hz), fricatives produce sustained high-frequency energy with low modulation.
2. **Formant transitions are cross-band coherent**: a formant transition produces a modulation that is phase-locked across multiple frequency bands.
3. **Modulation phase carries timing information**: Greenberg & Arai (2001) showed that reversing modulation phase at 100 ms segments reduces intelligibility to 4%.
4. **Cross-band coherence separates sources**: Elhilali et al. (2009) showed that temporal coherence across bands is the basis for auditory stream segregation.

## Implementation

`src/bifrost/cbmpc.py` provides:
- `CBMPCExtractor`: Pure signal-processing module (no learned parameters). Takes raw audio waveforms and returns the CBMPC feature vector. Supports "compact" (14 dims) and "rich" (462 dims) modes.
- `CBMPCClassifier`: CBMPC extractor + linear classifier.

## Validated results

**Dataset**: SpeechCommands v0.02, 10 classes, 200 samples/class, 5-fold stratified CV.

| Model | Accuracy | F1 | Features |
|---|---|---|---|
| **CBMPC-STFT** | **0.41 ± 0.04** | **0.38 ± 0.04** | 462 |
| STFT baseline | 0.27 ± 0.01 | 0.22 ± 0.02 | 513 |
| Mel baseline | 0.25 ± 0.01 | 0.20 ± 0.02 | 64 |
| Bifrost amp-only | 0.16 ± 0.02 | 0.11 ± 0.03 | 128 |
| CBMPC-Bifrost | 0.10 ± 0.00 | 0.04 ± 0.02 | 462 |

**H2 (CBMPC-STFT beats STFT baseline): SUPPORTED**
- +13.65 pp, p = 0.0033, Bonferroni-corrected α = 0.0167, Cohen's d ≈ 3.5

**H1 (CBMPC-Bifrost beats STFT baseline): NOT SUPPORTED**
- CBMPC-Bifrost at chance — the Bifrost SSM destroys modulation structure.

## Theoretical grounding

| Reference | Contribution |
|---|---|
| Chi et al. (1999, JASA) | Spectrotemporal modulation transfer functions predict speech intelligibility |
| Drullman et al. (1994, JASA) | Temporal envelope modulations below 4 Hz are critical for intelligibility |
| Greenberg & Arai (2001, Eurospeech) | Complex modulation spectrum (amplitude + phase) correlates with intelligibility |
| Elhilali et al. (2009, Neuron) | Temporal coherence across frequency bands organizes auditory perception |
| Lachaux et al. (1999, Human Brain Mapping) | Phase locking value as a measure of synchrony |
| Hegde et al. (2007, EURASIP) | Phase-derived features (group delay) improve phoneme recognition |

## What CBMPC is not

- It is not raw spectral phase coherence (the original Bifrost L1 hypothesis).
- It is not a neural model (it is a signal processing technique with no learned parameters).
- It is not a replacement for the Bifrost pipeline (it is a feature extraction layer).
- It is not a black-box embedding (each feature corresponds to a specific modulation rate and frequency band).

## Open questions

1. Does the PLV component contribute beyond the per-band modulation amplitudes? (Ablation needed.)
2. Are the 7 chosen modulation frequencies (0.5, 1, 2, 4, 8, 16, 32 Hz) optimal?
3. Does CBMPC generalize to environmental sounds (ESC-50), music (NSynth), or cross-modal tasks?
4. Can a modulation-preserving SSM be designed that maintains both SSM temporal tracking and CBMPC modulation coherence?
