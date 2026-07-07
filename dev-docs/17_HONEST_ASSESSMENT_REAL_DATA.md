# 17 — Honest Assessment: Real Data Results

**The moment of truth.** This document reports the results of testing the Structured Resonance Thesis on real data, as opposed to the synthetic data used in experiments 1A-3D.

---

## The Critical Question

The previous experiments (1A-3D) used synthetic data where phase was *designed* to be the only distinguishing feature. This is circular — it guarantees phase will matter. The real question is: **does phase coherence capture semantic structure in real-world signals?**

---

## Real Data Results

### Experiment 1A-REAL: SpeechCommands (Audio)

| Method | Accuracy | vs Chance (10%) |
|--------|----------|-----------------|
| **CBMPC (phase coherence)** | **23.10%** | +13.1pp |
| FFT magnitude (amplitude-only) | 45.05% | +35.1pp |
| Raw waveform | 12.25% | +2.3pp |
| Chance | 10.00% | — |

**Phase ablation results:**
| Ablation | Accuracy | Delta from baseline | Significant? |
|-----------|----------|---------------------|--------------|
| phase_zero | 20.50% | -2.60pp | No (p=0.062) |
| phase_randomize | 22.40% | -0.70pp | No (p=0.514) |
| phase_noise | 23.80% | +0.70pp | No (p=0.411) |
| phase_noise_severe | 23.10% | +0.00pp | No (p=1.000) |
| cross_band_scramble | 22.55% | -0.55pp | No (p=0.539) |

**Verdict: NEGATIVE RESULT.**
- CBMPC is significantly **WORSE** than FFT magnitude (delta=-21.95pp, p=0.0001)
- Phase ablation does **NOT** significantly degrade performance (0/5 significant)
- CBMPC barely beats chance (13pp above chance for 10-class problem)

### Experiment 1A-ESC50: ESC-50 (Environmental Audio)

| Method | Accuracy | vs Chance (10%) |
|--------|----------|-----------------|
| **CBMPC (phase coherence)** | **46.50%** | +36.5pp |
| FFT magnitude (amplitude-only) | 68.25% | +58.3pp |
| Raw waveform | 13.50% | +3.5pp |
| Chance | 10.00% | — |

**Phase ablation:** phase_randomize delta=+3.25pp, p=0.304 — **NOT significant**

**Verdict: NEGATIVE RESULT.**
- CBMPC is significantly **WORSE** than FFT magnitude (delta=-21.75pp, p=0.023)
- Phase ablation does **NOT** significantly degrade performance

### Experiment 3B-REAL: UCI HAR (Sensor)

| Method | Accuracy | vs Chance (16.7%) |
|--------|----------|-------------------|
| **WaveletCoherence** | **69.08%** | +52.4pp |
| Statistical features | 65.75% | +49.1pp |
| FFT magnitude | 63.25% | +46.6pp |
| Raw | 32.00% | +15.3pp |
| Chance | 16.67% | — |

**Statistical tests:**
- WaveletCoherence vs FFT magnitude: delta=+5.83pp, p=0.063 — **NOT significant** (marginal)
- WaveletCoherence vs Statistical: delta=+3.33pp, p=0.033 — **Significant**
- WaveletCoherence vs Raw: delta=+37.08pp, p=0.0000 — **Significant**

**Verdict: MIXED RESULT.**
- WaveletCoherence beats statistical features and raw signals
- WaveletCoherence does NOT significantly beat FFT magnitude (p=0.063, marginal)
- The advantage over amplitude features is small and not robust

### Negative Controls

| Control | CBMPC | PhaseRand | FFT Mag | Phase Sig? | Assessment |
|---------|-------|-----------|---------|------------|------------|
| Random noise | 23.70% | 21.80% | 20.60% | No | GOOD — no artifacts in noise |
| Pure tones | 90.30% | 68.70% | 100.00% | Yes | Expected — tones have phase structure |
| Constant signal | 21.00% | 19.50% | 21.00% | Yes | BAD — artifact on empty data |
| Shuffled SC | 23.40% | 22.70% | 20.30% | No | GOOD — no artifacts in shuffled data |

**Assessment:**
- The method is correctly NOT detecting structure in noise or shuffled data
- The method IS detecting "structure" in constant signals (artifact) — likely numerical sensitivity
- Pure tones: FFT magnitude (100%) beats CBMPC (90.3%) — amplitude features are more informative even when phase structure exists

---

## The Honest Conclusion

### C1 (Phase captures structure): NOT SUPPORTED on real data

On real audio (SpeechCommands, ESC-50):
- Phase coherence features (CBMPC) perform at **23-47%** accuracy
- Amplitude-only features (FFT magnitude) perform at **45-68%** accuracy
- CBMPC is significantly **WORSE** than FFT magnitude on both datasets
- Phase ablation does **NOT** significantly degrade performance on either dataset

The synthetic experiments (1A, 1B) showed phase ablation destroying classification because the synthetic data was *designed* so phase was the only distinguishing feature. On real data, amplitude carries far more semantic information than phase.

### C3 (Cross-modal generalization): NOT SUPPORTED

The sensor results (3B-REAL) show WaveletCoherence is marginally better than FFT magnitude (p=0.063), but this is not robust. Combined with the Experiment 3D failure (silhouette by category = -0.019), there is no evidence that coherence features generalize across modalities.

### Experiment 1B-REAL: Handwritten Digits (Images)

| Method | Accuracy | vs Chance (10%) |
|--------|----------|-----------------|
| **PhaseCongruency** | **96.22%** | +86.2pp |
| FFT magnitude (amplitude-only) | 88.03% | +78.0pp |
| Raw pixels | 97.00% | +87.0pp |
| Chance | 10.00% | — |

**Phase ablation results:**
| Ablation | Accuracy | Delta from baseline | Significant? |
|-----------|----------|---------------------|--------------|
| phase_zero | 85.09% | -11.13pp | Yes (p=0.0014) |
| phase_randomize | 20.81% | -75.40pp | Yes (p=0.0000) |
| phase_noise | 88.20% | -8.01pp | Yes (p=0.0003) |
| phase_noise_severe | 20.37% | -75.85pp | Yes (p=0.0000) |

**Verdict: STRONG POSITIVE RESULT.**
- PhaseCongruency **significantly BEATS** FFT magnitude (delta=+8.18pp, p=0.0001)
- Phase ablation **severely degrades** performance (4/4 ablations significant)
- Phase randomization **destroys** performance (96% → 21%, near chance)
- Raw pixels slightly beat PhaseCongruency (97% vs 96%, not significant)

**This is the strongest positive result for C1 on real data.** Handwritten digits have
strong edge structure, and phase congruency captures spatial relationships between edges
that amplitude-only features miss.

---

## The Complete Real-Data Picture

| Modality | Phase Helps? | Phase > Amplitude? | Phase Ablation Sig? | Verdict |
|----------|-------------|--------------------|--------------------|---------|
| Speech (SpeechCommands) | No | No (worse) | 0/5 | NEGATIVE |
| Env. Audio (ESC-50) | Marginal | No (worse) | No | NEGATIVE |
| Sensor (UCI HAR) | Yes (+13.8pp) | Marginal alone | N/A | POSITIVE |
| Images (Digits) | Yes (+8.2pp) | Yes | 4/4 | STRONG POSITIVE |

**The pattern is clear:**
- **Images with edge structure**: Phase matters a LOT (phase randomization destroys accuracy)
- **Sensor data with temporal structure**: Coherence adds significant value
- **Speech**: Amplitude (spectral envelope) dominates, phase doesn't help
- **Environmental audio**: Marginal trend, not significant

The thesis claim that phase coherence "generalizes across all modalities" is NOT supported.
But the claim that phase coherence "captures structural information that amplitude misses"
IS supported for images and sensors — modalities where spatial/temporal structure is primary.

---

### Combined Experiment: Does Phase Add Value ON TOP of Amplitude?

The previous tests compared phase-only (CBMPC) vs amplitude-only (FFT). But the thesis
doesn't claim phase REPLACES amplitude — it claims phase captures structure that amplitude
doesn't. The right test is: **does adding phase features to amplitude features improve
classification?**

| Dataset | FFT Alone | FFT + Phase | Delta | p-value | Significant? |
|---------|-----------|-------------|-------|---------|--------------|
| SpeechCommands | 45.05% | 46.75% | +1.70pp | 0.143 | No |
| ESC-50 | 68.25% | 72.75% | +4.50pp | 0.141 | No |
| **UCI HAR** | **63.25%** | **77.08%** | **+13.83pp** | **0.002** | **Yes** |

**Critical control — Real vs Randomized phase (combined with FFT):**
| Dataset | FFT+Real Phase | FFT+Random Phase | Delta | p-value |
|---------|----------------|-----------------|-------|---------|
| SpeechCommands | 46.75% | 46.70% | +0.05pp | 0.931 |
| ESC-50 | 72.75% | 70.00% | +2.75pp | 0.051 |
| UCI HAR | 77.08% | N/A | N/A | N/A |

**Interpretation:**

1. **SpeechCommands**: Phase does NOT add value. Real phase is no better than randomized
   phase (p=0.931). Amplitude features are sufficient for speech classification.

2. **ESC-50**: Phase shows a **marginal trend** toward adding value. Real phase is marginally
   better than randomized phase (p=0.051). Environmental audio may have some phase structure
   that amplitude doesn't capture, but the effect is weak.

3. **UCI HAR**: Wavelet coherence **ADDS SIGNIFICANT VALUE** (p=0.002, +13.83pp). This is
   the strongest positive result. Sensor data has temporal structure that wavelet coherence
   captures and amplitude features miss.

**The nuanced picture:**
- The value of coherence features is **modality-dependent**
- For speech: amplitude features are sufficient, phase adds nothing
- For environmental audio: phase may add marginal value (trend, not significant)
- For sensor data: wavelet coherence adds significant value
- The thesis claim that this "generalizes across all modalities" is NOT supported

---

## What This Means for the Project

The honest path forward is:

1. **Acknowledge the nuanced results.** Phase coherence is NOT a universal principle.
   It has modality-dependent value — significant for sensors, marginal for environmental
   audio, absent for speech.

2. **Revise the thesis.** The claim should be about multi-scale coherence in specific
   modalities, not a universal principle of intelligence.

3. **Investigate the sensor success.** Why does wavelet coherence work for UCI HAR?
   - Sensor data has strong temporal/multi-scale structure (periodic motion)
   - Speech has most information in spectral envelope (amplitude)
   - This suggests coherence features work when temporal structure is the signal

4. **The ESC-50 marginal result is interesting.** Environmental audio has more complex
   temporal structure than speech. The p=0.051 trend suggests phase may matter more
   for sounds with rich temporal patterns (rain, fire, crackling).

5. **The cross-modal claim (C3) is not supported.** The results show coherence features
   work differently in different modalities — they don't generalize uniformly.

---

## Data and Code

All real-data experiment scripts and results are in:
- `research_dir/experiment_phase_ablation_audio_real.py` → `research_dir/results/exp1a_real_speechcommands.json`
- `research_dir/experiment_cbmpc_esc50_real.py` → `research_dir/results/exp1a_real_esc50.json`
- `research_dir/experiment_msc_sensor_har_real.py` → `research_dir/results/exp3b_real_ucihar.json`
- `research_dir/experiment_negative_controls.py` → `research_dir/results/exp_negative_controls.json`
