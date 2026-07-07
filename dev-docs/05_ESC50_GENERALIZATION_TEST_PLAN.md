# 05 — ESC-50 Generalization Test Plan

**Status**: Pre-registration  
**Goal**: Test whether CBMPC generalizes beyond speech commands to environmental sound classification.

---

## Motivation

The CBMPC technique was validated on SpeechCommands (spoken word classification). Speech commands have well-defined temporal modulation structure (syllable-rate onsets, formant transitions). The question is whether CBMPC generalizes to environmental sounds, which have different temporal structures:

- **Dog bark**: sharp onset, rapid decay, repetitive
- **Rain**: sustained, low modulation, broadband
- **Church bells**: periodic, harmonic, long decay
- **Crying baby**: quasi-periodic, variable modulation
- **Helicopter**: sustained, low-frequency rotor modulation

If CBMPC generalizes to ESC-50, the modulation phase coherence structure is a general semantic feature, not specific to speech. If it does not, CBMPC is speech-specific and the technique's scope is narrower than hoped.

## Dataset

**ESC-50**: Environmental Sound Classification dataset.
- 2000 environmental audio recordings.
- 50 classes, 40 clips per class.
- 5-second clips, 44.1 kHz sample rate.
- 5 folds (pre-defined in the dataset metadata).

Classes grouped by category:
- Animals: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow
- Natural soundscapes: rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm
- Human non-speech: crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking sipping
- Interior/domestic: door knock, mouse click, keyboard typing, door wood creaks, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking
- Exterior/urban: helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw

## Pre-registered protocol

### Hypothesis

**H1**: CBMPC-STFT achieves significantly higher test accuracy than the STFT magnitude baseline on ESC-50, using the dataset's pre-defined 5-fold cross-validation.

**H0**: There is no significant difference.

### Models

1. **CBMPC-STFT**: Raw STFT → CBMPC rich feature extraction → linear classifier.
2. **STFT magnitude baseline**: Log STFT magnitude → mean-pool → linear classifier.
3. **Mel baseline**: 64-bin mel-spectrogram → mean-pool → linear classifier.
4. **CBMPC-Bifrost**: Bifrost pipeline → CBMPC extraction on pipeline output → linear classifier (expected to fail, as on SpeechCommands).

### CBMPC configuration for ESC-50

ESC-50 clips are 5 seconds at 44.1 kHz, which is different from SpeechCommands (1 second at 16 kHz). The CBMPC extractor must be reconfigured:

- **Sample rate**: 44100 Hz (ESC-50 native) or resample to 16000 Hz for consistency with SpeechCommands.
- **Duration**: 5.0 seconds (ESC-50 native).
- **n_fft**: 2048 (larger window for lower-frequency environmental sounds).
- **hop_length**: 1024.
- **n_mels**: 64 (same as SpeechCommands for comparability).
- **Modulation frequencies**: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0] — add 64 Hz to capture the faster modulations in some environmental sounds (e.g., helicopter rotor, engine).

**Decision**: Resample to 16000 Hz for consistency. This loses some high-frequency content but enables direct comparison with the SpeechCommands results. A separate run at 44100 Hz can be done as a post-hoc analysis.

### Evaluation

- Use the ESC-50 pre-defined 5-fold cross-validation (not random splits).
- Metric: macro-averaged test accuracy and F1.
- Paired t-test between CBMPC-STFT and STFT baseline across the 5 folds.
- Bonferroni-corrected alpha = 0.05 / 2 = 0.025 (two primary comparisons: CBMPC vs STFT, CBMPC vs Mel).

### Success criterion

CBMPC-STFT must exceed the STFT magnitude baseline by ≥ 5 absolute percentage points in mean test accuracy, with p < 0.025 after Bonferroni correction.

### Failure modes and interpretations

- **CBMPC beats STFT on ESC-50**: modulation phase coherence generalizes to environmental sounds. The technique is not speech-specific. This strengthens the claim that CBMPC captures a general principle of auditory semantic structure.

- **CBMPC does not beat STFT on ESC-50**: modulation phase coherence is speech-specific. Environmental sounds may not have the same cross-band modulation coherence structure. This would narrow the claim but not invalidate the SpeechCommands result.

- **All models perform poorly (< 20% accuracy)**: 50-class classification with linear classifiers on 5-second clips is harder than 10-class on 1-second clips. A nonlinear classifier (MLP or CNN) may be needed. This would be documented as a task difficulty issue, not a technique failure.

- **CBMPC-Bifrost is at chance**: expected, consistent with the SpeechCommands finding that the SSM destroys modulation structure.

## Implementation steps

### Step 1: Download ESC-50

```python
# ESC-50 is available as a torchaudio dataset or direct download
# URL: https://github.com/karolpiczak/ESC-50
# Structure: audio/{fold}/{filename}.wav, meta/esc50.csv
```

### Step 2: Implement ESC-50 data loader

```python
def load_esc50(config):
    # Read meta/esc50.csv
    # Load audio files, resample to 16 kHz
    # Pad/truncate to 5.0 seconds
    # Return (signals, labels, class_names, fold_indices)
```

### Step 3: Run the pre-registered protocol

Use the same experiment script structure as `experiment_cbmpc_comparison.py`, adapted for ESC-50's pre-defined folds.

### Step 4: Report results

Regardless of outcome, report:
- Mean ± std accuracy and F1 for each model.
- Paired t-test results.
- Per-class accuracy (to identify which environmental sound categories benefit most from CBMPC).
- Comparison with SpeechCommands results.

## Additional analyses (post-hoc, not pre-registered)

1. **Per-category analysis**: which ESC-50 categories (animals, natural, human, interior, exterior) benefit most from CBMPC?
2. **Modulation frequency importance**: which modulation frequencies are most discriminative for ESC-50?
3. **44100 Hz vs 16000 Hz**: does the higher sample rate improve CBMPC performance?
4. **Nonlinear classifier**: does an MLP on CBMPC features improve over a linear classifier?

## Files to create

| File | Purpose |
|---|---|
| `research_dir/experiment_cbmpc_esc50.py` | ESC-50 experiment script |
| `research_dir/results/cbmpc_esc50_comparison.json` | Results JSON |

## Expected outcome

Based on the SpeechCommands result (+13.65 pp), I predict CBMPC will also outperform the STFT baseline on ESC-50, but with a smaller effect size (~5–10 pp). Environmental sounds have more diverse temporal structures than speech commands, so the modulation phase coherence may be less consistently informative. The natural soundscapes category (rain, wind, sea waves) may show the smallest benefit, as these are sustained sounds with low modulation structure.

**Confidence**: moderate. The prediction is based on the theoretical grounding (modulation structure is a general auditory principle) but the empirical evidence is currently limited to speech.

---

## Results

### Pre-registered protocol execution

ESC-50, 50 classes, 40 clips/class, pre-defined 5-fold CV, 30 epochs, batch size 32.

| Model | Test accuracy | F1 macro |
|---|---|---|
| Mel baseline | 0.21 ± 0.04 | 0.17 ± 0.04 |
| STFT baseline | 0.16 ± 0.03 | 0.12 ± 0.02 |
| CBMPC-STFT | 0.12 ± 0.02 | 0.09 ± 0.01 |

**H1 (CBMPC beats STFT by 5 pp): NOT SUPPORTED**
- Delta = −3.95 pp (CBMPC is worse than STFT)
- p = 0.08

### Interpretation

CBMPC does **not** generalize to environmental sounds. The modulation phase coherence structure that works for speech commands does not work for ESC-50. This is an important negative finding that narrows the scope of the technique.

Key observations:
1. **The mel baseline is strongest on ESC-50** (0.21), while it was the weakest on SpeechCommands (0.25 vs. STFT 0.27). This suggests that for environmental sounds, mel-scale frequency resolution is more important than modulation structure.
2. **CBMPC is worse than even the STFT baseline** on ESC-50, suggesting that the modulation features may be noise for environmental sounds that don't have the characteristic modulation structure of speech.
3. **All models perform poorly** (10–21% on 50 classes, chance = 2%), which is expected for linear classifiers on a 50-class problem. A nonlinear classifier would likely improve all models.

### Calibrated conclusion

CBMPC is **speech-specific**, not a general audio semantic structure extractor. The modulation phase coherence structure captures the temporal dynamics of speech (syllable rate, formant transitions, onset patterns) but does not capture the semantic structure of environmental sounds, which have more diverse and less predictable temporal patterns.

This narrows the claim but does not invalidate the SpeechCommands result. CBMPC is a validated technique for speech classification, not a universal audio feature extractor.

### Revised scope of CBMPC

- **Validated**: speech command classification (SpeechCommands, 10 classes, +13.65 pp over STFT baseline).
- **Not validated**: environmental sound classification (ESC-50, 50 classes, −3.95 pp vs. STFT baseline).
- **Future test**: phoneme recognition (TIMIT), which is closer to the speech domain where CBMPC was validated.
