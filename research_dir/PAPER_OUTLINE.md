# Bifrost: Towards Semantic Structure from Continuous Spectral Signals

**Working title**: *Bifrost: Learning Multimodal Semantic Structure from Phase-Coherent Frequency Representations*  
**Target venues**: NeurIPS, ICML, ICLR, Interspeech, IEEE TSP, or multimodal-AI workshop track.  
**Paper type**: Method + empirical validation  
**Status**: Outline with pilot results. The main semantic hypothesis is not yet supported on real audio; the paper now includes a calibrated negative-result narrative.

---

## 1. Abstract

**Goal**: One paragraph that summarizes the positive contribution.

> We introduce Bifrost, a framework that learns multimodal representations from the structure of continuous signals rather than from discrete tokens. Bifrost converts audio, image, text, and sensor data into complex spectra and builds a hierarchy of structural representations: phase coherence, compositional timescales, directed causal influence, topological fingerprints, qualitative temporal relations, detected symmetries, and disentangled factors. We present the architecture, a self-driving research loop for validating each layer, and the first empirical probe of the core claim: that phase coherence correlates with semantic category structure. On a synthetic audio task the model learns expected frequency-defined categories, but on real speech commands a fine-tuned Bifrost classifier does not outperform a simple STFT-magnitude baseline in a pilot cross-validation study. We report these results transparently, outline the pre-registered protocol needed to test the hypothesis decisively, and discuss which architectural assumptions must be revised before strong semantic claims can be made.

**Notes**:
- The final abstract will be rewritten once experiments are complete.
- Do not claim "semantic understanding" unless at least three layers and one cross-modal task are validated.

---

## 2. Introduction

### 2.1 Motivation

- Current deep learning is dominated by token-based models (transformers, patch tokens, mel-spectrogram tokens).
- Tokenization is lossy: it discards phase, fine temporal dynamics, and continuous amplitude structure.
- Many meaningful properties of signals are inherently continuous: pitch, timbre, rhythm, causality, symmetry.
- Hypothesis: semantic structure can be built from the structural layers of continuous signals, not just from discrete tokens.

### 2.2 The Bifrost thesis

- Semantic understanding is not a single statistical property but a hierarchy of structural patterns.
- The seven layers: distributional, compositional, causal, topological, temporal, symmetry, disentanglement.
- Each layer requires a different mathematical object: phase coherence, hierarchical SSM, Granger causality, persistent homology, Allen interval algebra, stabiliser groups, TC-VAE.
- Bifrost processes all modalities through the same continuous spectral pipeline.

### 2.3 Positive framing

- Bifrost is not a replacement for LLMs but a complement.
- It provides:
  - A multimodal encoder that does not tokenize early.
  - A physically grounded similarity signal (phase coherence).
  - A structural verifier for LLM reasoning.
  - A path toward representations that preserve continuous structure.

### 2.4 Research questions

1. Does phase coherence correlate with semantic similarity on real multi-modal data?
2. Can a hierarchical spectral SSM capture compositional temporal structure?
3. Can Granger causality on SSM states recover directed influence?
4. Do topological fingerprints improve semantic classification?
5. Can Allen interval relations capture temporal narrative structure?
6. Can symmetry detection generalize across modalities?
7. Can TC-VAE disentangle content, style, and temporal factors in real audio?

### 2.5 Contributions

- A unified continuous-spectral pipeline for multimodal representation learning.
- Seven concrete structural representations derived from signal-level primitives.
- Empirical validation across [datasets to be determined].
- A public codebase and reproducible benchmark suite.

---

## 3. Related work

### 3.1 Continuous and complex-valued representations

- Deep Complex Networks (Trabelsi et al., 2018).
- Complex-valued neural networks for audio and speech.
- Fourier neural operators.

### 3.2 State space models

- Mamba / S4 / S5 (Gu & Dao, 2023; Smith et al., 2023).
- Linear-time sequence modeling.
- Selective state spaces.

### 3.3 Phase and oscillator models

- Adler equation (1946) and coupled oscillator theory.
- Synchrony in neuroscience and signal processing.
- Phase coherence in speech and audio (e.g., CASA, harmonicity).

### 3.4 Causal inference on time series

- Granger causality.
- Causal discovery in neural time series.
- Directed acyclic graphs and intervention.

### 3.5 Topological data analysis

- Persistent homology.
- TDA for time series and audio.
- Differentiable topology.

### 3.6 Temporal reasoning

- Allen interval algebra (1983).
- Temporal event understanding.
- Narrative structure.

### 3.7 Symmetry and invariance

- Group theory in deep learning.
- Harmonic structure in music.
- Equivariant networks.

### 3.8 Disentangled representations

- β-VAE, β-TC-VAE.
- DCI disentanglement metric.
- Content-style separation in speech and music.

### 3.9 Multimodal learning

- Audio-visual correspondence.
- Cross-modal retrieval.
- Multimodal transformers.

---

## 4. Method

### 4.1 Pipeline overview

Present the S0→S4 pipeline as in the README and Engineering Plan.

- **S0 — Canonicalization**: STFT, complex spectrum, normalization, uncertainty.
- **S1 — Complex SSM**: `h[t] = exp(−Δ·A)·h[t−1] + Δ·B[t]·x[t]`; parallel associative scan.
- **S2 — Spectral Binding**: multi-band phase coherence; harmonic binding; collapse-proof attention.
- **S3 — Phase-Lock Bridge**: stable attractors via Adler dynamics; VQ-VAE codebook.
- **S4 — Riemannian Coherence**: learned metric `G = LLᵀ`; geodesic distance.

### 4.2 The seven structural representations

For each layer, describe:

- **Input**: what spectral tensor it consumes.
- **Operation**: the mathematical transformation.
- **Output**: the tensor representation.
- **Claim**: what semantic structure it is supposed to capture.

| Layer | Component | Input | Output | Claim |
|---|---|---|---|---|
| L1 | `ResonanceAttention` | SpectralTensor | Coherence matrix | Statistical co-occurrence |
| L2 | `HierarchicalComplexSSM` | SpectralTensor | Multi-timescale states | Compositional part-whole structure |
| L3 | `GrangerCausalityExtractor` | SSM hidden states | `CausalGraphTensor` | Directed influence |
| L4 | `TDAPersistenceExtractor` | Amplitude surface | `PersistenceTensor` | Topological shape |
| L5 | `AllenRelationExtractor` | Attractor activations | `TemporalRelationTensor` | Event relations |
| L6 | `SymmetryDetector` | SpectralTensor | `SymmetryTensor` | Invariance groups |
| L7 | `TCVAEEncoder` | Attractor features | `DisentangledTensor` | Independent factors |

### 4.3 Integration with language models

- Spectral prefix.
- Parameter-efficient adapter.
- Structural coherence verifier.

### 4.4 Loss functions

- Contrastive phase loss.
- Ratio loss / collapse prevention.
- Riemannian triplet loss.
- TDA Wasserstein distance (if used).
- TC-VAE loss.

### 4.5 Implementation details

- PyTorch 2.3+, complex-valued operations.
- Distributed training with DDP.
- Checkpointing and reproducibility.

---

## 5. Experimental design

### 5.1 Datasets

| Experiment | Dataset | Rationale |
|---|---|---|
| Phase coherence semantic correlation | ESC-50, VGGSound | Real audio categories |
| Hierarchical boundary detection | Switchboard / LibriSpeech | Speech compositional structure |
| Granger causality | EEG causal benchmark / synthetic VAR | Known directed influence |
| TDA classification | NSynth, VGGSound | Instrument / phoneme topology |
| Temporal relations | TimeBank, synthetic intervals | Event ordering |
| Symmetry | NSynth, TUT Sound Events | Octave, periodicity, invariance |
| Disentanglement | VCTK, Nsynth | Speaker vs. content |
| Cross-modal retrieval | VGGSound / AudioCaps + images/captions | Audio ↔ visual/text correspondence |
| Reasoning verifier | StrategyQA / GSM8K CoT | Hallucination reduction |

### 5.2 Baselines

For each experiment, define a strong baseline that isolates the Bifrost contribution:

- Standard spectrogram + CNN or transformer.
- Mel-filterbank + wav2vec 2.0 or similar pretrained model.
- Flat SSM (no hierarchy).
- Random graph / random features.
- Standard VQ-VAE (no disentanglement).

### 5.3 Metrics

- Classification accuracy, F1, Recall@K.
- Pearson / Spearman correlation.
- Effect size (Cohen's d).
- Statistical significance (p-value, confidence interval).
- DCI disentanglement score.
- Betti number stability.

### 5.4 Statistical protocol

- Run each experiment with at least 5 random seeds.
- Report mean and standard error.
- Use paired non-parametric tests (Wilcoxon signed-rank) when comparing Bifrost vs. baseline.
- Report effect sizes, not just p-values.
- Pre-register hypotheses to avoid HARKing.

---

## 6. Results (pilot and pending)

### 6.1 Main results table

| Layer | Dataset | Metric | Bifrost | Baseline | p-value | Effect size | Status |
|---|---|---|---|---|---|---|---|
| L1 | Synthetic tones (5 classes) | Test accuracy | 0.98 | chance = 0.20 | < 0.001 | very large | Sanity check; trivial frequency separation |
| L1 | SpeechCommands (5 classes, pilot) | Test accuracy (3-fold CV) | 0.28 ± 0.08 | STFT mag = 0.35 ± 0.09 | 0.169 | small | **Not supported** |
| L1 | SpeechCommands (5 classes, frozen) | Test accuracy (3-fold CV) | 0.20 ± 0.01 | chance = 0.20 | — | — | No discriminative structure in untrained Bifrost |
| L2 | Switchboard | Boundary F1 | TBD | TBD | TBD | TBD | Pending |
| L3 | EEG | Edge precision | TBD | TBD | TBD | TBD | Pending |
| L4 | NSynth | Accuracy | TBD | TBD | TBD | TBD | Pending |
| L5 | TimeBank | Relation accuracy | TBD | TBD | TBD | TBD | Pending |
| L6 | NSynth | Invariance accuracy | TBD | TBD | TBD | TBD | Pending |
| L7 | VCTK | DCI score | TBD | TBD | TBD | TBD | Pending |
| Cross-modal | VGGSound | Recall@10 | TBD | TBD | TBD | TBD | Pending |
| Reasoning | GSM8K | Accuracy gain | TBD | TBD | TBD | TBD | Pending |

### 6.2 Ablation studies

- Effect of each SSM timescale in hierarchical model.
- Effect of phase vs. amplitude in ResonanceAttention.
- Effect of TDA dimension (β₀, β₁, β₂).
- Effect of TC penalty weight β.

### 6.3 Qualitative analysis

- Visualize attention maps on harmonic vs. inharmonic signals.
- Visualize causal graphs on known EEG ground truth.
- Visualize persistence diagrams for different phonemes / chords.
- Show cross-modal retrieval examples.

---

## 7. Discussion

### 7.1 What the results would mean

If the experiments support the hypotheses:

- Frequency-level representations can capture multiple kinds of semantic structure.
- Phase coherence is a meaningful cross-modal signal.
- Continuous representations complement token-based models.

### 7.2 Null results and negative evidence

The first empirical probe of the L1 hypothesis produced a null result on real audio. A fine-tuned Bifrost classifier did not outperform a simple STFT-magnitude baseline in a 3-fold pilot on SpeechCommands (mean accuracy 0.28 vs. 0.35; p = 0.169). The untrained Bifrost representation was at chance (0.20 ± 0.01), and larger-scale fine-tuning led to severe overfitting (train 0.99, test 0.16). These findings are reported as first-class evidence because they constrain the theory: either the current architecture does not naturally encode semantic category structure, or the chosen pooling, objective, and training regime are not aligned with the target task. The hypothesis remains open but is not supported by the current evidence.

### 7.3 Limitations

- Granger causality is predictive, not intervention-based.
- TDA is computationally expensive for long sequences.
- Disentanglement on real audio is difficult without strong supervision.
- Symmetry detection may be brittle for non-musical signals.
- The full seven-layer pipeline is large and requires significant compute.
- The L1 pilot is underpowered (50 samples per class) and uses a single embedding-pooling strategy; a full pre-registered protocol is needed.
- Early experiments were exploratory and involved post-hoc protocol adjustments (HARKing); the pre-registered baseline comparison is the first rigorous test.

### 7.4 Future work

- Scale to the planned 3PB multimodal corpus.
- Train the full pipeline end-to-end.
- Integrate with frozen LLMs for grounding and reasoning.
- Explore landmark TDA and faster causal inference.
- Develop domain-specific symmetry priors for speech, images, and sensors.

---

## 8. Conclusion

- Bifrost presents a unified research direction: building semantic representations from continuous signal structure.
- The framework is grounded in physics and mathematics, and each layer is testable.
- The first empirical probe of the phase-coherence hypothesis did not support the claim on real audio; we report this null result and the pre-registered protocol needed to test it decisively.
- The paper contributes a methodology for rigorously validating continuous-spectral representations, an architecture, and a transparent account of where the theory currently stands.
- The broader vision remains open: a multimodal AI system that does not need to tokenize the world to understand it, but this vision requires empirical validation rather than architectural assertion.

---

## 9. Reproducibility statement

- All code will be released at [repository URL].
- All checkpoints and configs will be versioned.
- Dataset preparation scripts will be included.
- Random seeds and hyperparameters will be reported.
- Statistical analysis code will be included.

---

## 10. References

Selected references to include:

- Gu & Dao (2023) — Mamba.
- Trabelsi et al. (2018) — Deep Complex Networks.
- Blelloch (1990) — Prefix sums.
- Adler (1946) — Oscillator locking.
- Nickel & Kiela (2017) — Poincaré embeddings.
- Pearl (2009) — Causality.
- Carlsson (2009) — Topology and Data.
- Allen (1983) — Temporal interval algebra.
- Higgins et al. (2017) — β-VAE.
- Chen et al. (2018) — β-TC-VAE.

Additional literature will be added by the background literature-search subagent.

---

## Appendix A: Hypothesis-to-experiment mapping

This maps each hypothesis in the research loop to a paper section.

| Hypothesis ID | Paper section | Expected evidence |
|---|---|---|
| `phase_coherence_semantic_correlation` | §5.1, §6.1 | Correlation table on ESC-50/VGGSound |
| `hierarchical_ssm_boundaries` | §4.2, §6.1 | F1 improvement on Switchboard |
| `granger_causality_asymmetry` | §4.2, §6.1 | Asymmetric edge ratio on EEG |
| `tda_instrument_discrimination` | §4.2, §6.1 | Classification accuracy on NSynth |
| `allen_temporal_relations` | §4.2, §6.1 | Relation accuracy on TimeBank |
| `symmetry_octave_detection` | §4.2, §6.1 | Invariance accuracy on NSynth |
| `disentanglement_speaker_content` | §4.2, §6.1 | DCI score on VCTK |
| `cross_modal_audio_image_retrieval` | §5.2, §6.1 | Recall@10 on VGGSound |
| `structural_verifier_reduces_hallucination` | §4.3, §6.1 | Accuracy gain on GSM8K |

---

## Appendix B: Author checklist

Before submission, verify:

- [ ] All claims match the experimental evidence.
- [ ] No simulated or placeholder results are reported as real.
- [ ] All baselines are strong and relevant.
- [ ] Statistical tests are correct and reported.
- [ ] Code is public and reproducible.
- [ ] At least two internal reviewers have reproduced the main experiments.
- [ ] At least two external pre-reviewers have read the paper.
- [ ] The abstract does not overstate the results.
- [ ] Limitations are clearly discussed.
