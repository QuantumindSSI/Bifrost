# Bifrost Research Dossier

**Project**: Bifrost — Spectral neural processing with phase-coherent representations  
**Vision**: Establish structural intelligence across data modalities for AI, AGI, and ASI  
**Thesis**: Intelligence is structured resonance — semantic structure is encoded in the phase coherence of oscillatory components across multiple scales, and this principle generalizes across all modalities and all levels of intelligence.  
**Maintainer**: Oluwaferanmi Oluwagbamila (Type Ω Epistemic Intelligence)  
**Status**: Active research — first positive result achieved (CBMPC, H2 supported); MSC framework defined; AGI/ASI path mapped  
**Last updated**: July 2026

---

## Purpose of this directory

This directory is the canonical development documentation for the Bifrost research program. It is distinct from `research_dir/`, which holds experimental scripts, raw results, and pre-registration documents. The `dev-docs/` directory holds:

1. **Synthesized research findings** — what we know, what we don't know, and with what confidence.
2. **Implementation roadmaps** — concrete engineering plans for the next iteration.
3. **Architectural decisions** — why specific designs were chosen or rejected, with evidence.
4. **Epistemic audit trail** — the adversarial self-review process applied to every claim.

All documents in this directory are kept in the repository for peer review and reproducibility.

---

## Document index

| Document | Purpose | Status |
|---|---|---|
| `RESEARCH_DOSSIER.md` | This file — top-level synthesis and navigation | Living document |
| `01_EPISTEMIC_AUDIT_SUMMARY.md` | Summary of the Type Ω audit of the first validation loop | Complete |
| `02_CBMPC_TECHNIQUE_OVERVIEW.md` | Overview of the new CBMPC technique and its theoretical grounding | Complete |
| `03_CBMPC_PRE_SSM_INTEGRATION_PLAN.md` | Implementation plan: integrating CBMPC as a pre-SSM feature extraction layer | Complete |
| `04_MODULATION_PRESERVING_SSM_INVESTIGATION.md` | Investigation of SSM architectures that preserve modulation structure | Complete |
| `05_ESC50_GENERALIZATION_TEST_PLAN.md` | Protocol for testing CBMPC generalization on ESC-50 | Complete |
| `06_MSC_FRAMEWORK.md` | Multi-Scale Structural Coherence framework — the unifying principle | Complete |
| `07_MSC_MODALITY_INSTANCES.md` | Modality-specific MSC instances (audio, image, sensor, text) | Complete |
| `08_CROSS_MODAL_VALIDATION_PROTOCOL.md` | Pre-registered protocol for cross-modal validation | Complete |
| `09_RESEARCH_PATHS_COMPENDIUM.md` | Complete catalog of all research directions | Complete |
| `10_FREQUENCY_LEVEL_DATA_MODELS.md` | Survey of frequency representations and their structural priors | Complete |
| `11_AGI_ASI_STRUCTURAL_INTELLIGENCE.md` | Strategic synthesis: Bifrost's path to AGI/ASI | Complete |
| `12_LITERATURE_SURVEY_EXTERNAL.md` | External research papers grounding the project | Complete |
| `13_ENGINEERING_REQUIREMENTS.md` | Engineering required to prove the Bifrost thesis | Complete |
| `14_REFINED_ENGINEERING_PLAN_STEPS_1_3.md` | Concrete plan for steps 1-3 (minimal viable proof) | Complete |
| `15_ENGINEERING_LESSONS.md` | Non-obvious insights from proving steps 1-3 | Complete |
| `16_LM_INTEGRATION_RESEARCH.md` | Research survey: integrating structured resonance into language models | Complete |
| `17_HONEST_ASSESSMENT_REAL_DATA.md` | Real-data experiments: honest negative/mixed results | Complete |
| `18_REVISED_THESIS_AND_FALSIFIABILITY.md` | Revised thesis matching evidence, falsifiability criteria | Complete |
| `19_LM_REASONING_EXPERIMENT.md` | Phase coherence vs spectral alpha in LLM hidden states | Complete |
| `20_SPECTRAL_ALPHA_MONITORING_NEGATIVE.md` | Real-time alpha monitoring: negative results on 0.5B model | Complete |
| `21_WAVELET_AUGMENTATION_NEGATIVE.md` | Wavelet augmentation fine-tuning: negative result | Complete |
| `22_FINAL_RESEARCH_SUMMARY.md` | Consolidation of all findings + literature roadmap | Complete |
| `23_SPECTRAL_AGENTIC_ROADMAP.md` | Agentic implementation plan + SRA architecture proposal | Complete |

---

## Current state of the project

### What we know (calibrated confidence)

**UPDATED July 2026 following real-data experiments (docs 17-18):**

1. **Phase coherence does NOT capture semantic structure in real speech** (confidence: very high). On SpeechCommands, CBMPC achieves 23.10% vs FFT magnitude 45.05% — CBMPC is significantly WORSE (p=0.0001). Phase ablation does not significantly degrade performance (0/5 ablations significant). The synthetic experiments were circular.

2. **Phase coherence does NOT capture semantic structure in real environmental audio** (confidence: high). On ESC-50, CBMPC achieves 46.50% vs FFT magnitude 68.25% — CBMPC is significantly WORSE (p=0.023). Phase ablation is not significant (p=0.304).

3. **Wavelet coherence ADDS SIGNIFICANT VALUE for sensor data** (confidence: high). On UCI HAR, FFT+WaveletCoherence achieves 77.08% vs FFT alone 63.25% — a significant improvement (+13.83pp, p=0.002). This is the strongest positive result in the project.

4. **The value of coherence features is modality-dependent** (confidence: high). Sensors: significant benefit. Environmental audio: marginal trend (p=0.051). Speech: no benefit. The thesis claim of universal generalization is NOT supported.

5. **Cross-modal alignment (C3) is NOT supported** (confidence: high). Experiment 3D showed silhouette by category = -0.019 (worse than random). The modality gap is a fundamental geometric problem.

6. **The original thesis was overclaimed** (confidence: very high). "Intelligence is structured resonance" is not supported by the evidence. The revised thesis (doc 18) makes much weaker, evidence-backed claims about modality-dependent coherence.

7. **CBMPC captures semantic structure in speech** (confidence: moderate, from earlier work). CBMPC-STFT achieves 0.41 on SpeechCommands vs 0.27 STFT baseline (p=0.0033). NOTE: This earlier result used a different CBMPC implementation. The real-data experiments (doc 17) used the current implementation and found CBMPC at 23.10%. This discrepancy needs investigation.

8. **CBMPC does NOT generalize to environmental sounds** (confidence: high, confirmed by real-data experiment). On ESC-50, CBMPC is worse than STFT baseline.

### What we don't know

1. Whether CBMPC generalizes to phoneme recognition (TIMIT) or other speech tasks beyond command classification.
2. Whether the PLV component of CBMPC contributes beyond the per-band modulation amplitudes (ablation not yet run).
3. Whether a modulation-preserving SSM architecture can add value on top of CBMPC for speech tasks.
4. Whether CBMPC features are complementary to or redundant with self-supervised audio embeddings (wav2vec 2.0, HuBERT).
5. Whether the 7 modulation frequencies we chose (0.5, 1, 2, 4, 8, 16, 32 Hz) are optimal for speech.
6. Whether a nonlinear classifier (MLP, CNN) on CBMPC features would dramatically improve ESC-50 performance.

### What we are committing to

1. **Pre-registration before examination.** No future experiment will be run without a written protocol specifying hypotheses, success criteria, and stopping rules.
2. **Baseline comparisons.** Every Bifrost/CBMPC result will be compared against at least one strong baseline on the same data and split.
3. **Honest reporting.** Null and negative results will be reported as first-class evidence.
4. **Cross-validation.** No single-split results will be reported as confirmatory.

---

## Research artifacts in the repository

| Artifact | Location | Description |
|---|---|---|
| Epistemic audit | `research_dir/EPISTEMIC_AUDIT.md` | Type Ω self-review of the first validation loop |
| CBMPC technique proposal | `research_dir/CBMPC_TECHNIQUE_PROPOSAL.md` | Full specification, pre-registration, and results |
| Baseline comparison experiment | `research_dir/experiment_phase_coherence_baseline_comparison.py` | Original Bifrost vs. STFT baseline |
| CBMPC comparison experiment | `research_dir/experiment_cbmpc_comparison.py` | Pre-registered CBMPC vs. baselines |
| CBMPC extractor implementation | `src/bifrost/cbmpc.py` | The CBMPC feature extraction module |
| Research synthesis | `research_dir/BIFROST_RESEARCH_SYNTHESIS.md` | Living synthesis of all findings |
| Paper outline | `research_dir/PAPER_OUTLINE.md` | Draft paper structure with calibrated claims |
| Literature survey | `research_dir/LITERATURE_SURVEY.md` | Peer-reviewed references per layer |
| Self-driving research loop | `research_dir/SELF_DRIVING_RESEARCH_LOOP.md` | Methodology for autonomous validation |
| Results JSON | `research_dir/results/` | All experimental results as JSON |

---

## Key numbers

| Metric | Value | Source |
|---|---|---|
| CBMPC-STFT accuracy (10-class SpeechCommands) | 0.41 ± 0.04 | `results/cbmpc_baseline_comparison.json` |
| STFT baseline accuracy (SpeechCommands) | 0.27 ± 0.01 | Same |
| Mel baseline accuracy (SpeechCommands) | 0.25 ± 0.01 | Same |
| Bifrost amp-only accuracy (SpeechCommands) | 0.16 ± 0.02 | Same |
| CBMPC-Bifrost accuracy (SpeechCommands) | 0.10 ± 0.00 (chance) | Same |
| H2 p-value (CBMPC-STFT vs STFT, SpeechCommands) | 0.0033 | Same |
| H2 effect size (Cohen's d, SpeechCommands) | ~3.5 | Same |
| CBMPC+SSM combined accuracy (5-class pilot) | 0.47 ± 0.04 | `results/cbmpc_pre_ssm_integration.json` |
| CBMPC-only accuracy (5-class pilot) | 0.44 ± 0.08 | Same |
| CBMPC-STFT accuracy (50-class ESC-50) | 0.12 ± 0.02 | `results/cbmpc_esc50_comparison.json` |
| STFT baseline accuracy (ESC-50) | 0.16 ± 0.03 | Same |
| Mel baseline accuracy (ESC-50) | 0.21 ± 0.04 | Same |
| ESC-50 H1 (CBMPC beats STFT) | NOT SUPPORTED (−3.95 pp) | Same |
| Bonferroni-corrected alpha (SpeechCommands) | 0.0167 | Pre-registered |
