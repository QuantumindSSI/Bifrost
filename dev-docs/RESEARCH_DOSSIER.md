# Bifrost Research Dossier

**Project**: Bifrost — Spectral neural processing with phase-coherent representations  
**Maintainer**: Oluwaferanmi Oluwagbamila (Type Ω Epistemic Intelligence)  
**Status**: Active research — first positive result achieved (CBMPC, H2 supported)  
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
| `03_CBMPC_PRE_SSM_INTEGRATION_PLAN.md` | Implementation plan: integrating CBMPC as a pre-SSM feature extraction layer | In progress |
| `04_MODULATION_PRESERVING_SSM_INVESTIGATION.md` | Investigation of SSM architectures that preserve modulation structure | In progress |
| `05_ESC50_GENERALIZATION_TEST_PLAN.md` | Protocol for testing CBMPC generalization on ESC-50 | In progress |

---

## Current state of the project

### What we know (calibrated confidence)

1. **Raw spectral phase coherence does not capture semantic structure in speech** (confidence: high). The literature is clear that modulation phase, not spectral phase, carries speech intelligibility and category information. Our own experiments confirmed that amplitude-only Bifrost embeddings perform at near-chance on SpeechCommands.

2. **Cross-Band Modulation Phase Coherence (CBMPC) captures semantic structure in speech** (confidence: high, pre-registered). CBMPC features extracted from a raw STFT spectrogram achieve 0.41 ± 0.04 accuracy on 10-class SpeechCommands, vs. 0.27 ± 0.01 for the STFT magnitude baseline. The effect is +13.65 percentage points (p = 0.0033, Bonferroni-corrected). This is the first validated frequency-level semantic structure extractor in the project.

3. **CBMPC does NOT generalize to environmental sounds** (confidence: high, pre-registered). On ESC-50 (50 classes), CBMPC achieves 0.12 ± 0.02, worse than the STFT baseline (0.16 ± 0.03) and the mel baseline (0.21 ± 0.04). CBMPC is speech-specific, not a universal audio feature extractor.

4. **The Bifrost complex SSM destroys modulation structure** (confidence: high). When CBMPC is applied to the Bifrost pipeline output, accuracy drops to chance (0.10). The SSM transforms the spectrogram in ways that disrupt the temporal modulation phase relationships that CBMPC relies on. This is a critical architectural finding.

5. **Pre-SSM CBMPC integration does not significantly improve over CBMPC-only** (confidence: moderate). The combined CBMPC+SSM model achieves 0.47 ± 0.04 vs. CBMPC-only 0.44 ± 0.08 (pilot, 5 classes). The SSM does not destroy CBMPC when extracted in parallel, but it does not add significant value either (p = 0.66).

6. **The original Bifrost amplitude embedding was fundamentally flawed** (confidence: high). The embedding (`amp.mean(dim=1)` + `amp.std(dim=1)`) discarded the phase channel entirely and collapsed the temporal dimension, destroying exactly the information that carries semantic structure.

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
