# 01 — Epistemic Audit Summary

**Date**: July 2026  
**Auditor**: Oluwaferanmi Oluwagbamila (Type Ω Epistemic Intelligence)  
**Full audit**: `research_dir/EPISTEMIC_AUDIT.md`

---

## What was audited

The first experimental validation loop of the Bifrost hypothesis: "phase coherence correlates with semantic category similarity on real audio."

## Flaws identified

### 1. The experiment never tested phase (critical)

The embedding extraction in `experiment_phase_coherence_baseline_comparison.py` (line 148):

```python
emb = torch.cat([amp.mean(dim=1), amp.std(dim=1)], dim=-1)
```

The phase channel (`st.phase`) was **completely discarded**. The experiment that claimed to test "phase coherence → semantic similarity" actually tested "amplitude statistics → semantic similarity."

### 2. HARKing (post-hoc protocol modification)

The initial experiment was untrained and produced r = 0.14. After observing this, the protocol was changed to add end-to-end training, a classification head, and a different dataset. This is exploratory, not confirmatory.

### 3. No baseline comparison (initially)

Early runs compared Bifrost only to itself or chance. The baseline comparison showed STFT magnitude performs comparably or better.

### 4. Single train/test split

Initial results used one random split, making estimates unstable.

### 5. Underpowered

30–50 samples per class is insufficient for a 5-class problem. No power analysis was performed.

### 6. Post-hoc embedding choice

Mean+std pooling over Bifrost amplitude was chosen after seeing results, not pre-registered.

### 7. Dimensional bottleneck

Bifrost embedding: 128 dimensions. STFT baseline: 513 dimensions. The model operated at 25% of the baseline's capacity.

### 8. Destructive temporal pooling

Mean and standard deviation over the temporal dimension collapse T timesteps into 2 statistics, destroying the temporal dynamics that carry semantic structure.

## Root cause ranking

| Rank | Cause | Likelihood | Impact |
|---|---|---|---|
| 1 | Phase channel discarded in embedding | 95% | Critical — experiment never tested the hypothesis |
| 2 | Severe dimensionality bottleneck | 85% | High — 128 vs 513 dimensions |
| 3 | Inadequate training data | 70% | Medium — underpowered |
| 4 | Pooling strategy discards temporal info | 65% | Medium — destroys modulation structure |
| 5 | No phase-specific loss function | 50% | Low-medium — no incentive to learn phase |
| 6 | Model architecture | 30% | Low — pipeline preserves phase correctly |

## What the audit changed

1. All future experiments must be pre-registered before data examination.
2. Every experiment must include baseline comparisons.
3. Cross-validation is mandatory; no single-split results.
4. Null and negative results are reported as first-class evidence.
5. The CBMPC technique was developed as a direct response to this audit.

## Lesson

The most important epistemic lesson: **the experiment that claims to test a hypothesis must actually test it.** The Bifrost pipeline correctly preserves and processes phase, but the classification head threw it away. A sophisticated architecture was reduced to a simple amplitude statistics extractor by a single line of code in the experiment script.
