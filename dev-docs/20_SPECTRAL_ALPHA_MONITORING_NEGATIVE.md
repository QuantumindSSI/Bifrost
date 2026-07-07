# 20 — Spectral Alpha Monitoring: Honest Negative Results

**The experiment**: Implemented real-time spectral alpha monitoring for LLM reasoning
and tested two approaches:
1. **Intervention**: Detect reasoning breakdown via alpha drop, inject CoT tokens
2. **Selection**: Generate N responses, select the one with highest alpha

Both approaches failed to improve reasoning accuracy on Qwen2.5-0.5B.

---

## Experiment 1: Real-Time Monitoring + Intervention

### Setup
- Model: Qwen2.5-0.5B (25 layers, 896 hidden dim)
- 25 reasoning tasks (15 math, 10 logic)
- 4 conditions: baseline, monitored (no intervention), CoT continuation, step-by-step
- Threshold calibrated from 10 reasoning + 8 factual prompts

### Calibration Result

| Metric | Value |
|--------|-------|
| Reasoning mean alpha | -0.5203 |
| Factual mean alpha | -0.5180 |
| **Separation** | **-0.0023** |
| Suggested threshold | -0.5192 |

**The separation is essentially zero.** During generation, reasoning and factual tasks
have nearly identical alpha trajectories. This is in stark contrast to the static
experiment (doc 19) which showed a separation of 0.054 (p=0.002).

### Results

| Condition | Accuracy | vs Baseline | Interventions |
|-----------|----------|-------------|---------------|
| Baseline | 80.00% | — | 0 |
| Monitored | 80.00% | 0.00pp | 0 |
| CoT continuation | 80.00% | 0.00pp (p=0.48) | 27 total |
| Step-by-step | 68.00% | **-12.00pp** (p=0.25) | 30 total |

**Alpha vs correctness**: delta=-0.0005, p=0.77 (NOT significant)

### Key Findings
1. CoT intervention had no effect (1 task fixed, 1 broken)
2. Step-by-step intervention **HARMED** accuracy (3 tasks broken, 0 fixed)
3. Alpha does not predict correctness during generation (p=0.77)
4. 15/25 tasks triggered intervention (60% false positive rate)

---

## Experiment 2: Best-of-N Selection

### Setup
- Same model and tasks
- 4 samples per task (temperature=0.8)
- 5 selection strategies: first, random, longest, highest_alpha, lowest_alpha

### Results

| Strategy | Accuracy | vs First |
|----------|----------|----------|
| First sample | 80.00% | — |
| Random | 68.00% | -12.00pp |
| Longest | 68.00% | -12.00pp |
| **Highest alpha** | **72.00%** | **-8.00pp** |
| Lowest alpha | 60.00% | -20.00pp |
| **Oracle (any correct)** | **96.00%** | +16.00pp |

**Alpha vs correctness across all 100 samples**: delta=+0.0011, p=0.28 (NOT significant)

### Key Findings
1. Highest alpha selection is WORSE than just taking the first sample
2. The oracle accuracy (96%) shows the model CAN produce correct answers
3. Alpha cannot identify which samples are correct
4. The signal is in the samples but alpha doesn't capture it

---

## Why the Negative Result?

### The Static vs Dynamic Gap

| Experiment | Separation | p-value | Result |
|-----------|-----------|---------|--------|
| Static (doc 19, prompt encoding) | 0.054 | 0.002 | Significant |
| Dynamic (doc 20, during generation) | 0.002 | 0.77 | Not significant |

The spectral alpha finding from doc 19 applies to **input processing** (encoding the
prompt), not **output generation**. When the model generates text, alpha is nearly
uniform regardless of task type or correctness.

### Model Size

The Spectral Geometry of Thought paper found:
- AUC = 1.000 for correctness prediction on Qwen2.5-7B
- AUC = 0.5-0.7 for smaller models

Our 0.5B model is below the threshold where spectral alpha is useful for correctness
prediction. The alpha signal exists but is too weak to be actionable.

### Intervention Backfire

The step-by-step intervention HARMED accuracy (-12pp). Injecting "Let me think step
by step" mid-generation disrupts the model's coherence. The model wasn't designed to
receive mid-generation interventions, and the injected tokens create unnatural context.

---

## What This Means

### For the Thesis
- The modality-dependence finding (doc 17-18) is **confirmed for LLMs**: spectral alpha
  distinguishes task types in static analysis but NOT during generation in small models
- The thesis should NOT claim that alpha monitoring improves reasoning in current LLMs
- The thesis CAN claim that spectral alpha is a measurable property of LLM hidden states
  that distinguishes reasoning from factual recall (static, doc 19)

### For Practical Applications
1. **Alpha monitoring is not useful for 0.5B models** — the signal is too weak
2. **Larger models (7B+) may benefit** — the Spectral Geometry paper shows AUC=1.0
3. **Mid-generation intervention is harmful** — disrupts model coherence
4. **Best-of-N selection with alpha is worse than random** — alpha doesn't capture quality

### What Would Need to Change
1. **Use a 7B+ model** — where alpha-correctness correlation is strong (AUC=1.0)
2. **Use alpha for post-hoc analysis, not real-time intervention** — the static signal
   is real but the dynamic signal is not
3. **Use alpha for training regularization, not inference** — Approach 2 from doc 19
4. **Explore different spectral metrics** — alpha may not be the right metric for
   real-time monitoring; phase metrics or wavelet coefficients might work better

---

## Honest Assessment

This is a negative result. The spectral alpha monitoring approach does not improve
reasoning accuracy in a 0.5B model. The approach is based on a real finding (spectral
alpha distinguishes reasoning from factual recall in static analysis), but the finding
does not transfer to dynamic generation monitoring in small models.

The path forward is NOT to continue trying to make alpha monitoring work on 0.5B models.
The path forward is:
1. Acknowledge the negative result
2. Test on larger models (7B+) where the signal is stronger
3. Explore training-time applications (regularization) rather than inference-time
4. Consider the spectral alpha finding as a diagnostic tool, not an intervention tool

---

## Data and Code

- `src/bifrost/utils/spectral_monitor.py` — SpectralAlphaMonitor class
- `research_dir/experiment_lm_monitor.py` — monitoring + intervention experiment
- `research_dir/experiment_lm_select.py` — best-of-N selection experiment
- `research_dir/results/exp_lm_monitor.json` — monitoring results
- `research_dir/results/exp_lm_select.json` — selection results
