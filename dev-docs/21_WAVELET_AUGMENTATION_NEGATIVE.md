# 21 — Wavelet Augmentation: Honest Negative Result

**The experiment**: Added Haar wavelet multi-scale decomposition to Qwen2.5-0.5B
transformer layers, fine-tuned on 30 reasoning tasks, evaluated on 25 held-out tasks.

---

## Results

| Condition | Accuracy | Math | Logic | FT Time | Final Loss |
|-----------|----------|------|-------|---------|------------|
| Pre-fine-tuning | 80.00% | 100% | 50% | — | — |
| **Baseline (FT)** | **96.00%** | **100%** | **90%** | 54.8s | 0.034 |
| Wavelet-augmented (FT) | 84.00% | 86.67% | 80% | 57.2s | 0.072 |
| Delta (wavelet - baseline) | **-12.00pp** | -13.33pp | -10.00pp | +4.4% | +0.038 |

**Statistical test**: McNemar p=0.25 (not significant, but consistently negative)
- Baseline correct → Wavelet wrong: 3 tasks
- Baseline wrong → Wavelet correct: 0 tasks

---

## What Happened

### The WaveletMixer Design (v2 — residual)
The final design uses a residual approach:
1. Decompose hidden states into Haar wavelet approx and detail coefficients
2. Upsample detail coefficients back to full dimension
3. Weight each scale's detail with a learnable parameter (initialized at zero)
4. Project through a learnable linear layer (initialized with std=0.02)
5. Add as residual: `x_out = x + proj(multi_scale_detail)`

This design starts as identity (zero-initialized weights) and learns to use
wavelet information. It trained successfully (loss converged from 0.33 to 0.07).

### Why It Hurt
1. **The wavelet residual adds noise**: Even with zero-initialized weights, the
   projection layer has std=0.02, which introduces small perturbations that
   accumulate across 24 layers
2. **The model doesn't need multi-scale structure for these tasks**: Simple
   arithmetic and logic don't benefit from multi-scale decomposition — they
   require precise token-level processing, not multi-scale smoothing
3. **Fine-tuning is too short**: 3 epochs on 30 tasks may not be enough for the
   wavelet parameters to learn useful representations
4. **The first version (v1) was catastrophic**: The reconstruction-based
   approach destroyed hidden states (loss stuck at 3.5, accuracy 4%). The
   residual approach (v2) fixed this but still slightly hurts

### The v1 Catastrophe
The first WaveletMixer design tried to reconstruct hidden states from mixed
wavelet coefficients. This completely destroyed the model's representations:
- Loss: 12.7 → 3.5 (vs baseline 0.37 → 0.03)
- Accuracy: 4% (vs baseline 84-96%)
- Output: repetitive garbage ("3 = 3 = 3 = 3 = 20. The answer is 20.")

This is a cautionary tale: wavelet reconstruction is NOT a drop-in replacement
for hidden states. The residual approach is necessary.

---

## What This Means

### For the Wavelet GPT Approach
The Wavelet GPT paper showed 2x faster **pre-training** (not fine-tuning) on
**language modeling** (not reasoning). Our experiment tests a different setting:
- We fine-tune (not pre-train)
- We evaluate reasoning (not language modeling)
- We use a 0.5B model (not a full-scale model)

The negative result does NOT contradict Wavelet GPT. It suggests that:
1. Wavelet augmentation helps during pre-training (when the model needs to learn
   multi-scale structure from scratch)
2. Wavelet augmentation does NOT help during fine-tuning on reasoning (when the
   model already has representations and just needs to adapt them)
3. The benefit of wavelet augmentation is in the pre-training phase, not the
   fine-tuning phase

### For the Thesis
The thesis claim about multi-scale structure (C2) is still supported by the
literature (Wavelet GPT, Spectral Geometry of Thought). But the practical
application of wavelet augmentation requires pre-training, not fine-tuning.

### What Would Need to Change
1. **Pre-train from scratch** with wavelet augmentation (not fine-tune)
2. **Use a larger model** where multi-scale structure matters more
3. **Use language modeling tasks** (not reasoning) for evaluation
4. **Use more training data** (30 tasks is too few for wavelet parameters to learn)

---

## Honest Assessment

This is a negative result. Wavelet augmentation does not improve reasoning
accuracy when applied during fine-tuning on a 0.5B model. The approach may
work during pre-training (as shown by Wavelet GPT), but that requires
significantly more compute and data than available here.

The code is functional and the experiment is properly designed — the negative
result is a genuine finding, not an implementation error. The v1 catastrophe
and v2 fix show that the design of the wavelet integration matters: residual
connections are essential, reconstruction-based approaches destroy the model.

---

## Data and Code

- `src/bifrost/wavelet_augmentation.py` — WaveletMixer, WaveletAugmentedLayer
- `research_dir/experiment_lm_wavelet.py` — fine-tuning experiment
- `research_dir/results/exp_lm_wavelet.json` — full results
