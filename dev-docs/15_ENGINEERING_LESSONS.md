# 15 — Engineering Lessons: Non-Obvious Insights from Proving Steps 1-3

**Status**: Formal documentation of engineering insights discovered during implementation  
**Purpose**: Save future researchers from rediscovering these non-obvious truths about phase coherence engineering.

---

## Overview

During the implementation and execution of experiments 1A-3C, four engineering insights emerged that were not predicted by the theoretical framework. Each one caused an experiment to fail on first attempt, required debugging to understand, and revealed something fundamental about phase coherence that the original thesis did not make explicit.

These are not implementation details. They are properties of phase coherence as a mathematical object that constrain any system attempting to use it for semantic representation.

---

## Insight 1: PLV is invariant to constant phase offsets

### Statement

> PLV is invariant to constant phase offsets — |exp(i*offset)| = 1. To destroy cross-band coherence, you must add independent per-element noise, not per-band offsets.

### Mathematical proof

Phase Locking Value is defined as:

```
PLV = |(1/N) * Σ_n exp(i * (φ_a[n] - φ_b[n]))|
```

If we add a constant offset `δ` to all phases in band b:

```
PLV' = |(1/N) * Σ_n exp(i * (φ_a[n] - (φ_b[n] + δ)))|
     = |(1/N) * Σ_n exp(i * (φ_a[n] - φ_b[n] - δ))|
     = |exp(-iδ) * (1/N) * Σ_n exp(i * (φ_a[n] - φ_b[n]))|
     = |exp(-iδ)| * |(1/N) * Σ_n exp(i * (φ_a[n] - φ_b[n]))|
     = 1 * PLV
     = PLV
```

The offset `δ` factors out as a unit-magnitude complex number and is eliminated by the absolute value.

### What this means

PLV measures **relative phase consistency across elements**, not absolute phase. A constant offset applied to an entire band, scale, or channel does not change PLV. This is a feature, not a bug — it means PLV is invariant to propagation delays, which is physically correct.

### What went wrong

The first implementation of `cross_scale_destroy` added a constant random phase offset per scale:

```python
# WRONG: constant offset per scale
offset = torch.rand(1) * 2 * pi
phase = phase + offset
```

This had **zero effect** on PLV. The experiment showed no degradation, which would have falsely suggested that cross-scale coherence doesn't matter.

### The fix

Add independent random noise to each element (time point, frequency bin):

```python
# CORRECT: per-element noise
noise = torch.randn(shape) * pi
phase = phase + noise
```

This breaks the **temporal alignment** of phases across scales, which is what PLV actually measures.

### Implication for the thesis

The thesis claims "phase coherence captures semantic structure." This insight refines that claim: **it is the relative phase relationships across elements that carry semantic structure, not absolute phase values.** A system that preserves absolute phase but destroys relative phase relationships would fail to capture semantic structure.

This means any phase-preserving architecture must maintain the *pattern* of phase relationships, not just the phase values themselves. Complex-valued SSMs do this naturally — the state transition `h[t] = exp(-ΔA)h[t-1] + ΔBx[t]` preserves relative phase through the multiplicative `exp(-ΔA)` term.

---

## Insight 2: Ablation must target the correct representation level

### Statement

> Ablation must target the correct representation level — CBMPC computes PLV from modulation spectrum phase, not STFT phase. Ablating STFT phase doesn't affect CBMPC features.

### What went wrong

The first implementation of experiment 1A ablated the STFT phase:

```python
stft = torch.stft(waveform)
amplitude = stft.abs()
phase = stft.angle()
# Ablate phase here...
```

But CBMPC's pipeline is:

```
STFT → magnitude → mel projection → log compression → temporal FFT → modulation spectrum
```

The PLV is computed from the **modulation spectrum phase** (phase of the temporal FFT of log-magnitude), not from the STFT phase. Ablating STFT phase changes the complex STFT, but CBMPC only uses `stft.abs()` — the magnitude. So the ablation had **zero effect** on CBMPC features.

### The fix

Apply ablation to the modulation spectrum phase:

```python
stft = torch.stft(waveform)
mel_mag = mel_filterbank @ stft.abs()  # magnitude only
log_mag = log(mel_mag + eps)
mod_spectrum = fft(log_mag, dim=time)  # modulation spectrum
mod_phase = mod_spectrum.angle()
# Ablate mod_phase here — this is what PLV uses
```

### Implication for the thesis

Phase coherence exists at **multiple levels of representation**, and each level has its own phase structure. The STFT phase captures instantaneous frequency information. The modulation spectrum phase captures temporal modulation patterns. Cross-scale phase captures structural relationships across analysis scales.

The thesis must specify **which level of phase coherence** carries semantic structure. The experiments suggest it is the **modulation spectrum phase** (for audio) and **spatial frequency phase** (for images) — not the raw signal phase.

This has implications for architecture design: a phase-preserving pipeline must track phase through every transformation, not just the initial FFT. Each transformation (mel projection, log compression, temporal FFT) creates a new phase space that must be preserved or explicitly ablated.

---

## Insight 3: Synthetic data must control for amplitude

### Statement

> Synthetic data must control for amplitude — if classes differ in frequency content, amplitude-only features already separate them. To test phase specifically, all classes must share the same amplitude spectrum.

### What went wrong

The first synthetic audio generator gave each class a different fundamental frequency:

```python
f0 = 100 + c * 50  # different frequency per class
```

This meant the amplitude spectrum was different for each class. A magnitude-only classifier achieved 100% accuracy. Phase ablation had no effect because amplitude alone was sufficient.

### The fix

All classes share the same frequencies and amplitudes. Only the phase relationships differ:

```python
# Same for all classes
base_freqs = [200, 400, 600, 800, 1000]
base_amps = [1.0, 0.5, 0.33, 0.25, 0.2]

# Class-specific: only phase offsets
phase_offset = c * 0.3 * (harmonic_index + 1)
```

Now amplitude-only features are identical across classes. Phase is the *only* distinguishing feature. Phase ablation now properly degrades performance.

### Implication for the thesis

The thesis claims "phase coherence captures semantic structure." But this claim is only testable if you control for amplitude. In real data, amplitude and phase are correlated — signals with different semantic content often have different amplitude spectra *and* different phase structure.

The stronger, more precise claim is: **phase coherence captures semantic structure that amplitude cannot.** This is what the experiments now test. The synthetic data design ensures that if phase ablation degrades performance, it is because phase carried information that amplitude did not.

For real datasets (SpeechCommands, CIFAR-10), this control is not possible — the data has both amplitude and phase differences. The synthetic experiments provide the controlled proof; the real-data experiments provide ecological validity.

---

## Insight 4: Cross-scale destroy requires per-time-point noise

### Statement

> Cross-scale destroy requires per-time-point noise — adding a constant offset per scale doesn't break PLV because PLV measures relative phase consistency, not absolute phase.

### What went wrong

The first implementation of `cross_scale_destroy` added a constant offset per scale:

```python
# WRONG: constant offset per scale
for i in range(n_scales):
    offset = torch.rand(1) * 2 * pi
    phase[i] = phase[i] + offset
```

This is a special case of Insight 1 applied to scales instead of bands. The PLV between scale i and scale j is:

```
PLV(i,j) = |mean_n exp(i * (phase_i[n] - phase_j[n]))|
```

Adding constants `δ_i` and `δ_j` gives:

```
PLV' = |mean_n exp(i * (phase_i[n] + δ_i - phase_j[n] - δ_j))|
     = |exp(i(δ_i - δ_j))| * PLV
     = PLV
```

### The fix

Add independent noise to each time point in each scale:

```python
# CORRECT: per-element noise per scale
for i in range(n_scales):
    noise = torch.randn(phase[i].shape) * pi
    phase[i] = phase[i] + noise
```

This breaks the *temporal alignment* between scales — at each time point, the phase relationship between scale i and scale j is now random, so the mean across time points converges to zero (PLV → 0).

### Result

After the fix, `cross_scale_destroy` caused accuracy to drop from 83.9% to 9.8% — a 74.1 pp degradation (p=0.0000). This is the strongest single result supporting C2.

### Implication for the thesis

Cross-scale coherence is not about "having multiple scales" — it is about the **temporal alignment of phase patterns across scales**. If the phases at different scales are temporally aligned (phase at scale i at time t is predictive of phase at scale j at time t), the structure is preserved. If they are temporally decorrelated, the structure is destroyed regardless of how many scales are present.

This means the cross-scale coherence module must compute PLV **per time point** and aggregate, not compute a single global PLV. The temporal dimension is where the semantic structure lives.

---

## General principle

All four insights reduce to a single principle:

> **Phase coherence is a relational property. It lives in the relationships between phases at different elements (bands, scales, channels, time points), not in the phase values themselves.**

Any operation that preserves these relationships (constant offsets, permutations) does not affect coherence. Any operation that destroys these relationships (per-element noise, scrambling) does affect coherence.

This means:
1. **Ablation must be relational** — you must destroy *relationships*, not values
2. **Metrics must be relational** — PLV, coherence, congruency all measure relationships
3. **Architecture must preserve relationships** — complex-valued operations preserve phase relationships; real-valued operations on phase as a feature do not
4. **Data must control for non-relational features** — amplitude is a non-relational feature that can dominate if not controlled

The thesis, refined: **Semantic structure is encoded in the relational geometry of phase across scales, bands, and time. This relational geometry is what phase coherence measures, and it is what phase-coherent representations preserve.**
