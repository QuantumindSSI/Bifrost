# 04 — Modulation-Preserving SSM Investigation

**Status**: Research investigation  
**Goal**: Determine whether an SSM architecture can be designed that preserves the temporal modulation structure of speech while still learning phase relationships.

---

## The problem

The CBMPC experiment produced a critical architectural finding: the Bifrost complex SSM destroys the modulation structure of speech. When CBMPC features are extracted from the SSM output, accuracy drops to chance (0.10). When extracted from the raw STFT, accuracy is 0.41.

This means the SSM's transformation of the spectrogram — specifically, the complex projection from n_freq to d_model dimensions and the complex state transition — disrupts the temporal modulation phase relationships across frequency bands that carry semantic structure.

## Why the current SSM destroys modulation structure

### 1. Frequency projection

The `ComplexSpectralDecomposer` projects the input from n_freq (257 or 513 dimensions) to d_model (64 dimensions) via a learned complex linear layer:

```python
self.input_proj = ComplexLinear(self.n_freq, d_model, bias=True)
```

This projection mixes frequency bands, destroying the band-wise structure that CBMPC relies on. After projection, the d_model dimensions no longer correspond to specific frequency bands, so cross-band modulation coherence cannot be measured.

### 2. Complex state transition

The SSM state transition is:
```
h[t] = exp(-Δ[t] · A) · h[t-1] + Δ[t] · B[t] · x[t]
```

where A = A_real + i·A_imag. This transition applies a **global** temporal filter to all d_model dimensions simultaneously. It does not preserve the per-band temporal modulation structure because:
- The decay (A_real) and oscillation (A_imag) parameters are shared across what were originally different frequency bands.
- The state transition mixes information across the d_model dimensions, further blurring band-wise structure.

### 3. Output projection

The output projection (`ComplexLinear(d_model, d_model)`) further transforms the representation, making it impossible to recover the original band-wise modulation structure.

## Candidate architectures

### Architecture A: Band-wise SSM (no cross-band mixing)

**Idea**: Run a separate SSM for each frequency band, preserving the band-wise structure.

```
Input: (B, T, n_freq)
→ For each frequency band f:
    → SSM_f(x[:, :, f]) → h_f  (separate SSM per band)
→ Output: (B, T, n_freq) with learned temporal dynamics per band
→ CBMPC extraction on this output
```

**Pros**:
- Preserves band-wise structure — CBMPC can be applied after.
- Each band learns its own temporal dynamics.
- No cross-band mixing in the SSM.

**Cons**:
- n_freq separate SSMs (257 or 513) is computationally expensive.
- No cross-band interaction in the temporal model (may miss cross-band phase relationships).
- The original Bifrost design intended cross-band interaction via the binding stage, not the SSM.

**Mitigation**: Use n_mels (64) bands instead of n_freq (513), with a fixed mel filterbank projection before the SSM. This reduces to 64 band-wise SSMs.

### Architecture B: Modulation-domain SSM

**Idea**: Transform to the modulation domain first, then apply the SSM in the modulation domain.

```
Input: (B, T, n_freq)
→ Log spectrogram: L(t, f) = log(|S(t, f)| + ε)
→ Temporal FFT: L̃(ω, f) = FFT_t{L(t, f)}
→ SSM in modulation domain: process L̃(ω, f) along the ω axis
→ Output: learned modulation spectrum
→ CBMPC extraction (PLV, amplitudes) on this output
```

**Pros**:
- The SSM operates directly in the modulation domain, where semantic structure lives.
- The PLV can be computed from the SSM output.
- No destruction of modulation structure — the SSM enhances it.

**Cons**:
- The temporal FFT requires the full sequence (no streaming).
- The SSM in the modulation domain processes modulation frequencies, not time steps — this is a different kind of temporal model.
- May lose fine-grained temporal phase information.

### Architecture C: Residual SSM (modulation-preserving skip connection)

**Idea**: Keep the current SSM but add a residual skip connection that preserves the original spectrogram's modulation structure.

```
Input: (B, T, n_freq)
→ Mel projection: (B, T, n_mels)
→ SSM: (B, T, d_model) — learned temporal dynamics
→ Residual: output = SSM_output + α · mel_projection
→ CBMPC extraction on the residual output
```

**Pros**:
- Minimal change to the existing architecture.
- The residual connection preserves the original modulation structure.
- The SSM can still learn phase relationships on top of the residual.

**Cons**:
- The residual and SSM may interfere if α is not tuned properly.
- The SSM output is in d_model dimensions, while the mel projection is in n_mels — they need to be aligned.

### Architecture D: Dual-path SSM (parallel temporal and modulation paths)

**Idea**: Run two parallel SSMs — one in the time domain (current Bifrost SSM) and one in the modulation domain — and combine their outputs.

```
Input: (B, T, n_freq)
→ Path 1 (temporal): Current Bifrost SSM → (B, T, d_model)
→ Path 2 (modulation): Modulation FFT → SSM in modulation domain → (B, n_mod, d_model)
→ Combine: concat or gate the two paths
→ CBMPC extraction from Path 2 output
```

**Pros**:
- Preserves both temporal phase tracking (Path 1) and modulation structure (Path 2).
- Each path specializes in what it does best.
- CBMPC features come from the modulation path, which is designed to preserve them.

**Cons**:
- More parameters and computation.
- The combination of two paths may not be straightforward.

## Recommended investigation order

1. **Architecture C (residual SSM)** — simplest to implement, minimal change to existing code. If this works, it's the fastest path to a combined model.

2. **Architecture A (band-wise SSM)** — more principled, preserves band structure. If C fails, this is the next step.

3. **Architecture D (dual-path)** — most expressive, but most complex. If A and C both fail, this is the research direction.

4. **Architecture B (modulation-domain SSM)** — most theoretically aligned with CBMPC, but requires the biggest architectural change. Long-term research direction.

## Pre-registered protocol for the investigation

### Hypothesis

At least one of the modulation-preserving SSM architectures (C, A, or D) will produce a CBMPC-SSM combined model that:
1. Preserves CBMPC features (CBMPC on SSM output ≥ CBMPC on raw STFT)
2. Beats CBMPC-only by ≥ 3 percentage points (the SSM adds value)

### Evaluation

- Same protocol as the CBMPC experiment: SpeechCommands, 10 classes, 200 samples/class, 5-fold CV.
- Baselines: CBMPC-STFT (0.41), STFT magnitude (0.27).
- Success criterion: combined model ≥ 0.44 accuracy with p < 0.05 vs. CBMPC-STFT.

### Failure interpretation

- If no architecture preserves CBMPC features: the SSM fundamentally conflicts with modulation structure. Use CBMPC standalone and relegate the SSM to a different role.
- If CBMPC features are preserved but the SSM adds no value: the SSM's temporal phase tracking is not useful for this task. This would be a significant negative result for the Bifrost pipeline's value proposition.

## Files to create/modify

| File | Change |
|---|---|
| `src/bifrost/decomposer/modulation_preserving_ssm.py` | New module with Architectures C, A, D |
| `src/bifrost/pipeline.py` | Add option to use modulation-preserving SSM |
| `research_dir/experiment_modulation_preserving_ssm.py` | New experiment comparing architectures |

## Theoretical grounding

The modulation-preserving SSM investigation is grounded in:

- **Elhilali et al. (2009)**: temporal coherence across frequency bands is the basis of auditory perception. An SSM that destroys this coherence is working against the auditory system's organizing principle.
- **Chi et al. (1999)**: the modulation spectrum is where speech intelligibility lives. An SSM that operates in the spectral domain but disrupts the modulation domain is removing the most informative features.
- **Gu & Dao (2023, Mamba)**: selective SSMs can be designed to preserve specific structure by controlling the A and B matrices. The current Bifrost SSM does not exploit this for modulation preservation.
