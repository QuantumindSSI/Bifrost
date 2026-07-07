# 03 — CBMPC Pre-SSM Integration Plan

**Status**: Implementation plan  
**Goal**: Integrate CBMPC as a feature extraction layer that operates on the canonical spectrogram **before** the complex SSM, not after.

---

## Problem statement

The CBMPC experiment showed that:
- CBMPC-STFT (on raw STFT) achieves 0.41 accuracy — **SUPPORTED**.
- CBMPC-Bifrost (on pipeline output) achieves 0.10 accuracy — **at chance**.

The Bifrost complex SSM destroys the modulation structure that CBMPC relies on. The SSM projects the spectrogram from n_freq dimensions to d_model dimensions and applies a complex state transition, which disrupts the temporal modulation phase relationships across frequency bands.

## Solution: pre-SSM integration

Instead of extracting CBMPC features from the pipeline output, extract them from the canonical spectrogram (the output of `SpectralCanonicalizer`) **before** the SSM processes it. The SSM then operates on the CBMPC-enhanced representation.

### Current pipeline flow

```
Audio → SpectralCanonicalizer → SpectralTensor (B, T, n_freq)
    → ComplexSpectralDecomposer (SSM) → SpectralTensor (B, T, d_model)
    → SpectralBinding → bound SpectralTensor
    → [attractor learning, semantic coherence]
    → Output
```

### Proposed pipeline flow

```
Audio → SpectralCanonicalizer → SpectralTensor (B, T, n_freq)
    → CBMPCExtractor → CBMPC features (B, feature_dim)
    → [parallel branch] ComplexSpectralDecomposer (SSM) → SpectralTensor (B, T, d_model)
    → SpectralBinding → bound SpectralTensor
    → CBMPC feature injection (concatenation or gating)
    → Output
```

The key insight: CBMPC operates on the canonical spectrogram (which has the natural modulation structure of speech), while the SSM operates in parallel for temporal phase tracking. The two representations are combined at the output.

## Implementation steps

### Step 1: Add CBMPC as a pipeline component

Add an optional `use_cbmpc` parameter to `BifrostPipeline.__init__`:

```python
def __init__(
    self,
    ...
    use_cbmpc: bool = False,
    cbmpc_n_mels: int = 64,
    cbmpc_modulation_freqs: list = None,
    ...
):
    ...
    self.use_cbmpc = use_cbmpc
    if use_cbmpc:
        from .cbmpc import CBMPCExtractor
        self.cbmpc_extractor = CBMPCExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft_canonical,
            hop_length=n_fft_canonical // 2,
            n_mels=cbmpc_n_mels,
            modulation_freqs=cbmpc_modulation_freqs,
            duration_seconds=1.0,
            feature_mode="rich",
        )
```

### Step 2: Extract CBMPC features in the forward pass

In `BifrostPipeline.forward`, after canonicalization and before/at the end:

```python
def forward(self, signal, metadata=None, h_0=None):
    self._validate_input(signal)
    canonical = self.canonicalizer(signal, metadata)

    # Extract CBMPC features from the canonical spectrogram
    cbmpc_features = None
    if self.use_cbmpc:
        cbmpc_features = self.cbmpc_extractor(signal)  # operates on raw audio

    # Continue with SSM pipeline (unchanged)
    if self.use_complex_ssm:
        decomposed, _ = self.decomposer(canonical, h_0)
    else:
        decomposed = self.decomposer(canonical)

    bound_st, coherence = self._run_binding(decomposed, canonical)
    bound_st = self._run_attractor_learning(bound_st)
    bound_st = self._run_semantic_coherence(bound_st)

    # Attach CBMPC features to the output metadata
    if cbmpc_features is not None:
        bound_st.metadata['cbmpc_features'] = cbmpc_features

    return bound_st, coherence
```

### Step 3: Create a CBMPC-enhanced classifier

```python
class BifrostCBMPCClassifier(nn.Module):
    def __init__(self, pipeline, cbmpc_feature_dim, d_model, n_classes, dropout=0.0):
        super().__init__()
        self.pipeline = pipeline
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Combine SSM embedding (d_model * 2) with CBMPC features
        self.classifier = nn.Linear(d_model * 2 + cbmpc_feature_dim, n_classes)

    def forward(self, x, *_):
        st, _ = self.pipeline(x, metadata={"sample_rate": 16000.0})
        amp = st.amplitude
        if amp.dim() == 2:
            ssm_emb = amp
        else:
            ssm_emb = torch.cat([amp.mean(dim=1), amp.std(dim=1)], dim=-1)
        cbmpc_emb = st.metadata.get('cbmpc_features', torch.zeros(x.shape[0], 0))
        emb = torch.cat([ssm_emb, cbmpc_emb], dim=-1)
        emb = self.dropout(emb)
        return self.classifier(emb)
```

### Step 4: Test that CBMPC features survive the pipeline

A critical test: verify that `cbmpc_features` in the metadata has the correct shape and is not corrupted by the pipeline forward pass.

### Step 5: Run the pre-registered protocol

Compare:
1. CBMPC-only (no SSM) — the validated baseline (0.41)
2. SSM-only (no CBMPC) — the original Bifrost (0.16)
3. CBMPC + SSM combined — does the combination beat CBMPC-only?

**Success criterion**: CBMPC + SSM must beat CBMPC-only by ≥ 3 percentage points (a lower bar than beating STFT, since CBMPC-only is already strong). If it does not, the SSM adds no value for this task, and CBMPC should be used standalone.

## Risks

1. **The SSM may still interfere** even with parallel CBMPC extraction. If the classifier concatenates SSM and CBMPC features, the SSM noise may degrade the CBMPC signal.
2. **Dimensional imbalance**: CBMPC (462 dims) + SSM (128 dims) = 590 dims. The CBMPC features may dominate, making the SSM contribution negligible.
3. **The canonical spectrogram may not have enough temporal resolution** for CBMPC if the STFT hop length is too large.

## Files to modify

| File | Change |
|---|---|
| `src/bifrost/pipeline.py` | Add `use_cbmpc` parameter, CBMPC extraction in forward, attach features to metadata |
| `src/bifrost/cbmpc.py` | No changes needed (already implemented) |
| `research_dir/experiment_cbmpc_pre_ssm.py` | New experiment script for the combined model |

## Timeline

- Step 1–3: Implementation (1 session)
- Step 4: Sanity check (1 run)
- Step 5: Pre-registered protocol (5-fold CV, 10 classes, 200 samples/class)
