# FBC Roadmap at a Glance

## The Five Phases

| Phase | Name | What | Duration | Status |
|-------|------|------|----------|--------|
| **1** | **Spectral Encoder** | Audio/Image ingestion → float32 tensors | 1 week | ✅ DONE |
| **2** | **Extended Ingest** | Add Text & Tensor decoders | 1 week | 📋 TODO |
| **3** | **S0-S1** | Spectral decomposition with **Mamba-3 backbone** | 2-3 weeks | 📋 TODO |
| **4** | **S2-S4** | Attention + Attractors + Knowledge Graph | 3-4 weeks | 📋 TODO |
| **5** | **Production** | Optimize (C++/CUDA) + Deploy | 2-3 weeks | 📋 TODO |

**Total timeline:** ~10-15 weeks (3-4 months)

---

## What Each Phase Builds

### Phase 1: Data Ingest ✅ COMPLETE
```
WAV, PNG files
    ↓
[Decoder]    Extract int16/uint8
    ↓
[Validator]  Check constraints
    ↓
[Normalizer] Convert to float32 [-1,1] or [0,1]
    ↓
Output: Canonical tensors ready for ML
```

### Phase 2: All Modalities 📋 TODO
```
Extend Phase 1 to handle:
├─ Text (CSV, JSON, Parquet)
├─ Tensors (NPZ, HDF5, Zarr)
└─ Same decode/validate/normalize pipeline
```

### Phase 3: Spectral Decomposition �� TODO
```
float32 tensor (from Phase 2)
    ↓
S0 Canonicaliser
    ├─ FFT projection
    └─ Output: SpectralTensor (amplitude + phase)
    ↓
S1 Spectral Decomposer ⭐ MAMBA-3 BACKBONE
    ├─ FFT (coarse frequencies)
    ├─ Wavelet Bank (multi-resolution)
    └─ Mamba-3 SSM (selective scan) ← KEY
    ↓
Output: Spectral embedding [batch, hidden_dim, time]
```

### Phase 4: Frequency Models 📋 TODO
```
Spectral embedding (from Phase 3)
    ↓
S2 Resonance Attention ⭐ PHASE-COHERENT ROUTING
    ├─ Q, K, V in spectral space
    ├─ Compute phase-coherence matrix
    └─ Route by PHASE not magnitude
    ↓
S3 Attractor Identifier
    ├─ Cluster spectral features
    ├─ Extract stable attractors
    └─ Output: {frequency, amplitude, phase, coherence}
    ↓
S4 Spectral Knowledge Graph + Phase-Lock Bridge ⭐
    ├─ Build persistent attractor graph
    ├─ Cross-domain transfer
    └─ Zero-shot analogy detection
    ↓
Output: FBC knowledge graph (audio/image/text integrated)
```

### Phase 5: Production 📋 TODO
```
If profiling shows Python bottleneck:
├─ C++ kernels for FFT/Wavelet
├─ CUDA kernels for Resonance Attention
├─ RingBuffer for streaming
└─ Docker/K8s packaging
```

---

## Key Innovations by Phase

| Innovation | Phase | Why |
|-----------|-------|-----|
| Canonical float32 tensors | 1 | Standardized format for all modalities |
| Mamba-3 SSM backbone | 3 | O(n) complexity, efficient for continuous signals |
| Resonance Attention | 4 | Phase-coherent routing vs dot-product |
| Spectral Knowledge Graph | 4 | Persistent memory, not stateless |
| Phase-Lock Bridge | 4 | Zero-shot cross-domain transfer ⭐ KEY |

---

## Effort & Timeline

```
Phase 1: ✅ 1 week   (DONE)
Phase 2: 📋 1 week   (straightforward, copy patterns)
Phase 3: 📋 2-3 wks  (learning Mamba-3, some research)
Phase 4: 📋 3-4 wks  (design knowledge graph, testing)
Phase 5: 📋 2-3 wks  (only if optimization needed)
         ─────────
         3-4 months total
```

---

## What You Need to Know

### Don't Build This (❌)
- Transformers for spectral data
- Dot-product attention on amplitudes
- Stateless attention-only models
- Token-based embeddings for signals

### Build This Instead (✅)
- Mamba-3 SSM for spectral decomposition
- Resonance Attention (phase-coherent routing)
- Persistent knowledge graphs
- Frequency attractors + phase-lock detection

### Why Different?
FBC works in **frequency domain** (amplitude + phase), not token space.
Models route by **phase coherence** (spectral relationships), not semantic similarity.
Knowledge is **persistent** (graph), not ephemeral (attention weights).
Transfer is **zero-shot** (phase-lock), not gradient-based.

---

## Decision Tree

### "When do I need C++?"
```
Start: Pure Python (Phase 1-3)
    ↓
Does Phase 3 run slow? (profile FFT/Mamba-3)
    ├─ NO (< 100ms per sample) → Stay Python
    └─ YES (> 100ms per sample) → Phase 5 optimization
```

### "When do I use Mamba-3?"
```
Phase 3 only (S1 Spectral Decomposition)
├─ FFT (Phase 3 early)
├─ Wavelet Bank (Phase 3 early)
└─ Mamba-3 SSM (Phase 3 main) ← Here
   Not in Phase 1, 2, 4, or 5
```

### "When do I build knowledge graphs?"
```
Phase 4 (after attractors are identified in Phase 3)
├─ S3 gives you attractors
├─ S4 builds graph from attractors
└─ Phase-Lock Bridge queries graph
```

---

## Files to Read

| File | Purpose | Read When |
|------|---------|-----------|
| **NEXT_PHASES_ROADMAP.md** | Full details (27KB) | Planning Phase 2+ |
| **FBC Engineering Script.md** | Theory + specs | Before Phase 3 |
| **Agentic CTO-Persona.pdf** | Strategic decisions | Project context |
| **AFTER_PHASE_1_SUMMARY.txt** | This overview | Right now |
| **PHASE_1_COMPLETION_REPORT.md** | What was built | Understand Phase 1 |

---

## Quick Start: Phase 2

To begin Phase 2 (extended ingest):

1. **TextDecoder**
   ```python
   class TextDecoder(BaseDecoder):
       def decode(self, data: bytes) -> Tuple[np.ndarray, dict]:
           # Parse CSV/JSON/Parquet
           # Convert text → float32 embedding
           # Return (array, metadata)
   ```

2. **TensorDecoder**
   ```python
   class TensorDecoder(BaseDecoder):
       def decode(self, data: bytes) -> Tuple[np.ndarray, dict]:
           # Load NPZ/HDF5/Zarr
           # Ensure float32
           # Return (array, metadata)
   ```

3. **Validators & Normalizers** (copy audio/image pattern)

4. **Tests** (20+ cases, parity with Phase 1)

**Effort:** 1 week, straightforward extension

---

## Success Criteria by Phase

| Phase | Criterion |
|-------|-----------|
| **1** | ✅ All tests pass, git repo, production-ready |
| **2** | 📋 All modalities ingested, 40+ tests, docs |
| **3** | 📋 Spectral embeddings generated, Mamba-3 trained |
| **4** | 📋 Attractors identified, knowledge graph built, phase-lock works |
| **5** | 📋 P99 latency < 100ms, deployment ready |

---

## Next Action

👉 **Read:** `NEXT_PHASES_ROADMAP.md` (detailed roadmap for Phases 2-5)

👉 **Then:** Confirm Phase 2 scope with team

👉 **Then:** Begin Phase 2 implementation (copy Phase 1 patterns)

---

**Questions?** See `AFTER_PHASE_1_SUMMARY.txt` for Q&A.
