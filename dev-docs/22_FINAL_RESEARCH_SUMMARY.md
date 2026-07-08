# 22 — Final Research Summary and Literature Roadmap

**This document consolidates all findings from the Bifrost project (docs 1-21) and
maps them to the current state of frequency-level approaches for improving LLM
semantic structure and creative capabilities.**

---

## Part I: Bifrost Project Consolidation

### The Original Thesis (OVERCLAIMED)

> "Intelligence is structured resonance. Semantic structure is encoded in the phase
> coherence of oscillatory components across multiple scales, and this principle
> generalizes across all modalities and all levels of intelligence."

**Status: NOT SUPPORTED.** The thesis made three claims:
- C1: Phase coherence captures semantic structure
- C2: Multi-scale structure is necessary
- C3: This generalizes across all modalities

### What the Evidence Shows

#### Real-Data Experiments (Docs 17-18)

| Modality | What captures structure | Phase matters? | Evidence |
|----------|------------------------|----------------|----------|
| **Images** (digits) | Phase congruency | **Yes** | +8.2pp over FFT, p=0.0001, 4/4 ablations significant |
| **Sensors** (UCI HAR) | Wavelet coherence | **Yes** | +13.8pp combined, p=0.002 |
| **Speech** (SpeechCommands) | Spectral envelope (amplitude) | **No** | CBMPC worse than FFT, p=0.0001 |
| **Env. Audio** (ESC-50) | Mostly amplitude | **Marginal** | Trend only (p=0.051) |
| **LLM hidden states** (Qwen2.5-0.5B) | Spectral alpha (amplitude) | **No** | p=0.002 reasoning vs factual (static) |

**The pattern**: Phase coherence captures structure when spatial/temporal structure
IS the signal (edges in images, periodic patterns in sensors). In distributed
representations (speech envelope, LLM hidden states), the amplitude spectrum's
multi-scale structure carries the information instead.

#### LM Experiments (Docs 19-21)

| Approach | Result | Why |
|----------|--------|-----|
| Static spectral alpha analysis | **Positive** (p=0.002) | Alpha distinguishes reasoning from factual in prompt encoding |
| Real-time alpha monitoring | Negative | Alpha signal disappears during generation in 0.5B model |
| Best-of-N alpha selection | Negative | Alpha doesn't predict response quality in 0.5B model |
| Wavelet augmentation (fine-tuning) | Negative | Wavelet helps pre-training, not fine-tuning |

#### Code Quality (Path B)

- **60 tests** added across 8 test files — all passing
- **5 duplicated patterns** extracted to shared `spectral_utils.py`
- **3 pre-existing bugs** fixed
- Test coverage for MSC modules: 0% → 60-100%

### The Revised Thesis (EVIDENCE-BACKED)

> **Multi-scale coherence features capture structural information that amplitude-only
> features miss. This effect is modality-dependent: strong for images with edge structure,
> significant for sensor data with temporal structure, and for LLMs, the relevant
> coherence is in the amplitude spectrum (spectral alpha) rather than phase.**

### Key Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Real-data experiments | `research_dir/experiment_*_real.py` | SpeechCommands, ESC-50, UCI HAR, digits |
| LM reasoning experiment | `research_dir/experiment_lm_reasoning_phase.py` | Static alpha analysis |
| Alpha monitor | `src/bifrost/utils/spectral_monitor.py` | Real-time monitoring class |
| Wavelet augmentation | `src/bifrost/wavelet_augmentation.py` | Haar wavelet transformer wrapper |
| Shared utilities | `src/bifrost/utils/spectral_utils.py` | wrap_phase, compute_plv, etc. |
| Tests | `tests/test_*.py` | 60 tests across 8 files |
| Results | `research_dir/results/` | All JSON results |

---

## Part II: Literature — Frequency-Level Approaches for LM Improvement

### A. Architecture-Level Approaches (Pre-Training)

#### 1. Wavelet GPT (Guthikonda et al., 2024)
- **arXiv**: 2409.12924
- **What**: Inserts Haar/learnable wavelet transforms into GPT decoder layers during
  pre-training, giving every next-token prediction access to multi-scale intermediate
  embeddings at different temporal resolutions
- **Results**: 2x faster pre-training with same performance on text, audio, and images.
  No extra parameters. Significant gains when trained for same number of steps.
- **Relevance to Bifrost**: This is the approach we tested in doc 21 (negative result
  on fine-tuning). The literature confirms it works for PRE-TRAINING, not fine-tuning.
  Our negative result is consistent — wavelet augmentation needs to be applied during
  pre-training from scratch.
- **Key insight**: Multi-scale structure is an inductive bias that helps the model
  LEARN representations, not adapt existing ones.

#### 2. FANformer (Dong et al., 2025)
- **arXiv**: 2502.21309
- **Code**: https://github.com/YihongDong/FANformer
- **Model**: https://huggingface.co/dongyh/FANformer-1B
- **What**: Adapts Fourier Analysis Network (FAN) into the attention mechanism to
  achieve efficient periodicity modeling. Modifies the feature projection process
  of attention to introduce Fourier principles for capturing periodic patterns.
- **Results**: FANformer-1B (1.1B params, 1 trillion tokens) consistently outperforms
  standard Transformer when scaling up. Superior ability to learn and apply rules
  for reasoning compared to Transformer.
- **Relevance to Bifrost**: This directly addresses the reasoning improvement goal.
  The key insight is that PERIODICITY (a frequency-domain property) is fundamental
  to reasoning — "Reasoning is Periodicity?" The FAN layer captures periodic patterns
  that standard attention misses.
- **Key insight**: Reasoning relies on periodic patterns (rules, cycles, patterns).
  Explicitly modeling periodicity via Fourier analysis improves reasoning ability.

#### 3. PRISM (Yildirim & Yucedag, 2026)
- **arXiv**: 2512.01208
- **Code**: https://github.com/AlperYildirim1/Language-as-Waves
- **What**: Complex-valued encoder that enforces unit-norm constraint (|z|=1) and
  replaces attention with gated harmonic convolutions. Encodes semantic identity
  as resonant frequencies in the complex domain.
- **Results**: Synonym pairs exhibit significantly higher phase coherence than random
  pairs (R=0.198 vs 0.072, p<0.001). 96% 5-shot acquisition of novel concepts with
  negligible degradation (-0.84 BLEU) vs Transformer's catastrophic forgetting
  (-10.55 BLEU). Hybrid Wave-Particle Transformer matches baselines at 33M params.
- **Relevance to Bifrost**: This is the STRONGEST evidence for phase encoding
  semantic information in LMs. PRISM demonstrates that phase angles CAN encode
  semantic information when magnitude is constrained. This partially supports C1
  for LMs — but only in a constrained complex-valued architecture, not standard
  real-valued transformers.
- **Key insight**: Phase encodes semantics when the model is FORCED to use it
  (unit-norm constraint). In standard transformers, magnitude dominates and phase
  is a byproduct. The plasticity-stability advantage is a major finding.

#### 4. Spectral Dictionary Learning (2025)
- **arXiv**: 2505.00033
- **What**: Replaces self-attention with a global time-varying Fourier dictionary
  and per-token mixing coefficients. Enforces reconstruction losses in both time
  and frequency domains (STFT magnitude matching).
- **Results**: Competitive perplexity on WikiText2 and Penn Treebank with linear
  complexity (vs quadratic attention). Significant inference latency reduction.
- **Relevance**: Shows that frequency-domain token mixing can replace attention
  entirely while maintaining performance. The STFT magnitude matching loss is
  directly related to our amplitude spectrum findings.

#### 5. Multi-Domain Fourier-Wavelet Attention (MDFWA)
- **What**: Replaces attention with Fourier mixing (global dependencies) + wavelet
  filters (local dependencies). Full encoder-decoder with causal masking.
- **Results**: Sub-quadratic time and memory. Improved expressive power on
  abstractive summarization.
- **Relevance**: Combines the global structure of Fourier transforms with the local
  structure of wavelets — exactly the multi-scale approach Bifrost advocates.

### B. Weight-Level Approaches (Fine-Tuning)

#### 6. Spectral Modulation (Kim et al., 2024)
- **ACL**: 2024.findings-emnlp.224
- **What**: Treats weight matrices of linear layers as 2D signals, applies 2D FFT.
  Hypothesizes low-frequency = signal, high-frequency = noise. Modulates
  high-frequency components to denoise the model.
- **Results**: Improves LLM performance WITHOUT fine-tuning — pure spectral
  filtering of existing weights.
- **Relevance to Bifrost**: This is a zero-shot frequency-level improvement. It
  shows that the weight matrices of trained LLMs have spectral structure that can
  be exploited. Low-frequency = essential knowledge, high-frequency = noise.
- **Key insight**: Training introduces noise into weight space. Spectral filtering
  can remove this noise without retraining.

#### 7. SpectralLoRA (2026)
- **arXiv**: 2604.10649
- **What**: Analyzes the spectral structure of LoRA weight updates via 2D DCT.
  Finds that LoRA updates are universally dominated by low-frequency components
  (33% of DCT coefficients capture 90% of spectral energy).
- **Results**: Retaining only 10% of frequency coefficients reduces adapter storage
  10x while sacrificing only 1.95pp. Frequency masking at 50% IMPROVES over full
  LoRA on 3/8 model-task pairs (high-frequency = adaptation noise).
- **Relevance**: Shows that task adaptation is spectrally sparse — it lives in
  low-frequency space. This means frequency-level interventions during fine-tuning
  can be more efficient than full-parameter updates.

#### 8. Fourier-Activated Adapter (FAA, 2025)
- **arXiv**: 2512.22378
- **What**: PEFT method that incorporates random Fourier features into adapter
  modules. Decomposes intermediate representations into low and high-frequency
  components, enabling frequency-aware modulation.
- **Results**: Competitive or superior to existing PEFT methods on GLUE, E2E NLG,
  and instruction-tuning benchmarks.
- **Relevance**: A practical frequency-level adapter that could be tested on
  reasoning tasks. The frequency-aware activation is exactly the kind of
  multi-scale intervention Bifrost's thesis predicts should help.

#### 9. SMoA: Spectrum Modulation Adapter (2026)
- **arXiv**: 2605.21147
- **What**: Partitions weight matrices into spectral blocks and applies
  Hadamard-modulated low-rank branches to each block. Broader coverage of
  pretrained spectral directions than standard LoRA.
- **Results**: Improves over LoRA at lower parameter budget.
- **Relevance**: Shows that spectral structure of weights can be exploited for
  more efficient fine-tuning.

#### 10. sDCTFT: Selective DCT Fine-Tuning (2024)
- **arXiv**: 2410.09103
- **What**: Projects weight changes into discrete cosine space, selects most
  critical frequency components in each partition.
- **Results**: Outperforms LoRA with 0.05M vs 38.2M trainable parameters on
  LLaMA3.1-8B instruction tuning.
- **Relevance**: Demonstrates that frequency-domain fine-tuning is dramatically
  more parameter-efficient than spatial-domain approaches.

### C. Generation-Level Approaches (Inference)

#### 11. FourierSampler (2025)
- **What**: Frequency-domain sliding window on hidden states to guide diffusion
  LLMs to first decode structural content (low-frequency), then detail
  (high-frequency).
- **Results**: Consistent improvements on code and math tasks, surpassing
  autoregressive models of same size.
- **Relevance**: Directly relevant to reasoning — the frequency-ordered decoding
  (structure first, detail second) mirrors how humans reason (outline then fill in).

#### 12. WavePhaseNet (2026)
- **arXiv**: 2602.14419
- **What**: Applies DFT along sequence dimension to decompose semantic information
  into frequency bands. Low-frequency = global meaning/intent, high-frequency =
  local syntax/expression. Constructs Semantic Conceptual Hierarchy Structure (SCHS).
- **Results**: Theoretically derives that GPT-4's 24,576-dim embedding space has
  1/f spectral structure and can be reduced to ~3,000 dimensions while preserving
  meaning. Enables rigorous reasoning while suppressing hallucination.
- **Relevance to Bifrost**: This is the most theoretically aligned paper to the
  Bifrost thesis. It explicitly constructs a semantic hierarchy using frequency
  decomposition and shows it reduces hallucination. The 1/f spectral structure
  is a multi-scale property — exactly what C2 predicts.
- **Key insight**: Hallucination is a structural limitation caused by the embedding
  space not being isomorphic to the semantic truth set. Frequency decomposition
  can construct the missing hierarchy.

### D. Position Encoding Approaches

#### 13. Wavelet-like Properties in Transformers (ACL 2025)
- **What**: Shows that RoPE (Rotary Positional Embeddings) naturally develops
  wavelet-like properties during training — multi-scale structure emerges
  spontaneously in position encodings.
- **Relevance**: Confirms that multi-scale structure (C2) is not just an external
  addition but emerges naturally in trained transformers.

---

## Part III: Synthesis — What Actually Works

### What the evidence supports

| Approach | Phase | Evidence | Bifrost alignment |
|----------|-------|----------|-------------------|
| **FANformer** (periodicity in attention) | Pre-training | 1B model, superior reasoning | C2 (multi-scale) |
| **PRISM** (phase-constrained encoding) | Pre-training | Phase encodes semantics when constrained | C1 (phase) — but only in complex-valued nets |
| **Wavelet GPT** (multi-scale embeddings) | Pre-training | 2x faster training | C2 (multi-scale) |
| **Spectral Modulation** (weight denoising) | Post-training | Zero-shot improvement | Frequency structure in weights |
| **SpectralLoRA** (frequency-sparse adaptation) | Fine-tuning | 10x compression, sometimes improves | Frequency structure in adaptation |
| **FAA** (Fourier adapter) | Fine-tuning | Competitive with PEFT | C2 (multi-scale) |
| **WavePhaseNet** (semantic hierarchy via DFT) | Architecture | Reduces hallucination, 1/f structure | C2 + semantic structure |
| **FourierSampler** (frequency-ordered decoding) | Inference | Improves code and math | Structure-first generation |

### What does NOT work (from Bifrost experiments)

| Approach | Phase | Result | Why |
|----------|-------|--------|-----|
| Phase coherence (CBMPC) on speech | Feature extraction | Negative | Amplitude dominates in speech |
| Alpha monitoring during generation | Inference | Negative | Signal too weak in 0.5B model |
| Wavelet augmentation during fine-tuning | Fine-tuning | Negative | Wavelets help pre-training, not fine-tuning |
| Cross-modal coherence alignment | Architecture | Negative | Modality gap is fundamental |

### The key insight

**Frequency-level approaches work when they are built into the architecture during
pre-training, not when they are applied post-hoc during fine-tuning or inference.**

The successful approaches (FANformer, Wavelet GPT, PRISM) all modify the architecture
and train from scratch. The unsuccessful approaches (our wavelet fine-tuning, alpha
monitoring) try to add frequency structure to an already-trained model.

The exception is **Spectral Modulation** (weight denoising) and **SpectralLoRA**
(frequency-sparse adaptation), which work post-hoc but operate on WEIGHT matrices,
not hidden states.

---

## Part IV: Roadmap for Improving LM Semantic Structure and Creativity

### Tier 1: Most Promising (Architecture + Pre-Training)

#### A. FANformer-Style Periodicity Modeling
- **What**: Replace standard Q/K/V projections with FAN layers that introduce
  Fourier basis functions for periodicity modeling
- **Why**: FANformer-1B shows superior reasoning. Periodicity is fundamental to
  rules, patterns, and logical structure.
- **Effort**: High (requires pre-training from scratch)
- **Expected impact**: Significant reasoning improvement
- **Bifrost connection**: Directly tests C2 (multi-scale structure) at the
  architecture level. The Fourier basis is the multi-scale decomposition.

#### B. PRISM-Style Phase-Constrained Encoding
- **What**: Use complex-valued hidden states with unit-norm constraint. Encode
  semantic identity as resonant frequencies. Replace attention with gated
  harmonic convolutions.
- **Why**: PRISM shows phase encodes semantics when magnitude is constrained.
  96% plasticity with negligible forgetting.
- **Effort**: Very high (fundamentally different architecture)
- **Expected impact**: Major — solves plasticity-stability dilemma, enables
  real-time knowledge adaptation
- **Bifrost connection**: Directly tests C1 (phase coherence) in a setting where
  it can actually work (constrained complex-valued network)

#### C. WavePhaseNet-Style Semantic Hierarchy
- **What**: Apply DFT along sequence dimension to decompose semantics into
  frequency bands. Low-frequency = global meaning, high-frequency = local syntax.
  Construct explicit Semantic Conceptual Hierarchy Structure.
- **Why**: Reduces hallucination by constructing the missing semantic hierarchy.
  1/f spectral structure confirms multi-scale nature of language.
- **Effort**: Medium (can be applied to existing models)
- **Expected impact**: Hallucination reduction, more consistent reasoning
- **Bifrost connection**: Directly constructs the semantic hierarchy that C2
  predicts should exist

### Tier 2: Practical (Fine-Tuning + Adaptation)

#### D. Spectral Modulation (Weight Denoising)
- **What**: Apply 2D FFT to weight matrices, suppress high-frequency components,
  inverse FFT back. Zero-shot, no training required.
- **Why**: Removes training noise from weights. Improves performance without
  any fine-tuning.
- **Effort**: Low (post-hoc, no training)
- **Expected impact**: Modest but free improvement
- **Bifrost connection**: Exploits frequency structure in weight space

#### E. Fourier-Activated Adapter (FAA)
- **What**: PEFT adapter that decomposes representations into low/high frequency
  components with learnable frequency-aware modulation.
- **Why**: Frequency-aware adaptation is more efficient than spatial adaptation.
- **Effort**: Low (PEFT, fine-tune existing model)
- **Expected impact**: Better fine-tuning efficiency, potentially better reasoning
- **Bifrost connection**: Multi-scale adaptation

#### F. SpectralLoRA (Frequency-Sparse Adaptation)
- **What**: Apply LoRA in DCT space, keep only low-frequency components.
- **Why**: Task adaptation is spectrally sparse (33% of coefficients = 90% energy).
  High-frequency components are adaptation noise.
- **Effort**: Low (PEFT variant)
- **Expected impact**: 10x parameter reduction, sometimes improves over full LoRA
- **Bifrost connection**: Confirms frequency structure in adaptation

### Tier 3: Inference-Time (No Training)

#### G. FourierSampler (Frequency-Ordered Decoding)
- **What**: Guide diffusion LLMs to decode low-frequency (structural) content
  first, then high-frequency (detail) content.
- **Why**: Structure-first generation mirrors human reasoning (outline then fill).
- **Effort**: Low (decoding strategy)
- **Expected impact**: Improves code and math tasks
- **Bifrost connection**: Multi-scale generation order

#### H. Spectral Alpha Monitoring (on 7B+ models)
- **What**: Monitor spectral alpha of hidden states to detect reasoning breakdown.
  Our implementation works but needs a larger model.
- **Why**: Spectral Geometry of Thought shows AUC=1.0 on 7B models.
- **Effort**: Low (code already written, just needs bigger model)
- **Expected impact**: Reasoning quality detection
- **Bifrost connection**: Direct application of our doc 19 finding

### Tier 4: Creative Capabilities

#### I. Frequency-Band Conditioning for Creative Generation
- **What**: Condition generation on specific frequency bands. Low-frequency =
  theme/structure, high-frequency = style/detail. Allow users to control
  creativity by adjusting frequency band weights.
- **Why**: WavePhaseNet shows frequency bands correspond to semantic hierarchy.
  FourierSampler shows frequency-ordered generation works.
- **Effort**: Medium
- **Expected impact**: Controllable creativity — users can dial up structural
  novelty (low-frequency variation) or stylistic variation (high-frequency)
- **Bifrost connection**: Multi-scale creative control

#### J. Phase Interference for Novel Combinations
- **What**: Use PRISM-style phase interference to create novel semantic
  combinations. Phase rotations can create new meaning from existing concepts.
- **Why**: PRISM shows phase rotations resolve lexical ambiguity. The same
  mechanism could create novel semantic combinations for creativity.
- **Effort**: High
- **Expected impact**: Novel idea generation through phase manipulation
- **Bifrost connection**: C1 (phase) in a constrained setting where it works

---

## Part V: Honest Assessment and Recommendations

### What Bifrost Got Right

1. **Multi-scale structure matters** (C2) — confirmed by Wavelet GPT, FANformer,
   WavePhaseNet, and our own spectral alpha finding
2. **The modality-dependence of phase** — confirmed: phase matters for images/sensors,
   amplitude matters for speech/LLMs. PRISM shows phase CAN matter for LMs but only
   in constrained architectures
3. **Spectral alpha distinguishes reasoning from factual recall** — confirmed and
   replicated (Spectral Geometry of Thought)
4. **The synthetic experiments were circular** — honestly acknowledged
5. **The original thesis was overclaimed** — honestly revised

### What Bifrost Got Wrong

1. **Phase coherence is NOT the universal mechanism** — it's modality-dependent
2. **Cross-modal generalization (C3) is NOT supported** — modality gap is fundamental
3. **Real-time alpha monitoring doesn't work on small models** — needs 7B+
4. **Wavelet augmentation doesn't help during fine-tuning** — needs pre-training
5. **The "intelligence is structured resonance" claim was too strong** — multi-scale
   spectral structure is real, but it's not THE mechanism of intelligence

### Recommended Next Steps

**For improving LLM reasoning:**
1. **Use FANformer architecture** for pre-training (best evidence for reasoning)
2. **Apply Spectral Modulation** to existing models (zero-shot improvement)
3. **Test FAA adapters** for fine-tuning on reasoning tasks
4. **Test alpha monitoring on 7B+ models** (our code is ready)

**For improving LLM creativity:**
1. **Implement frequency-band conditioning** (WavePhaseNet-inspired)
2. **Explore PRISM-style phase interference** for novel combinations
3. **Use FourierSampler-style frequency-ordered decoding** for structured generation

**For the Bifrost thesis:**
1. **C2 (multi-scale) is supported** — focus on this
2. **C1 (phase) is conditionally supported** — only in constrained architectures
3. **C3 (cross-modal) is not supported** — acknowledge this
4. **The revised thesis (doc 18) is the honest claim**

---

## References

1. Guthikonda et al. (2024). "Wavelet GPT - Wavelet Inspired Large Language Models." arXiv:2409.12924
2. Dong et al. (2025). "Reasoning is Periodicity? FANformer." arXiv:2502.21309
3. Yildirim & Yucedag (2026). "Language as a Wave Phenomenon: PRISM." arXiv:2512.01208
4. Kim et al. (2024). "Unleashing the Potential of LLMs through Spectral Modulation." EMNLP Findings
5. SpectralLoRA (2026). arXiv:2604.10649
6. FAA (2025). "Fourier-Activated Adapter." arXiv:2512.22378
7. WavePhaseNet (2026). arXiv:2602.14419
8. FourierSampler (2025). OpenReview
9. Spectral Dictionary Learning (2025). arXiv:2505.00033
10. MDFWA (2025). "Beyond Self-Attention: Fourier-Wavelet Transformer."
11. SMoA (2026). arXiv:2605.21147
12. sDCTFT (2024). arXiv:2410.09103
13. Spectral Adapter (2024). arXiv:2405.13952
14. Wavelet-like Properties in Transformers (ACL 2025)
15. Spectral Geometry of Thought — referenced in doc 16
