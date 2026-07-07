# 16 — Integrating Structured Resonance into Language Models

**Research survey**: How the Bifrost experiments (C1-C3) connect to language model architecture, results from the literature, and engineering tradeoffs.

---

## 1. The Core Question

The Structured Resonance Thesis claims that intelligence is structured resonance — phase-coherent multi-scale representations capture semantic structure across modalities. Our experiments (1A-3D) provide evidence for C1 (phase captures structure), C2 (multi-scale is necessary), and partial evidence for C3 (cross-modal generalization).

**The question**: Can this principle be integrated into language models, and what does the literature say about the results and tradeoffs?

The answer is a qualified **yes**. There is a rapidly growing body of work (2022-2026) that directly tests phase-based, spectral, and wavelet methods in language models. The results are promising but reveal specific engineering tradeoffs that must be navigated.

---

## 2. Relevant Literature: Five Threads

### Thread 1: Phase Encodes Semantics in Complex-Valued LMs

**PRISM** (Yıldırım & Yücedağ, 2025, arXiv:2512.01208) is the most directly relevant paper. It introduces a complex-valued encoder with a unit-norm constraint (|z|=1), forcing all information into phase angles. Key findings:

- **Synonym pairs exhibit significantly higher phase coherence than random pairs** (R=0.198 vs 0.072, p<0.001). This directly confirms our C1 claim — phase coherence reflects semantic relationships.
- **Lexical ambiguity is resolved via phase rotation**, not magnitude amplification. The model rotates phase across layers to disambiguate word senses while maintaining near-unit gain.
- **Phase representations are robust to scalar attenuation** — retaining 97% of translation quality when signal magnitude is uniformly reduced. This confirms that phase, not magnitude, carries the semantic load.
- **Spectral density threshold**: the model fails on isolated tokens, requiring minimum sequence length to produce interference patterns. Phase-based computation is inherently multi-token.
- **Hybrid architecture (Wave-Particle Transformer)**: combining phase-based encoder with standard attention matches Transformer baselines at 33M parameters with fewer non-embedding parameters.

**Connection to Bifrost**: PRISM's finding that synonyms have higher phase coherence than random pairs is the language-model analog of our Experiment 1A/1B finding that phase ablation destroys classification. Both show phase carries semantic structure. PRISM's spectral density threshold (minimum sequence length) is the language analog of our Experiment 2A finding that multi-scale coherence is necessary — phase computation requires sufficient "spectral density" to produce interference.

**Phase-Coherent Transformer (PCT)** (arXiv:2605.10123) extends this by replacing softmax attention with a phase-preserving gate on L2-normalized complex query-key similarities. Key results:

- PCT **consistently outperforms both standard softmax Transformer and direct complex-valued counterpart** under parameter-fair comparison.
- PCT shows **no depth-related accuracy collapse** across tested depth range (standard transformers degrade with depth).
- Gates that **delete negatively-aligned phase components** collapse on long-range retrieval — confirming that phase alignment (not just magnitude) is critical.
- PCT remains competitive with Multiscreen (strongest real-valued NN baseline) even on tasks traditionally difficult for complex-valued networks.

**Token2Wave** (Zhang & Sheng, 2024, arXiv:2411.06989) represents each token as a complex vector G·e^(iα) where magnitude G captures global semantics and phase α encodes token-to-global relationships. Results:

- **Significantly reduces video memory usage and training time** compared to BERT.
- Wave-like operations (interference, modulation) during forward propagation are effective for text classification.
- Gradient analysis shows Token2Wave has unique training dynamics compared to standard embeddings.

**Complex Hilbert Space LLMs** (Zenodo:17890922) proposes training LLMs in complex Hilbert space with imaginary-time gradient flow. Token embeddings, hidden states, and attention features all live in C^d. Claims:

- Improved gradient stability through norm-preserving flows.
- Increased expressivity via phase-coded structure and interference.
- "Resonant Phase-Locking hypothesis": large-scale training induces phase-locked internal modes supporting long-range credit assignment.

**CAWN** (arXiv:2604.04250) introduces Continuous Acoustic Wave Networks for autoregressive language modeling. Projects hidden states into multi-headed complex-domain phasors with O(L) causal phase accumulation. Grammar and semantics are modeled as constructive/destructive wave interference in the complex domain.

### Thread 2: Spectral Analysis Reveals Phase Transitions in LLMs

**The Spectral Geometry of Thought** (arXiv:2604.15350) is a landmark finding. Through spectral analysis of 11 LLMs across 5 architecture families (Qwen, Pythia, Phi, Llama, DeepSeek-R1), they identify seven phenomena:

1. **Reasoning Spectral Compression**: 9/11 models show significantly lower spectral α for reasoning vs factual recall (p<0.05). Effect size correlates with model capability.
2. **Instruction Tuning Spectral Reversal**: base models compress for reasoning, instruction-tuned models reverse this — instruction tuning fundamentally reorganizes spectral structure.
3. **Spectral Scaling Law**: α_reasoning ∝ -0.074 ln(N) across 4 Qwen base models (R²=0.46). Larger models compress more for reasoning.
4. **Token-Level Spectral Cascade**: adjacent layers have highly synchronized spectral dynamics (ρ=0.84 at distance 9), decaying exponentially with layer distance (ρ~e^(-d/19.8)). Reasoning tasks show systematically lower cross-layer coupling.
5. **Reasoning Step Spectral Punctuation**: phase transition signatures in the α gradient coincide with reasoning step boundaries ("Step 1:", "therefore", new paragraphs).
6. **Perfect Spectral Correctness Prediction**: spectral α alone achieves AUC=1.000 (Qwen2.5-7B, late layers) and mean AUC=0.893 across 6 models in predicting correctness **before** the final answer is generated.

**Connection to Bifrost**: This is direct evidence for C2 (multi-scale coherence) in language models. The spectral α is essentially a measure of the eigenspectrum decay — a multi-scale property. The fact that it predicts correctness with AUC=1.000 means the spectral structure IS the semantic structure. The cross-layer synchronization decay (ρ~e^(-d/19.8)) is the language-model analog of our Experiment 2C finding that cross-scale coherence profiles are semantic. The "spectral punctuation" at reasoning step boundaries is the language analog of our cross-scale coherence being necessary for semantic structure.

**Spectral Probing** (Müller-Eberstein et al., EMNLP 2022) develops a learnable frequency filter to identify spectral profiles for NLP tasks. Finds that linguistic information is encoded at varying timescales (subwords, phrases) and that distinctive spectral profiles quantify cross-task similarity in a linguistically intuitive manner, remaining consistent across languages.

**CAST** (arXiv:2510.14262) analyzes transformer layer functions through spectral decomposition. Finds decoder models exhibit compression-expansion cycles while encoder models maintain consistent high-rank processing. Layers partition into three phases: feature extraction, compression, and specialization.

### Thread 3: Wavelet and Multi-Scale Methods in LLMs

**Wavelet GPT** (arXiv:2409.12924) infuses wavelet multi-scale structure into GPT-style LLM pre-training. Key results:

- **Same pre-training performance at 2x speed** for text, audio, and images.
- No extra parameters — imposes structure on intermediate embeddings via Haar/learnable wavelet pipeline.
- Every next token prediction gets access to intermediate embeddings at different temporal resolutions in every decoder block.
- Extends to Long Range Arena benchmark and multiple input representations (characters, BPE tokens, bytes, waveform, image pixels).

**Connection to Bifrost**: This directly implements our C2 claim (multi-scale coherence is necessary) in a language model. The 2x speedup with no performance loss shows that multi-scale structure is not just theoretically necessary but practically beneficial — it provides a better inductive bias for language.

**Multi-scale Transformer LMs** (Khandelwal et al., 2020, arXiv:2005.00581) presents three architectures with multi-scale inductive bias. Results:

- Hierarchical variant with 30 layers has **23% smaller memory footprint and better perplexity** than a vanilla transformer with less than half the layers.
- Favorable likelihood vs memory footprint trade-offs on Toronto BookCorpus.

**DWT for Embeddings** (ACL 2024 Findings) applies Discrete Wavelet Transforms to word and sentence embeddings:

- **50-93% dimensionality reduction with almost no change in performance** for semantic similarity tasks.
- Superior accuracy in most downstream tasks.
- DWT consolidates important semantic information in embedding vectors.

**Learnable Multi-Scale Wavelet Transformer (LMWT)** (arXiv:2504.08801) replaces self-attention with learnable multi-scale Haar wavelet transform:

- **Linear O(N) scaling** vs quadratic O(N²) self-attention.
- Competitive performance on WMT16 En-De translation.
- Learned wavelet coefficients provide interpretability.

**Graph Wavelet Transformer** (arXiv:2505.07862) replaces attention with learnable multi-scale wavelet transform over graph Laplacian from syntactic/semantic parses:

- Linear-time mixing capturing both local syntactic and global semantic context.
- K≪N bandpass filters in graph Fourier domain.

**Mamba/SSMs** (Gu & Dao, 2023): The S6 layer in Mamba can represent projections onto Haar wavelets (arXiv:2506.11891), providing an edge over S4D in approximating discontinuous functions. Mamba's selective state spaces are fundamentally spectral filters with input-dependent gating.

### Thread 4: Cross-Modal Alignment — The Modality Gap Problem

**ImageBind** (CVPR 2023) aligns 6 modalities (images, text, audio, depth, thermal, IMU) using only image-paired data. Key insight: all combinations of paired data are not necessary — image-paired data is sufficient to "bind" modalities together. Emergent alignment between modalities that are never observed together.

**Connection to Bifrost**: This is directly relevant to our Experiment 3C/3D. ImageBind shows that cross-modal alignment is possible, but it uses a **large-scale** approach (millions of image-paired samples) with **frozen pre-trained encoders**. Our Experiment 3D failed because we used a simple linear UCM with small synthetic data. The lesson: cross-modal alignment requires either (a) much larger data, (b) much more powerful encoders, or (c) a fundamentally different architecture.

**The Modality Gap** (Liang et al., NeurIPS 2022) is critical for understanding our 3D failure:

- Different modalities occupy **distinct cones** in the shared embedding space, even after contrastive training.
- The gap originates from **random initialization** (different random seeds create different cones).
- **Contrastive loss preserves and worsens the gap** — it does not close it.
- Each neural network layer **shrinks the angle** between embedding vectors, creating narrower cones in deeper architectures.
- Increasing the modality gap can actually **improve** downstream performance on some zero-shot tasks.

**Connection to Bifrost**: This explains our Experiment 3D result (silhouette by modality = 0.873, by category = -0.019). The modality gap is a **fundamental geometric phenomenon**, not a bug in our UCM. Even CLIP with 400M image-text pairs has this gap. The implication: our simple linear UCM cannot overcome the modality gap. Closing it requires:

1. **Shared encoders** (weight sharing between modalities) — as in AlignCLIP.
2. **Modality swapping** during training.
3. **Nonlinear projections** with sufficient capacity.
4. **Much larger datasets** (ImageBind uses millions of pairs; we used 200).

**CoAVT** (arXiv:2401.12264) uses a cognition-inspired approach with separate verbal (text) and non-verbal (audio-visual) systems that interact via a query encoder. This dual-system approach may be more appropriate for Bifrost than a single unified space.

**OneEncoder** (arXiv:2409.11059) uses a lightweight Universal Projection module that progressively aligns modalities. Key insight: train a lightweight projection to align image and text first, then freeze it and progressively align new modalities. This is more practical than training all modalities from scratch.

### Thread 5: FNet — Fourier Transforms Replace Attention

**FNet** (Lee-Thorp et al., NAACL 2022) replaces self-attention with unparameterized FFT:

- **92-97% of BERT accuracy** on GLUE benchmark.
- **80% faster training** on GPUs, 70% faster on TPUs.
- Significantly faster at longer input lengths.
- Light memory footprint, particularly efficient at smaller model sizes.

**Connection to Bifrost**: FNet shows that spectral mixing (Fourier transforms) can replace attention with minimal accuracy loss. However, FNet uses **real-valued** FFT, which restricts implicit phase to {0, π}. PRISM extends this to the full complex plane, showing that continuous phase offsets express semantic differences that {0, π} cannot. This is the language-model analog of our Experiment 1A finding that phase_randomize (which destroys continuous phase) is more destructive than phase_zero (which sets phase to 0).

---

## 3. Integration Architecture: Five Approaches

Based on the literature and our experimental results, here are five concrete approaches to integrating structured resonance into language models, ordered from least to most invasive:

### Approach A: Spectral Probing (Analysis Only)

**What**: Apply our phase coherence metrics (PLV, phase entropy, phase congruency) to existing LLM hidden states as a diagnostic tool.

**How**:
1. Extract hidden states from each layer of a pre-trained LLM
2. Compute FFT of hidden state sequences across token dimension
3. Apply PhaseCoherenceSignalMetrics to measure PLV, phase entropy, phase stability
4. Correlate with task performance, reasoning steps, correctness

**Evidence**: The Spectral Geometry of Thought paper shows spectral α predicts correctness with AUC=1.000. Our Experiment 1C shows phase_stability correlates with classification confidence (r=-0.31).

**Tradeoffs**:
- (+) No model changes, no retraining
- (+) Directly tests whether existing LLMs already use phase coherence
- (-) Cannot improve model performance, only analyze it
- (-) May require large models to see clear spectral structure

**Estimated effort**: Low. Can be implemented in days using existing Bifrost metrics modules.

### Approach B: Wavelet-Augmented Pre-training

**What**: Add multi-scale wavelet filters to intermediate embeddings during pre-training, following Wavelet GPT.

**How**:
1. Insert Haar/learnable wavelet transform after each decoder block
2. Allow next-token prediction to access multi-scale intermediate embeddings
3. Pre-train from scratch or fine-tune from existing checkpoint

**Evidence**: Wavelet GPT achieves 2x faster pre-training with same performance. Our Experiment 2A/2B shows multi-scale coherence is necessary (all ablations significant).

**Tradeoffs**:
- (+) 2x faster pre-training (Wavelet GPT result)
- (+) No extra parameters (wavelet transform is parameter-free)
- (+) Better inductive bias for hierarchical language structure
- (-) Requires pre-training from scratch for full benefit
- (-) Wavelet choice (Haar vs learnable) affects results
- (-) May not help for models already trained without wavelets

**Estimated effort**: Medium. Requires modifying transformer architecture and pre-training pipeline.

### Approach C: Complex-Valued Token Embeddings

**What**: Replace real-valued token embeddings with complex-valued representations where magnitude = global semantics and phase = token relationships, following Token2Wave.

**How**:
1. Represent each token as G·e^(iα) where G is magnitude vector, α is phase vector
2. Use wave interference and modulation for context-dependent embedding updates
3. Replace dot-product attention with phase-coherent similarity

**Evidence**: Token2Wave reduces memory and training time vs BERT. PRISM shows phase coherence correlates with semantic relationships. PCT outperforms softmax transformer.

**Tradeoffs**:
- (+) Phase explicitly encodes token relationships
- (+) Reduced memory and training time (Token2Wave)
- (+) Wave interference is a natural mechanism for context combination
- (-) Complex-valued operations require 2x memory per parameter
- (-) Backpropagation through complex domain requires Wirtinger calculus
- (-) Existing GPU hardware optimized for real-valued operations
- (-) Phase wrapping (modulo 2π) creates gradient discontinuities

**Estimated effort**: High. Requires fundamental changes to embedding layer, attention mechanism, and training pipeline.

### Approach D: Phase-Preserving Attention Replacement

**What**: Replace softmax attention with phase-coherent gating, following PCT and PRISM.

**How**:
1. L2-normalize complex query-key similarities
2. Apply real-valued, element-independent, smooth gate (not softmax competition)
3. Preserve phase information across layers
4. Use Gated Harmonic Convolution (O(N log N)) instead of quadratic attention

**Evidence**: PCT outperforms softmax transformer and shows no depth collapse. PRISM's hybrid matches Transformer at 33M params. FNet achieves 92-97% BERT accuracy with FFT mixing.

**Tradeoffs**:
- (+) O(N log N) complexity vs O(N²) attention
- (+) No depth-related accuracy collapse (PCT result)
- (+) Phase preserved across layers (not destroyed by softmax)
- (-) Loses content-based selectivity (the key Mamba insight)
- (-) PRISM shows spectral density threshold — fails on short sequences
- (-) Pure phase-based computation may not scale to large models (PRISM tested only 33M)
- (-) Hybrid approach (phase encoder + standard attention) may be needed for scale

**Estimated effort**: High. Requires replacing core attention mechanism.

### Approach E: Cross-Modal Coherence Binding

**What**: Extend Bifrost's UnifiedCoherenceMetric to bind language with audio/image/sensor modalities, following ImageBind but with coherence features.

**How**:
1. Extract coherence features from each modality (CBMPC for audio, PhaseCongruency for image, WaveletCoherence for sensor)
2. Extract spectral features from LLM hidden states (FFT across token dimension)
3. Train a contrastive model to align coherence features across modalities
4. Use shared encoder weights or progressive alignment (OneEncoder approach)

**Evidence**: ImageBind shows 6-modal alignment is possible with image-paired data. Our Experiment 3C shows partial cross-modal transfer (67.75% vs 20% chance). Our Experiment 3D shows the modality gap is the key obstacle.

**Tradeoffs**:
- (+) Directly tests C3 (cross-modal generalization) with language
- (+) Coherence features are modality-agnostic in principle
- (-) Modality gap is a fundamental geometric problem (NeurIPS 2022)
- (-) Requires large paired datasets (ImageBind: millions; we had 200)
- (-) Simple linear projections fail (our 3D result)
- (-) Need shared encoders or modality swapping to close gap
- (-) Language has no direct analog of audio phase or image phase — must use spectral features of hidden states

**Estimated effort**: Very High. Requires large-scale multi-modal training infrastructure.

---

## 4. Engineering Tradeoffs: A Summary Matrix

| Tradeoff | Approach A | Approach B | Approach C | Approach D | Approach E |
|----------|-----------|-----------|-----------|-----------|-----------|
| **Implementation effort** | Low | Medium | High | High | Very High |
| **Requires retraining** | No | Yes | Yes | Yes | Yes |
| **Compute complexity** | O(N²) | O(N log N) | O(N log N) | O(N log N) | O(N²) + alignment |
| **Memory overhead** | None | None | 2x (complex) | 2x (complex) | Large (multi-modal) |
| **Scalability to 7B+** | Yes | Yes | Uncertain | Uncertain | Yes (with data) |
| **Preserves phase** | Measures it | Implicit | Explicit | Explicit | Explicit |
| **Multi-scale** | No | Yes | No | Yes | Yes |
| **Cross-modal** | No | No | No | No | Yes |
| **Hardware compatible** | Yes | Yes | Partial | Partial | Yes |
| **Data requirements** | None | Standard | Standard | Standard | Large paired |

---

## 5. Key Engineering Insights

### 5.1 The Phase-Magnitude Tradeoff

PRISM's unit-norm constraint (|z|=1) is the cleanest test of phase-only computation, but it's too restrictive for practical LLMs. The literature suggests a **hybrid approach**:

- **Phase for structure**: Use phase to encode relationships (token-to-token, layer-to-layer)
- **Magnitude for salience**: Use magnitude to encode importance/frequency
- **Wave-Particle architecture**: PRISM's hybrid (phase encoder + magnitude attention) matches Transformer baselines

This mirrors our experimental finding: phase ablation destroys structure (C1), but amplitude-only baselines still achieve some accuracy. Both phase AND magnitude carry information — phase carries structure, magnitude carries salience.

### 5.2 The Spectral Density Threshold

PRISM discovers that phase-based computation requires minimum sequence length — isolated tokens fail because there aren't enough components to create interference patterns. This has a direct engineering implication:

- **Short sequences**: Standard attention is better (phase computation fails)
- **Long sequences**: Phase/spectral methods are better (O(N log N) and interference patterns emerge)
- **Hybrid**: Use attention for short-range, spectral mixing for long-range

This is consistent with FNet's result: FNet matches efficient transformers on Long Range Arena but loses 3-8% on GLUE (short sequences).

### 5.3 The Modality Gap is Fundamental

Our Experiment 3D failure (modality silhouette = 0.873, category silhouette = -0.019) is not a bug — it's a fundamental geometric phenomenon documented in CLIP:

- Different modalities occupy distinct cones at initialization
- Contrastive loss preserves the gap
- Deeper networks create narrower cones (worse gap)
- Closing the gap requires: shared encoders, modality swapping, or very large data

**Implication for Bifrost**: The UnifiedCoherenceMetric needs to be fundamentally redesigned for cross-modal alignment. Options:
1. **Shared encoder weights** across modalities (AlignCLIP approach)
2. **Progressive alignment** (OneEncoder: align two modalities, freeze, add more)
3. **Dual-system architecture** (CoAVT: separate verbal/non-verbal systems)
4. **Phase-based binding**: Use phase coherence as the alignment signal instead of cosine similarity

### 5.4 Complex-Valued Training is Hard

Multiple papers note the engineering challenges of complex-valued neural networks:

- **Wirtinger calculus** required for backpropagation through complex domain
- **Phase wrapping** (modulo 2π) creates gradient discontinuities
- **GPU optimization**: Current hardware/libraries optimized for real-valued operations
- **Initialization**: Complex-valued weight initialization is less well understood
- **Normalization**: BatchNorm/LayerNorm need complex-valued variants (ModReLU, complex LayerNorm)

**Mitigation**: PRISM's hybrid approach — use complex-valued encoding for early layers (where phase structure is established) and real-valued processing for later layers (where magnitude-based reasoning happens). This gives "the best of both worlds" while minimizing complex-valued training challenges.

### 5.5 Multi-Scale is Free Performance

The most practically impactful finding is that multi-scale structure provides "free" performance improvements:

- **Wavelet GPT**: 2x faster pre-training, same performance
- **Multi-scale Transformer**: 23% smaller memory, better perplexity
- **DWT for embeddings**: 50-93% dimensionality reduction, no performance loss
- **Mamba S6**: Can represent Haar wavelets, enabling discontinuous function approximation

**Implication for Bifrost**: The multi-scale coherence claim (C2) is not just theoretically supported — it's the most immediately practical finding. Adding wavelet multi-scale filters to existing LLM architectures is low-risk, high-reward.

---

## 6. Recommended Integration Path for Bifrost

Based on the literature and our experimental results, here is a phased integration path:

### Phase 1: Spectral Analysis of Existing LLMs (Approach A)

**Goal**: Verify that existing LLMs already exhibit phase coherence structure.

**Method**:
1. Extract hidden states from Llama/Qwen/Mistral at each layer
2. Apply Bifrost's PhaseCoherenceSignalMetrics (PLV, phase entropy, phase stability)
3. Test: Do spectral metrics predict task performance? (Following Spectral Geometry of Thought)
4. Test: Does phase ablation on hidden states destroy semantic structure? (Following our Experiment 1A method)

**Expected result**: Based on the Spectral Geometry of Thought paper, we expect to find:
- Spectral α predicts correctness (AUC > 0.89)
- Cross-layer synchronization decays exponentially
- Phase transitions at reasoning step boundaries

**Effort**: 1-2 weeks. Uses existing Bifrost modules.

### Phase 2: Wavelet-Augmented Fine-Tuning (Approach B)

**Goal**: Improve LLM fine-tuning with multi-scale wavelet filters.

**Method**:
1. Insert Haar wavelet transform after each decoder block (following Wavelet GPT)
2. Fine-tune on downstream tasks (GLUE, SuperGLUE)
3. Compare with baseline fine-tuning

**Expected result**: Based on Wavelet GPT, expect 1.5-2x faster convergence with same or better accuracy.

**Effort**: 2-4 weeks. Requires modifying transformer architecture.

### Phase 3: Phase-Coherent Attention (Approach D, hybrid)

**Goal**: Replace softmax attention with phase-preserving gating in a hybrid architecture.

**Method**:
1. Implement PCT-style phase-coherent attention for early layers (phase establishment)
2. Keep standard attention for later layers (magnitude-based reasoning)
3. Train on language modeling benchmark (WikiText-103 or similar)

**Expected result**: Based on PCT and PRISM, expect:
- No depth-related accuracy collapse
- Competitive or better performance
- O(N log N) complexity for phase-coherent layers

**Effort**: 4-8 weeks. Requires custom attention implementation and complex-valued training.

### Phase 4: Cross-Modal Coherence Binding (Approach E)

**Goal**: Bind language with audio/image/sensor modalities via coherence features.

**Method**:
1. Use shared encoder weights across modalities (to close modality gap)
2. Train with progressive alignment (OneEncoder approach)
3. Use phase coherence as alignment signal (not just cosine similarity)
4. Requires large paired dataset (AudioSet, VGGSound, or synthetic)

**Expected result**: Based on ImageBind, expect emergent cross-modal alignment with sufficient data. Based on our 3D failure, do NOT expect simple linear projections to work.

**Effort**: 3-6 months. Requires multi-modal training infrastructure and large datasets.

---

## 7. What This Means for the Structured Resonance Thesis

The literature provides **strong independent corroboration** of the thesis:

| Thesis Claim | Literature Evidence | Strength |
|---|---|---|
| C1: Phase captures structure | PRISM: synonyms have higher phase coherence (p<0.001). PCT: phase-preserving attention outperforms softmax. | **Strong** |
| C2: Multi-scale is necessary | Wavelet GPT: 2x faster with wavelets. Spectral Geometry: spectral α predicts correctness (AUC=1.000). Multi-scale Transformer: 23% smaller, better perplexity. | **Strong** |
| C3: Cross-modal generalization | ImageBind: 6-modal alignment possible. Modality Gap: fundamental geometric obstacle. Our 3D: simple UCM fails. | **Partial** — possible with right architecture and data, but not trivial |

The thesis is **not just theoretically motivated but practically validated** by independent work. The key insight from the literature is that the thesis works best as a **hybrid approach**: phase for structure, magnitude for salience, multi-scale for hierarchy, and shared encoders for cross-modal alignment.

---

## References

1. Yıldırım, A. & Yücedağ, İ. (2025). Language as a Wave Phenomenon: Semantic Phase Locking and Interference in Neural Networks. arXiv:2512.01208.
2. Spectral Geometry of Thought (2026). arXiv:2604.15350.
3. Wavelet GPT (2024). arXiv:2409.12924.
4. Zhang, X. & Sheng, V.S. (2024). Token2Wave. arXiv:2411.06989.
5. Phase-Coherent Transformer (2026). arXiv:2605.10123.
6. Lee-Thorp et al. (2022). FNet: Mixing Tokens with Fourier Transforms. NAACL.
7. CAWN: Continuous Acoustic Wave Networks (2026). arXiv:2604.04250.
8. Complex Hilbert Space LLMs (2026). Zenodo:17890922.
9. Khandelwal et al. (2020). Multi-scale Transformer Language Models. arXiv:2005.00581.
10. Müller-Eberstein et al. (2022). Spectral Probing. EMNLP.
11. DWT for Embeddings (2024). ACL Findings.
12. LMWT: Learnable Multi-Scale Wavelet Transformer (2025). arXiv:2504.08801.
13. Graph Wavelet Transformer (2025). arXiv:2505.07862.
14. Girdhar et al. (2023). ImageBind: One Embedding Space To Bind Them All. CVPR.
15. Liang et al. (2022). Mind the Gap: Understanding the Modality Gap. NeurIPS.
16. Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
17. Input Selectivity in Mamba (2025). arXiv:2506.11891.
18. CoAVT (2024). arXiv:2401.12264.
19. OneEncoder (2024). arXiv:2409.11059.
20. CAST (2025). arXiv:2510.14262.
21. AlignCLIP (2024). arXiv:2406.17639.
22. Tracing Representation Geometry of LMs (2025). arXiv:2509.23024.
23. Emergence of High-Dimensional Abstraction Phase (2024). arXiv:2405.15471.
24. UMST: Universal MultiScale Transformer (ICML 2022).
