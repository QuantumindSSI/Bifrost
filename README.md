# Bifrost

**Spectral Neural Computation Framework**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Bifrost is a frequency-domain neural computation framework that processes information through complex-valued spectral representations. By operating jointly in the amplitude and phase domains, Bifrost enables phase-coherent attention mechanisms and unified multi-modal spectral processing across audio, text, and image modalities.

## Overview

Traditional neural architectures process information as scalar activations, capturing signal magnitude while discarding phase structure. Bifrost addresses this by representing information as complex spectra — mathematical entities with both magnitude and angular components — enabling the system to capture phase alignment patterns that encode structural and topical relationships.

### Architecture

The framework implements a modular processing pipeline:

1. **Signal Canonicalization** — Converts raw multi-modal inputs into unified spectral tensor representations via Fourier transforms, with per-channel normalization and uncertainty quantification.

2. **Spectral Decomposition** — Processes complex spectra through complex-valued state space models. The core recurrence is implemented via a parallel associative scan (Blelloch algorithm, O(log n) depth) for efficient long-sequence modeling.

3. **Resonance Attention** — Routes information based on phase coherence alignment rather than dot-product similarity, enabling attention to topological relationships beyond semantic similarity.

4. **Phase-Lock Bridge** — Extracts stable frequency attractors with learned stability dynamics for cross-modal knowledge transfer, grounded in coupled oscillator physics via the Adler equation.

## Key Features

- **Spectral-First Processing** — All inputs converted to frequency domain for structural invariance
- **Phase-Coherence Routing** — Attention based on phase alignment across spectral components
- **Complex SSM Architecture** — Mathematically correct complex-valued state space model with parallel associative scan
- **Multi-Modal Canonicalization** — Unified spectral representation for audio, text, and image modalities
- **Learned Attractor Dynamics** — Neural stability predictor with temporal consistency tracking
- **GPU Acceleration** — Native CUDA support via PyTorch

## Installation

```bash
# Clone the repository
git clone https://github.com/quantumind/bifrost.git
cd bifrost

# Install with CPU support
pip install -e .

# Development setup
pip install -e ".[dev]"
```

## Quick Start

### Option 1: Using Bundled Sample Data

```python
from bifrost.data import quick_start_pipeline, list_samples

# See available samples
print(list_samples())
# {'audio': ['mono_16khz', 'mono_8khz', 'stereo_44khz'],
#  'image': ['gray', 'rgb', 'rgb_large']}

# Run full pipeline on sample audio
canonical, decomposed, spectral, attention = quick_start_pipeline("mono_16khz")
print(f"Spectral tensor: {spectral.amplitude.shape}")
print(f"Attention weights: {attention.shape}")
```

### Option 2: Manual Pipeline Setup

```python
import torch
from bifrost.data import load_sample_audio
from bifrost.canonicalizer.canonicalizer import SpectralCanonicalizer
from bifrost.decomposer.complex_decomposer import ComplexSpectralDecomposer
from bifrost.resonance_attention.attention import ResonanceAttention

# Load sample audio
audio, sr = load_sample_audio("mono_16khz")

# Initialize pipeline
canonicalizer = SpectralCanonicalizer(n_fft=1024, normalize_input=True)
decomposer = ComplexSpectralDecomposer(d_model=256, d_state=64)
attention = ResonanceAttention(num_heads=8, hidden_dim=256)

# Process
canonical = canonicalizer(audio)
spectral = decomposer(canonical)
output, weights = attention(spectral)
```

## Command Line Interface

After installation, use the `bifrost` command:

```bash
# Run atomic demos
bifrost demo 1              # Anti-phase discrimination
bifrost demo 2              # Harmonic binding
bifrost demo 3              # Cross-modal retrieval
bifrost demo all            # Run all demos

# Process audio through pipeline
bifrost process mono_16khz              # Process sample
bifrost process myfile.wav -o output.pt # Process file, save results

# Extract frequency attractors
bifrost attractors mono_16khz -o attractors.pt
bifrost attractors myfile.wav -o att.pt --domain audio

# Evaluate phase-lock bridge between attractor sets
bifrost bridge attractors_a.pt attractors_b.pt
bifrost bridge src.pt tgt.pt -o bridges.pt --min-locked 3

# List available sample data
bifrost samples

# Run benchmarks
bifrost bench attention     # Attention comparison benchmark
bifrost bench realistic     # Realistic workload benchmark

# Get help
bifrost --help
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=bifrost --cov-report=html

# Run specific component tests
pytest tests/test_canonicalizer.py -v
pytest tests/test_decomposer.py -v
```

## Benchmarking

```bash
# Compare resonance attention vs dot-product attention
python benchmarks/attention_comparison.py \
  --seq_len 512 \
  --hidden_dim 256 \
  --num_heads 8 \
  --num_iters 100
```

## Project Structure

```
bifrost/
├── src/bifrost/                # Core source code
│   ├── __init__.py
│   ├── cli/                    # Command line interface
│   ├── data/                   # Sample data utilities
│   ├── canonicalizer/          # Signal canonicalization
│   ├── decomposer/             # Spectral decomposition
│   ├── resonance_attention/    # Phase-coherence attention
│   ├── phase_lock_bridge/      # Cross-modal transfer
│   ├── s3_attractor/           # Learned attractor dynamics
│   ├── semantic_coherence/     # Semantic coherence metrics
│   ├── spectral_tensor.py      # Data structures
│   ├── pipeline.py             # End-to-end pipeline
│   └── llm_adapter.py          # LLM integration adapter
├── tests/                      # Unit tests
├── benchmarks/                 # Performance benchmarks
├── demos/                      # Atomic demos
├── docs/                       # Documentation
│   ├── guides/
│   ├── design/
│   └── api/
├── sample_data/                # Bundled audio/image samples
├── configs/                    # Configuration files
├── pyproject.toml              # Package config
├── README.md
└── LICENSE
```

## Research References

- Gu & Dao (2023): Mamba — Linear-Time Sequence Modeling with Selective State Spaces
- Trabelsi et al. (2018): Deep Complex Networks
- Blelloch (1990): Prefix Sums and Their Applications
- Adler (1946): A Study of Locking Phenomena in Oscillators

## License

MIT License. See [LICENSE](./LICENSE) for details.

## Contact

**Quantumind**  
engineering@quantumind.io
