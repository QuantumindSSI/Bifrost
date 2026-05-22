# Quantumind FBC Core

**Frequency-Based Cognition (FBC) Framework**

Phase-coherent neural computation for artificial general intelligence. Implements ResonanceAttention — an attention mechanism based on phase-locking rather than dot-product similarity.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

## Phase 1 Deliverables

✅ **S0 Signal Canonicalizer**: Converts raw multi-modal signals into canonical spectral tensor representation
✅ **S1 Spectral Decomposer**: Multi-resolution FFT + wavelet decomposition with Mamba-3 backbone
✅ **Resonance Attention**: Phase-coherence based attention mechanism (replaces dot-product attention)

## Key Features

- **Spectral-First Processing**: All inputs converted to frequency domain for structural invariance
- **Phase-Coherence Routing**: Attention based on phase alignment, not dot-product similarity
- **Multi-Resolution Decomposition**: FFT + continuous wavelet transforms across time-frequency
- **GPU Acceleration**: Native CUDA support via PyTorch and custom kernels
- **Experiment Tracking**: W&B integration for all benchmarks and ablations
- **Production-Ready**: Modular, testable, benchmarked against standard attention

## Data Inputs & Handling

### Supported Input Types

1. **Time-Series Data** (univariate/multivariate)
   - Shape: `(batch, seq_len)` or `(batch, seq_len, channels)`
   - Normalized to unit variance before FFT

2. **Pre-Computed Spectrograms**
   - Shape: `(batch, freq_bins, time_steps, channels)`
   - Direct amplitude + phase extraction

3. **Sensor Signals** (robot proprioception, IMU, pressure)
   - Raw sensor reads normalized per-channel
   - Canonical form: amplitude, phase, scale, uncertainty

4. **Multimodal Signals** (vision, audio, proprioception)
   - Each modality independently canonicalized
   - Cross-modal binding via resonance attention

### Input Processing Pipeline

```
Raw Input
    ↓
[S0_Canonicalizer]  → Normalize, preserve amplitude + phase
    ↓
[S1_SpectralDecomposer]  → FFT + wavelet, extract multiresolution bands
    ↓
[SpectralTensor]  → {amplitude, phase, scale, uncertainty}
    ↓
[ResonanceAttention]  → Phase-coherence routing
    ↓
Output
```

### Data Handling

- **Normalization**: Per-channel standardization (mean=0, std=1)
- **FFT Window**: Hann window, overlap 50%, zero-pad to power of 2
- **Wavelet Scales**: 16-64 scales covering 1 Hz to Nyquist frequency
- **Phase Extraction**: Continuous phase via analytic signal (Hilbert transform)
- **Uncertainty**: Confidence per frequency bin (inverse SNR, coherence-based)

## Installation

```bash
# Clone the repository
git clone https://github.com/quantumind/fbc-core.git
cd fbc-core

# Install with CPU support
pip install -e .

# Development setup
pip install -e ".[dev]"
```

## Quick Start

### Option 1: Using Bundled Sample Data (Fastest)

```python
from fbc.data import quick_start_pipeline, list_samples

# See what samples are available
print(list_samples())
# {'audio': ['mono_16khz', 'mono_8khz', 'stereo_44khz'],
#  'image': ['gray', 'rgb', 'rgb_large']}

# Run full pipeline on sample audio with one command
s0_out, s1_out, s2_out, attention = quick_start_pipeline("mono_16khz")
print(f"Spectral tensor: {s2_out.amplitude.shape}")
print(f"Attention weights: {attention.shape}")
```

### Option 2: Manual Pipeline Setup

```python
import torch
from fbc.data import load_sample_audio
from fbc.s0_canonicalizer import S0_Canonicalizer
from fbc.s1_decomposer import S1_SpectralDecomposer
from fbc.resonance_attention import ResonanceAttention

# Load sample audio
audio, sr = load_sample_audio("mono_16khz")  # Returns torch.Tensor

# Or load your own audio
# audio = torch.randn(1, 16000)  # Your data here

# Initialize FBC pipeline
s0 = S0_Canonicalizer(sample_rate=16000)
s1 = S1_SpectralDecomposer(n_scales=32, mamba_dim=256)
attn = ResonanceAttention(num_heads=8, hidden_dim=256)

# Process
canonical = s0(audio)
spectral = s1(canonical)
output, weights = attn(spectral)
```

## Command Line Interface

After installation, use the `fbc` command:

```bash
# Run atomic demos
fbc demo 1              # Anti-phase discrimination
fbc demo 2              # Harmonic binding
fbc demo 3              # Cross-modal retrieval
fbc demo all            # Run all demos

# Process audio through FBC pipeline (S0-S2)
fbc process mono_16khz              # Process sample
fbc process myfile.wav -o output.pt # Process file, save results

# Extract FrequencyAttractors (S3)
fbc attractors mono_16khz -o attractors.pt          # From sample
fbc attractors myfile.wav -o att.pt --domain audio  # From file
fbc attractors mono_16khz --n-bands 8 --max-display 5

# Phase-Lock Bridge evaluation (S4)
fbc bridge attractors_a.pt attractors_b.pt          # Evaluate cross-domain bridges
fbc bridge src.pt tgt.pt -o bridges.pt --min-locked 3
fbc bridge src.pt tgt.pt --activation-threshold 0.7 --band-threshold 0.5

# List available sample data
fbc samples

# Run benchmarks
fbc bench attention     # Attention comparison benchmark
fbc bench realistic     # Realistic workload benchmark

# Get help
fbc --help
fbc demo --help
fbc attractors --help
fbc bridge --help
```

### S3: Attractor Extraction

Extract stable spectral patterns (FrequencyAttractors) from audio:

```bash
fbc attractors mono_16khz -o audio_attractors.pt
```

**Output:**
- List of `FrequencyAttractor` objects with centroid, phase signature, and amplitude profile
- Saved to `.pt` file for downstream S4 bridge evaluation

**Options:**
- `-n, --n-bands`: Number of spectral bands per attractor (default: 8)
- `--domain`: Domain label for attractors (default: audio)
- `--prefix`: ID prefix for attractor naming (default: att)

### S4: Phase-Lock Bridge

Evaluate phase-locked relationships between two attractor sets:

```bash
# Extract attractors from two different audio samples
fbc attractors mono_16khz -o audio_16k.pt --domain audio
fbc attractors mono_8khz -o audio_8k.pt --domain audio

# Evaluate phase-lock bridge between them
fbc bridge audio_16k.pt audio_8k.pt -o bridges.pt
```

**Output:**
- List of activated bridge candidates with activation scores
- Per-band coherence metrics
- Configurable thresholds for bridge activation

**Options:**
- `--min-locked`: Minimum bands required for activation (default: 3)
- `--band-threshold`: Per-band coherence threshold (default: 0.5)
- `--activation-threshold`: Overall activation threshold (default: 0.6)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fbc --cov-report=html

# Run specific test
pytest tests/test_s0_canonicalizer.py -v
```

## Benchmarking

```bash
# Compare resonance attention vs dot-product attention
python benchmarks/attention_comparison.py \
  --seq_len 512 \
  --hidden_dim 256 \
  --num_heads 8 \
  --num_iters 100

# Log results to W&B
python benchmarks/attention_comparison.py \
  --seq_len 512 \
  --use_wandb \
  --project "fbc-phase1"
```

## Project Structure

```
fbc-core/
├── src/fbc/                    # Core source code
│   ├── __init__.py
│   ├── cli/                    # Command line interface
│   │   ├── __init__.py
│   │   └── main.py             # fbc command implementation
│   ├── data/                   # Sample data utilities
│   │   ├── __init__.py
│   │   └── loader.py           # quick_start_pipeline, load_sample_audio
│   ├── s0_canonicalizer/       # Signal canonicalization
│   │   ├── __init__.py
│   │   └── canonicalizer.py
│   ├── s1_decomposer/          # Spectral decomposition
│   │   ├── __init__.py
│   │   └── decomposer.py
│   ├── resonance_attention/    # Phase-coherence attention
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   └── binding.py
│   ├── phase_lock_bridge/     # Cross-modal transfer
│   │   ├── __init__.py
│   │   ├── bridge.py
│   │   └── attractor.py
│   ├── spectral_tensor/        # Data structures
│   │   ├── __init__.py
│   │   └── spectral_tensor.py
│   ├── bridge.py               # Pipeline bridge
│   └── pipeline.py             # End-to-end pipeline
├── tests/                      # Unit tests
├── benchmarks/                 # Performance benchmarks
├── demos/                      # Atomic demos (Demo 1-3)
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

- Kalman (1960): State space models for filtering
- Gu & Dao (2023): Mamba - Linear time sequence modeling
- Ha & Schmidhuber (2018): World models
- Hafner et al. (2020): Dreamer - Scalable world models
- Quantumind FBC v1-v3: Frequency-based cognition framework

## Licensing

MIT License. See [LICENSE](./LICENSE) file for details.

## Contact

**Quantumind Ltd**  
engineering@quantumind.io

---

*Phase 1 validated. Three atomic demos pass. Ready for Phase 2 integration.*
