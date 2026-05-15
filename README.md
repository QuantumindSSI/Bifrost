# FBC Phase 1: Spectral Encoder & Resonance Attention

**Quantumind QSSI Research Programme**

First-stage implementation of Frequency-Based Cognition (FBC) for artificial general intelligence.

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
# Clone or navigate to fbc-phase1
cd fbc-phase1

# Install with CPU support
pip install -e .

# Install with GPU + Mamba-3
pip install -e ".[cuda]"

# Development setup
pip install -e ".[dev,cuda]"
```

## Quick Start

```python
import torch
from fbc.s0_canonicalizer import S0_Canonicalizer
from fbc.s1_decomposer import S1_SpectralDecomposer
from fbc.resonance_attention import ResonanceAttention

# Initialize components
s0 = S0_Canonicalizer(sample_rate=16000)
s1 = S1_SpectralDecomposer(n_scales=32, mamba_dim=256)
attn = ResonanceAttention(num_heads=8, hidden_dim=256)

# Process input signal
x = torch.randn(4, 16000)  # (batch=4, seq_len=16000)
canonical = s0(x)  # SpectralTensor with amplitude, phase, scale, uncertainty
spectral_embedding = s1(canonical)  # (batch, seq_len, embedding_dim)
output = attn(spectral_embedding)  # (batch, seq_len, embedding_dim)
```

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
fbc-phase1/
├── fbc/
│   ├── __init__.py
│   ├── s0_canonicalizer/       # Signal canonicalization
│   │   ├── __init__.py
│   │   ├── spectral_tensor.py  # SpectralTensor dataclass
│   │   └── canonicalizer.py    # S0_Canonicalizer module
│   ├── s1_decomposer/          # Spectral decomposition + Mamba-3
│   │   ├── __init__.py
│   │   ├── fft_bank.py         # FFT processing
│   │   ├── wavelet_bank.py     # Wavelet decomposition
│   │   └── decomposer.py       # S1_SpectralDecomposer + Mamba-3
│   └── resonance_attention/    # Phase-coherence attention
│       ├── __init__.py
│       ├── coherence.py        # Phase coherence computation
│       └── attention.py        # ResonanceAttention layer
├── tests/
│   ├── test_s0_canonicalizer.py
│   ├── test_s1_decomposer.py
│   ├── test_resonance_attention.py
│   └── test_integration.py
├── benchmarks/
│   ├── attention_comparison.py
│   ├── throughput.py
│   └── memory_profile.py
├── configs/
│   ├── s0_default.yaml
│   ├── s1_default.yaml
│   └── attention_default.yaml
├── data/
│   └── sample_signals/         # Example data for testing
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Research References

- Kalman (1960): State space models for filtering
- Gu & Dao (2023): Mamba - Linear time sequence modeling
- Ha & Schmidhuber (2018): World models
- Hafner et al. (2020): Dreamer - Scalable world models
- Quantumind FBC v1-v3: Frequency-based cognition framework

## Licensing

MIT License. See LICENSE file for details.

## Contact

**Quantumind Ltd**  
Research Division  
April 2026
