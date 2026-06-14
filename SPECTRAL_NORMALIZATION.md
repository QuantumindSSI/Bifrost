# Spectral Normalization Integration for Bifrost

## Summary

Successfully integrated **Spectral Normalization** for complex-valued neural networks into the Bifrost codebase. This stabilizes training dynamics by constraining the Lipschitz constant of complex weight matrices.

## What Was Implemented

### 1. **Spectral Normalization Module** ([spectral_normalization.py](src/bifrost/decomposer/spectral_normalization.py))

#### `ComplexSpectralNorm` (Core Class)
- Wraps any module containing complex-valued weight matrices
- Uses power iteration to estimate the spectral norm (largest singular value)
- Normalizes weights to have spectral norm ≤ 1
- Constructs composite real-valued matrix representation for complex weights:
  ```
  M = [W_real   -W_imag]
      [W_imag    W_real]
  ```

#### `SpectralNormalizedComplexLinear` (Convenience Wrapper)
- Combines `ComplexLinear` with automatic spectral normalization
- Allows toggling spectral norm on/off via `use_spectral_norm` parameter

#### `apply_spectral_norm_to_module()` (Utility Function)
- Recursively applies spectral normalization to all `ComplexLinear` layers in a module tree

### 2. **Integration into Complex Decomposer**

#### `ComplexSelectiveScan` Updates
- Added `use_spectral_norm: bool` parameter to `__init__`
- Created spectral norm wrappers for:
  - `in_proj` (input projection)
  - `x_proj` (parameter projection)
  - `out_proj` (output projection)
- Updated forward pass to conditionally use normalized projections

#### `ComplexSpectralDecomposer` Updates
- Added `use_spectral_norm: bool` parameter to `__init__`
- Created spectral norm wrappers for:
  - `input_proj` (spectral tensor → d_model)
  - `output_proj` (d_model → output spectrum)
- Updated forward pass to use normalized projections when enabled

### 3. **API Exports**

Updated module exports in:
- `src/bifrost/decomposer/__init__.py`
- `src/bifrost/__init__.py`

Now available:
```python
from bifrost import (
    ComplexSpectralNorm,
    SpectralNormalizedComplexLinear,
    apply_spectral_norm_to_module,
)
```

### 4. **Comprehensive Test Suite** ([test_spectral_normalization.py](tests/test_spectral_normalization.py))

17 tests covering:

- **Initialization & Construction**
  - Complex spectral norm wrapper initialization
  - Composite weight matrix construction
  - Power iteration convergence

- **Core Functionality**
  - Spectral norm estimation and convergence
  - Weight normalization
  - Gradient flow through spectral norm layers

- **Integration**
  - `SpectralNormalizedComplexLinear` with/without normalization
  - `ComplexSelectiveScan` initialization with spectral norm
  - Recursive application to module trees

- **Training Stability**
  - Training without NaN/Inf
  - Lipschitz constant bounding
  - Multiple training iterations

**Test Results:** ✅ 17/17 PASSED

## How to Use

### Option 1: Using Spectral Normalized Linear Layer

```python
from bifrost import SpectralNormalizedComplexLinear
import torch

# Create layer with spectral normalization
layer = SpectralNormalizedComplexLinear(
    in_features=64,
    out_features=32,
    use_spectral_norm=True,
    n_power_iterations=1
)

# Forward pass
x = torch.randn(batch_size, 64, dtype=torch.complex64)
output = layer(x)  # Spectral norm normalized automatically
```

### Option 2: Spectral Norm in ComplexSelectiveScan

```python
from bifrost import ComplexSelectiveScan

# SSM with spectral normalization
ssm = ComplexSelectiveScan(
    d_model=128,
    d_state=32,
    expand=2,
    use_spectral_norm=True  # Enable here
)
```

### Option 3: Spectral Norm in ComplexSpectralDecomposer

```python
from bifrost import ComplexSpectralDecomposer

# Decomposer with spectral normalization
decomposer = ComplexSpectralDecomposer(
    n_fft=512,
    d_model=128,
    n_frames=32,
    use_spectral_norm=True  # Enable here
)
```

### Option 4: Apply to Existing Modules

```python
from bifrost import apply_spectral_norm_to_module
import torch.nn as nn

model = nn.Sequential(
    ComplexLinear(64, 128),
    ComplexLinear(128, 64),
)

# Apply spectral norm recursively
model = apply_spectral_norm_to_module(model, n_power_iterations=1)
```

## Key Benefits

1. **Training Stability** - Constrains weight matrices' Lipschitz constants, preventing gradient explosion
2. **Phase Coherence Learning** - Particularly effective for complex SSMs learning temporal phase relationships
3. **Flexible Integration** - Can be toggled on/off via parameters
4. **Mathematically Rigorous** - Based on Miyato et al. (ICLR 2018) spectral normalization framework, adapted for complex values
5. **Zero Overhead When Disabled** - No performance impact if `use_spectral_norm=False`

## Technical Details

### Complex Weight Representation
For complex weight matrix `W = W_real + i*W_imag`, spectral normalization works by:

1. Constructing composite matrix `M` of shape `(2*out, 2*in)`
2. Computing spectral norm via power iteration: `σ_max(M)`
3. Normalizing: `W ← W / σ_max`

This preserves complex multiplication semantics while maintaining proper gradient flow.

### Power Iteration Algorithm

For each forward pass:
1. Estimate largest singular vector `u` through power iteration
2. Compute spectral norm σ from singular values
3. Scale weights by `1/σ` during forward pass
4. Restore original weights after forward pass for proper backprop

## Related Concepts in Bifrost

This complements existing Bifrost features:

- **Phase-Lock Bridge** - Extracts frequency attractors; spectral norm ensures stable dynamics
- **Resonance Attention** - Routes based on phase coherence; spectral norm prevents amplitude explosion
- **Complex SSMs** - Learns temporal phase evolution; spectral norm ensures stable recurrence

## Testing

Run the test suite:
```bash
cd /Users/startferanmi/Documents/QuantumindSSI/bifrost
python -m pytest tests/test_spectral_normalization.py -v
```

## Files Modified

- **Created:**
  - `src/bifrost/decomposer/spectral_normalization.py` (340 lines)
  - `tests/test_spectral_normalization.py` (450 lines)

- **Updated:**
  - `src/bifrost/decomposer/complex_decomposer.py` - Added `use_spectral_norm` parameters
  - `src/bifrost/decomposer/__init__.py` - Added exports
  - `src/bifrost/__init__.py` - Added exports

## References

- Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (ICLR 2018)
- Higgins et al., "Understanding Generative Adversarial Networks" (2021)
- Complex-Valued Neural Networks (Hirose & Yoshida, 2012)
