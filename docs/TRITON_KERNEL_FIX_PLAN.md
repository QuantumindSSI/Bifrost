# Triton Complex SSM Kernel Fix Plan

## Current Status: DISABLED

**Date**: 2026-05-29
**Status**: Triton backend disabled due to incomplete complex number support
**Current Backend**: CUDA PyTorch (fully functional)

---

## Problem Summary

The Triton kernel at `src/bifrost/s1_decomposer/complex_ssm_triton.py` claims to implement
complex-valued selective scan SSM but has critical bugs:

1. **Imaginary parts hardcoded to 0.0** — Only real computation happens
2. **Output only stores real component** — Discards all imaginary data
3. **Complex multiplication incorrect** — Due to (1), complex math is broken

**Impact**: Kernel produces incorrect results while claiming 10-50x speedup.

---

## Root Cause

Triton language (as of v2.x) has **no native complex number type**. 
Workarounds required:
- Store real and imaginary parts separately
- Manually implement all complex arithmetic operations
- Manage two separate buffers throughout computation

The current implementation took shortcuts and only implemented real arithmetic.

---

## Fix Plan

### Phase 1: Data Structure Redesign (High Priority)

Change all complex tensor handling from:
```python
# Current (broken)
x: torch.Tensor  # complex64
triton kernel: x_real = load(...), x_imag = 0.0
```

To:
```python
# Fixed
x_real: torch.Tensor  # float32
x_imag: torch.Tensor  # float32
kernel: pass both pointers, compute both paths
```

**Changes Required**:

1. **Kernel Signature Update**:
```python
@triton.jit
def _complex_selective_scan_kernel(
    x_real_ptr, x_imag_ptr,  # Separate pointers for real/imag
    delta_ptr,
    A_real_ptr, A_imag_ptr,
    B_real_ptr, B_imag_ptr,
    C_real_ptr, C_imag_ptr,
    D_real_ptr, D_imag_ptr,
    y_real_ptr, y_imag_ptr,  # Separate output pointers
    h_T_real_ptr, h_T_imag_ptr,
    # ... strides for all
)
```

2. **Complex Arithmetic Implementation**:
```python
# In kernel body - manual complex multiply
def cmul(ar, ai, br, bi):
    return ar*br - ai*bi, ar*bi + ai*br

def cexp(a, b):  # exp(a + ib)
    ea = tl.exp(a)
    return ea * tl.cos(b), ea * tl.sin(b)
```

3. **State Update Loop**:
```python
for t in range(L):
    # h = exp(-dt_A) * h + dt_B_x
    exp_re, exp_im = cexp(-dt_A_real, -dt_A_imag)
    h_re, h_im = cmul(exp_re, exp_im, h_re, h_im)
    
    # Add input term
    dt_B_x_re, dt_B_x_im = cmul(dt, 0, B_re, B_im)
    dt_B_x_re, dt_B_x_im = cmul(dt_B_x_re, dt_B_x_im, x_re, x_im)
    
    h_re = h_re + dt_B_x_re
    h_im = h_im + dt_B_x_im
    
    # Output: y = C * h
    y_re, y_im = cmul(C_re, C_im, h_re, h_im)
    y_out = tl.sum(y_re), tl.sum(y_im)  # Store both!
    
    tl.store(y_real_ptr + y_idx, y_out[0])
    tl.store(y_imag_ptr + y_idx, y_out[1])
```

### Phase 2: Stride Management (Medium Priority)

Every complex tensor now needs **2x stride tracking**:
- `stride_x_real_b, stride_x_imag_b`
- `stride_x_real_l, stride_x_imag_l`
- etc.

Or pack real/imag interleaved (more complex, better memory coalescing):
- `stride_x_packed = 2 * stride_x`
- Index as: `real_idx = 2*idx, imag_idx = 2*idx + 1`

**Recommendation**: Start with separate pointers (easier), optimize to interleaved later.

### Phase 3: Python Wrapper Update (Medium Priority)

```python
def complex_selective_scan_triton(
    x: torch.Tensor,  # complex64
    # ... other args
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Split complex into real/imag
    x_real = x.real.contiguous()
    x_imag = x.imag.contiguous()
    B_real, B_imag = B.real.contiguous(), B.imag.contiguous()
    C_real, C_imag = C.real.contiguous(), C.imag.contiguous()
    D_real, D_imag = D.real.contiguous(), D.imag.contiguous()
    h_0_real = h_0.real.contiguous() if h_0 is not None else None
    h_0_imag = h_0.imag.contiguous() if h_0 is not None else None
    
    # Allocate outputs
    y_real = torch.empty_like(x_real)
    y_imag = torch.empty_like(x_imag)
    h_T_real = torch.empty(B, d_inner, d_state, dtype=torch.float32, device=x.device)
    h_T_imag = torch.empty(B, d_inner, d_state, dtype=torch.float32, device=x.device)
    
    # Launch kernel with all pointers
    _complex_selective_scan_kernel[grid](
        x_real, x_imag, delta, 
        A_real, A_imag, B_real, B_imag, C_real, C_imag, D_real, D_imag,
        y_real, y_imag, h_T_real, h_T_imag,
        # ... all strides
    )
    
    # Recombine into complex
    y = torch.complex(y_real, y_imag)
    h_T = torch.complex(h_T_real, h_T_imag)
    
    return y, h_T.detach()
```

### Phase 4: Testing & Validation (High Priority)

1. **Unit Tests**: Compare Triton vs PyTorch CPU output
   ```python
   def test_triton_complex_ssm():
       x = torch.randn(B, L, d_inner, dtype=torch.complex64)
       # ... setup
       y_triton, h_triton = complex_selective_scan_triton(...)
       y_cuda, h_cuda = complex_selective_scan_cuda(...)
       assert torch.allclose(y_triton, y_cuda, rtol=1e-4, atol=1e-5)
       assert torch.allclose(h_triton, h_cuda, rtol=1e-4, atol=1e-5)
   ```

2. **Gradient Tests**: Verify autograd works
   ```python
   def test_triton_gradients():
       x.requires_grad_(True)
       y, _ = complex_selective_scan_triton(x, ...)
       loss = y.abs().sum()
       loss.backward()
       assert x.grad is not None
       assert torch.isfinite(x.grad).all()
   ```

3. **Performance Benchmarks**: Verify speedup claims
   ```python
   def benchmark_triton():
       # Vary L from 32 to 4096
       # Measure Triton vs CUDA PyTorch vs CPU
       # Assert Triton > 2x CUDA PyTorch for L > 128
   ```

---

## Estimated Timeline

| Phase | Complexity | Estimated Time |
|-------|------------|----------------|
| Phase 1: Data Structure Redesign | High | 2-3 days |
| Phase 2: Stride Management | Medium | 1-2 days |
| Phase 3: Python Wrapper | Medium | 1 day |
| Phase 4: Testing & Validation | High | 2-3 days |
| **Total** | | **6-9 days** |

---

## Alternative: PyTorch CUDA Operations (Current)

Until Triton is fixed, the `complex_selective_scan_cuda` function provides:
- ✅ Correct complex arithmetic
- ✅ 2-5x speedup over CPU Python loop
- ⚠️ Slower than Triton for long sequences (L > 256)

**Recommendation**: Ship with CUDA backend, fix Triton in Phase 2.

---

## References

- Triton Complex Number Discussion: https://github.com/openai/triton/issues/XXXX
- Complex SSM Theory: `docs/FBC_COMPLEX_SSM.md`
- CUDA PyTorch Backend: `src/bifrost/s1_decomposer/complex_ssm_triton.py:25`
