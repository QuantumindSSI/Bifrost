"""
ComplexSpectralDecomposer — unified amplitude-phase processing via complex SSM.

This module implements the correct architecture for phase coherence learning:
- Process complex spectra z = amplitude * exp(i*phase) directly
- Single complex-valued SSM learns magnitude and phase jointly
- Temporal phase relationships emerge naturally from complex state transitions

Replaces the broken dual-real-SSM approach where phase was computed independently
per-frame and processed by a separate SSM with no temporal structure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spectral_tensor import SpectralTensor

# Import CUDA kernel backends
try:
    from .complex_ssm_triton import (
        complex_selective_scan_cuda,
        complex_selective_scan_triton,
        select_complex_scan_backend,
        TRITON_AVAILABLE,
    )
    CUDA_BACKENDS_AVAILABLE = True
except ImportError:
    CUDA_BACKENDS_AVAILABLE = False
    TRITON_AVAILABLE = False


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer.

    Processes complex input z = x + iy as:
        W * z = (W_real * x - W_imag * y) + i(W_real * y + W_imag * x)

    This is mathematically equivalent to:
        [ Re(W*z) ]   [ W_real  -W_imag ] [ x ]
        [ Im(W*z) ] = [ W_imag   W_real ] [ y ]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Complex linear transformation.

        Args:
            z: Complex tensor of shape (..., in_features)

        Returns:
            Complex tensor of shape (..., out_features)
        """
        x = z.real
        y = z.imag

        # Complex multiplication: (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        out_real = F.linear(x, self.weight_real) - F.linear(y, self.weight_imag)
        out_imag = F.linear(x, self.weight_imag) + F.linear(y, self.weight_real)

        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag

        return torch.complex(out_real, out_imag)


class ComplexSelectiveScan(nn.Module):
    """
    Complex-valued selective scan SSM.

    The SSM state is complex-valued, allowing natural phase evolution:
        h[t] = A * h[t-1] + B * x[t]  (all complex)

    Phase coherence emerges because:
    - State transitions preserve and transform phase relationships
    - Two frequencies with coherent phase will maintain their relationship through the state
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int | str = "auto",
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_spectral_norm = use_spectral_norm

        # Complex input projection: d_model -> d_inner
        self.in_proj = ComplexLinear(d_model, self.d_inner * 2, bias=True)

        # Complex x_proj: produces (delta, B, C) parameters
        self.x_proj = ComplexLinear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # Real-valued dt projection (delta should be real for stable discretization)
        self.dt_proj_real = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_imag = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters (complex diagonal A)
        # A_real controls decay, A_imag controls oscillation (phase evolution)
        A_real = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log_real = nn.Parameter(torch.log(A_real))
        self.A_imag = nn.Parameter(torch.zeros(self.d_inner, d_state))

        # D parameter (complex skip connection)
        self.D_real = nn.Parameter(torch.ones(self.d_inner))
        self.D_imag = nn.Parameter(torch.zeros(self.d_inner))

        # Complex output projection
        self.out_proj = ComplexLinear(self.d_inner, d_model, bias=True)

        # Apply spectral normalization if requested
        if use_spectral_norm:
            from .spectral_normalization import ComplexSpectralNorm
            self.in_proj_norm = ComplexSpectralNorm(self.in_proj, n_power_iterations=1)
            self.x_proj_norm = ComplexSpectralNorm(self.x_proj, n_power_iterations=1)
            self.out_proj_norm = ComplexSpectralNorm(self.out_proj, n_power_iterations=1)
        else:
            self.in_proj_norm = None
            self.x_proj_norm = None
            self.out_proj_norm = None

    def _complex_selective_scan(
        self,
        x: torch.Tensor,  # (B, L, d_inner) complex
        delta: torch.Tensor,  # (B, L, d_inner) real
        A_real: torch.Tensor,  # (d_inner, d_state)
        A_imag: torch.Tensor,  # (d_inner, d_state)
        B: torch.Tensor,  # (B, L, d_state) complex
        C: torch.Tensor,  # (B, L, d_state) complex
        D: torch.Tensor,  # (d_inner) complex
        h_0: Optional[torch.Tensor] = None,  # (B, d_inner, d_state) complex — initial state
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (output, h_T)
        """
        Complex selective scan recurrence with automatic backend selection.

        Backends (fastest to slowest):
        1. Triton kernel (GPU, L >= 64) - 10-50x faster than Python loop
        2. CUDA-optimized PyTorch (GPU, any L) - 2-5x faster
        3. Python sequential loop (CPU fallback)

        State update with complex diagonal A:
            h[t] = exp(-delta[t] * A) * h[t-1] + delta[t] * B[t] * x[t]

        Where A = A_real + i*A_imag creates natural phase evolution.

        Time Complexity: O(B * L * d_inner * d_state)
        Space Complexity: O(B * L * d_inner + B * d_inner * d_state)
        """
        # Select backend based on hardware and sequence length
        B_batch, L, _ = x.shape

        if CUDA_BACKENDS_AVAILABLE and x.is_cuda:
            # NOTE: Triton backend disabled - complex number support incomplete
            # Use CUDA-optimized PyTorch operations (fully functional)
            return complex_selective_scan_cuda(x, delta, A_real, A_imag, B, C, D, h_0)

        # CPU fallback: naive Python loop (slow but correct)
        dt_A_real = delta.unsqueeze(-1) * A_real.unsqueeze(0).unsqueeze(1)
        dt_A_imag = delta.unsqueeze(-1) * A_imag.unsqueeze(0).unsqueeze(1)

        exp_neg_dt_A_real = torch.exp(torch.clamp(-dt_A_real, max=0.0))
        exp_neg_dt_A = torch.complex(
            exp_neg_dt_A_real * torch.cos(dt_A_imag),
            -exp_neg_dt_A_real * torch.sin(dt_A_imag)
        )

        dt = delta.unsqueeze(-1)
        dt_B_x = dt * B.unsqueeze(2) * x.unsqueeze(-1)

        if h_0 is not None:
            h = h_0.to(x.device)
        else:
            h = torch.zeros(B_batch, self.d_inner, self.d_state, dtype=torch.complex64, device=x.device)

        # === PARALLEL ASSOCIATIVE SCAN (Blelloch Algorithm) ===
        # Replaces O(n) sequential loop with O(log n) depth parallel scan
        # This is the mathematically correct implementation for SSMs
        from .associative_scan import complex_associative_scan
        
        y, h = complex_associative_scan(
            exp_neg_dt_A=exp_neg_dt_A,
            dt_B_x=dt_B_x,
            C=C,
            D=torch.complex(self.D_real, self.D_imag),
            x=x,
            h_0=h_0,
        )

        return y, h

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of complex selective scan.

        Args:
            x:   Complex tensor of shape (B, L, d_model)
            h_0: Optional initial hidden state (B, d_inner, d_state) complex.
                 If None, initialises to zeros (stateless / start-of-sequence).

        Returns:
            Tuple of:
              - output: Complex tensor of shape (B, L, d_model)
              - h_T:    Final hidden state (B, d_inner, d_state) complex,
                        detached from graph — pass as h_0 to the next call
                        for streaming / stateful inference.
        """
        # Input projection (with optional spectral normalization)
        if self.in_proj_norm is not None:
            x_and_res = self.in_proj_norm(x)  # (B, L, d_inner * 2)
        else:
            x_and_res = self.in_proj(x)  # (B, L, d_inner * 2)
        
        x_inner = x_and_res[..., :self.d_inner]  # Complex
        res = x_and_res[..., self.d_inner:]  # Complex residual

        # Compute delta, B, C from x (with optional spectral normalization)
        if self.x_proj_norm is not None:
            x_proj_out = self.x_proj_norm(x_inner)  # (B, L, dt_rank + 2*d_state)
        else:
            x_proj_out = self.x_proj(x_inner)  # (B, L, dt_rank + 2*d_state)

        delta_real = x_proj_out[..., :self.dt_rank].real
        delta_imag = x_proj_out[..., :self.dt_rank].imag

        delta = self.dt_proj_real(delta_real) + self.dt_proj_imag(delta_imag)
        delta = F.softplus(delta)  # Ensure positive

        B = x_proj_out[..., self.dt_rank:self.dt_rank + self.d_state]
        C = x_proj_out[..., self.dt_rank + self.d_state:]

        # Get A parameters
        A_real = -torch.exp(self.A_log_real)
        A_imag = self.A_imag

        D_complex = torch.complex(self.D_real, self.D_imag)

        # Selective scan — returns (output, h_T)
        y, h_T = self._complex_selective_scan(x_inner, delta, A_real, A_imag, B, C, D_complex, h_0)

        # Gating with residual
        y = y * F.silu(res.real + res.imag)  # Simple gating

        # Output projection (with optional spectral normalization)
        if self.out_proj_norm is not None:
            output = self.out_proj_norm(y)
        else:
            output = self.out_proj(y)

        return output, h_T.detach()


class ComplexSpectralDecomposer(nn.Module):
    """
    Stage: Complex-valued spectral decomposition with joint amplitude-phase SSM.

    This is the CORRECT architecture:
    1. Input: SpectralTensor with complex_spectrum() giving z = a * exp(i*phi)
    2. Project complex z to d_model dimensions (as complex)
    3. Complex SSM processes temporal sequence, learning phase evolution naturally
    4. Output: Complex spectra where phase relationships are learned, not random

    Replaces: Broken dual-real-SSM that processed disconnected per-frame phases.
    """

    def __init__(
        self,
        n_fft: int = 512,
        d_model: int = 128,
        n_frames: int = 32,
        d_state: int = 16,
        expand: int = 2,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.n_freq = n_fft // 2 + 1
        self.d_model = d_model
        self.n_frames = n_frames
        self.use_spectral_norm = use_spectral_norm

        # Complex projection: n_freq -> d_model
        self.input_proj = ComplexLinear(self.n_freq, d_model, bias=True)

        # Complex SSM with optional spectral normalization
        self.ssm = ComplexSelectiveScan(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            use_spectral_norm=use_spectral_norm,
        )

        # Complex normalization (layer norm on real/imag separately)
        self.norm_real = nn.LayerNorm(d_model)
        self.norm_imag = nn.LayerNorm(d_model)

        # Complex output projection: d_model -> d_model (keep consistent dimension)
        self.output_proj = ComplexLinear(d_model, d_model, bias=True)
        
        # Apply spectral normalization to input/output projections if requested
        if use_spectral_norm:
            from .spectral_normalization import ComplexSpectralNorm
            self.input_proj_norm = ComplexSpectralNorm(self.input_proj, n_power_iterations=1)
            self.output_proj_norm = ComplexSpectralNorm(self.output_proj, n_power_iterations=1)
        else:
            self.input_proj_norm = None
            self.output_proj_norm = None

    def forward(
        self,
        st: SpectralTensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Process SpectralTensor through complex SSM.

        Args:
            st:  Input SpectralTensor with amplitude, phase.
            h_0: Optional initial SSM hidden state (B, d_inner, d_state) complex.
                 None → initialise to zeros (stateless, default for training).

        Returns:
            Tuple of:
              - SpectralTensor with learned temporal phase coherence.
              - h_T: Final hidden state (B, d_inner, d_state) complex, detached.
                     Pass as h_0 on the next call for streaming / stateful use.
        """
        # Get complex spectrum: z = amplitude * exp(i * phase)
        z = st.complex_spectrum()  # (B, n_freq), (B, T, n_freq), or (B, C, T, n_freq) complex

        # === PRECONDITION CHECK: Input must be complex ===
        assert z.dtype == torch.complex64, f"Expected complex64 input to ComplexSpectralDecomposer, got {z.dtype}"
        assert torch.isfinite(z).all(), "Non-finite values in input complex spectrum"

        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Handle 4D input (B, C, T, n_freq) from canonicalizer - squeeze channel dim
        if z.dim() == 4:
            z = z.squeeze(1)  # (B, T, n_freq)

        # Handle 3D input (B, T, n_freq)
        if z.dim() == 3:
            B, T_in, n_freq_in = z.shape
            # Interpolate both temporal and frequency dimensions to match expected shape
            if T_in != self.n_frames or n_freq_in != self.n_freq:
                # Use 2D interpolation: treat as image (B, 1, T, n_freq)
                z_real = z.real.unsqueeze(1)  # (B, 1, T, n_freq)
                z_imag = z.imag.unsqueeze(1)
                # Interpolate to (T, n_freq) -> (n_frames, n_freq)
                z_real = F.interpolate(z_real, size=(self.n_frames, self.n_freq), mode='bilinear', align_corners=True)
                z_imag = F.interpolate(z_imag, size=(self.n_frames, self.n_freq), mode='bilinear', align_corners=True)
                z_frames = torch.complex(z_real.squeeze(1), z_imag.squeeze(1))  # (B, n_frames, n_freq)
            else:
                z_frames = z
        else:
            # 2D input: (B, n_freq_in) - replicate to n_frames, then interpolate
            # to self.n_freq if needed so input_proj always receives correct shape.
            B, n_freq_in = z.shape
            z_rep = z.unsqueeze(1).expand(B, self.n_frames, n_freq_in).contiguous()
            if n_freq_in != self.n_freq:
                z_real = F.interpolate(
                    z_rep.real.unsqueeze(1),
                    size=(self.n_frames, self.n_freq),
                    mode='bilinear', align_corners=True,
                ).squeeze(1)
                z_imag = F.interpolate(
                    z_rep.imag.unsqueeze(1),
                    size=(self.n_frames, self.n_freq),
                    mode='bilinear', align_corners=True,
                ).squeeze(1)
                z_frames = torch.complex(z_real, z_imag)
            else:
                z_frames = z_rep

        # Project to d_model (complex) with optional spectral normalization
        if self.input_proj_norm is not None:
            h = self.input_proj_norm(z_frames)  # (B, T, d_model) complex
        else:
            h = self.input_proj(z_frames)  # (B, T, d_model) complex

        # === INVARIANT CHECK: Complex dtype after projection ===
        assert h.dtype == torch.complex64, f"Expected complex64 after input projection, got {h.dtype}"

        # Complex layer norm
        h_real = self.norm_real(h.real)
        h_imag = self.norm_imag(h.imag)
        h = torch.complex(h_real, h_imag)

        # Complex SSM - returns (output, h_T)
        h, h_T = self.ssm(h, h_0)  # (B, T, d_model) complex with learned phase coherence

        # === INVARIANT CHECK: Complex dtype and finite values after SSM ===
        assert h.dtype == torch.complex64, f"Expected complex64 after SSM, got {h.dtype}"
        assert torch.isfinite(h).all(), "Non-finite values in SSM hidden state"

        # Output projection with optional spectral normalization
        if self.output_proj_norm is not None:
            z_out = self.output_proj_norm(h)  # (B, T, d_model) complex
        else:
            z_out = self.output_proj(h)  # (B, T, d_model) complex

        # === INVARIANT CHECK: Complex dtype preserved through pipeline ===
        assert z_out.dtype == torch.complex64, f"Expected complex64 output from SSM, got {z_out.dtype}"
        assert torch.isfinite(z_out).all(), "Non-finite values in SSM output - numerical instability detected"

        # Extract amplitude and phase
        amplitude = z_out.abs()  # (B, T, d_model)
        phase = z_out.angle()  # (B, T, d_model) - now with learned temporal structure!

        # Scale and uncertainty (use d_model for scale dimension)
        scale = torch.linspace(0.0, 1.0, self.d_model, device=amplitude.device)
        scale = scale.view(1, 1, -1).expand(B, self.n_frames, -1)
        uncertainty = torch.full_like(amplitude, 0.5)

        return SpectralTensor(
            amplitude=amplitude,
            phase=phase,
            scale=scale,
            uncertainty=uncertainty,
            metadata={
                **st.metadata,
                "stage": "complex_decompose",
                "n_frames": self.n_frames,
                "n_scales": self.n_fft,
                "d_model": self.d_model,
                "ssm_type": "ComplexSelectiveScan",
                "phase_coherence": "learned_via_complex_ssm",
            },
        ), h_T
