"""
Spectral Normalization for Complex-Valued Neural Networks

Implements spectral normalization (constraining the Lipschitz constant of weights)
for complex-valued layers. This stabilizes training dynamics by normalizing weights
to have spectral norm (largest singular value) ≤ 1.

Theory:
    For complex weight matrices W, the spectral norm is the largest singular value σ_max(W).
    We normalize W ← W / σ_max(W) to constrain the Lipschitz constant.

    For complex linear layers with separate real and imaginary parts:
        W = W_real + i * W_imag
    We compute the spectral norm via power iteration on the composite matrix:
        M = [W_real   -W_imag]
            [W_imag    W_real]

References:
    - Spectral Normalization for GAN (Miyato et al., ICLR 2018)
    - Complex-Valued Neural Networks (Hirose & Yoshida, 2012)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ComplexSpectralNorm(nn.Module):
    """
    Spectral normalization wrapper for complex-valued weight matrices.
    
    Normalizes the spectral norm (largest singular value) of complex weight
    matrices to be ≤ 1, constraining the Lipschitz constant of the layer.
    
    This improves training stability for complex-valued networks by preventing
    gradient explosion and controlling the dynamics of complex state transitions.
    
    Parameters
    ----------
    module : nn.Module
        Module containing the weight to normalize (e.g., ComplexLinear).
    weight_name : str
        Name of the weight parameter(s) to normalize. For complex weights,
        this should refer to the real part (e.g., 'weight_real').
    imag_weight_name : str
        Name of the imaginary weight parameter. Default: 'weight_imag'.
    n_power_iterations : int
        Number of power iterations per forward pass. Default: 1.
        Higher values are more accurate but slower.
    eps : float
        Small constant for numerical stability. Default: 1e-12.
    
    Attributes
    ----------
    u : nn.Parameter
        Left singular vector (normalized). Shape: (out_features,).
    """
    
    def __init__(
        self,
        module: nn.Module,
        weight_name: str = 'weight_real',
        imag_weight_name: str = 'weight_imag',
        n_power_iterations: int = 1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.module = module
        self.weight_name = weight_name
        self.imag_weight_name = imag_weight_name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
        # Get weight tensors
        weight_real = getattr(module, weight_name)
        weight_imag = getattr(module, imag_weight_name)
        
        # out_features x in_features for both real and imaginary parts
        out_features, in_features = weight_real.shape
        
        # Register buffer for left singular vector (u)
        # For composite matrix of shape (2*out, 2*in), u should have size 2*out
        u = torch.randn(2 * out_features, dtype=weight_real.dtype)
        u = u / u.norm()
        self.register_buffer('u', u)
        
        # Track number of times forward has been called (for periodic spectral norm updates)
        self.register_buffer('n_calls', torch.tensor(0, dtype=torch.long))
    
    def _composite_weight_matrix(self) -> torch.Tensor:
        """
        Construct composite weight matrix from real and imaginary parts.
        
        For complex weight W = W_real + i*W_imag, the composite matrix is:
            M = [W_real   -W_imag]
                [W_imag    W_real]
        
        This representation preserves complex multiplication semantics:
        M * [x; y] = [W_real*x - W_imag*y; W_imag*x + W_real*y]
        
        Returns
        -------
        torch.Tensor
            Composite real-valued matrix of shape (2*out, 2*in).
        """
        weight_real = getattr(self.module, self.weight_name)
        weight_imag = getattr(self.module, self.imag_weight_name)
        
        out_features, in_features = weight_real.shape
        
        # Construct block matrix: [W_real, -W_imag; W_imag, W_real]
        # Top half: [W_real | -W_imag]
        top = torch.cat([weight_real, -weight_imag], dim=1)  # (out, 2*in)
        # Bottom half: [W_imag | W_real]
        bottom = torch.cat([weight_imag, weight_real], dim=1)  # (out, 2*in)
        # Full composite: (2*out, 2*in)
        composite = torch.cat([top, bottom], dim=0)
        
        return composite
    
    def _power_iteration(
        self,
        weight_composite: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Perform one power iteration step.
        
        Computes v = W^T u / ||W^T u||, then u = W v / ||W v||.
        The spectral norm σ ≈ ||W v||.
        
        Parameters
        ----------
        weight_composite : torch.Tensor
            Composite weight matrix (2*out_features x in_features).
        u : torch.Tensor
            Current left singular vector estimate.
        
        Returns
        -------
        u_new : torch.Tensor
            Updated left singular vector.
        sigma : float
            Estimated spectral norm.
        """
        # v = W^T u / ||W^T u||
        v = weight_composite.T @ u
        v = v / (v.norm() + self.eps)
        
        # u = W v / ||W v||
        u_new = weight_composite @ v
        sigma = u_new.norm()
        u_new = u_new / (sigma + self.eps)
        
        return u_new, sigma.item()
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with spectral normalization applied.
        
        Performs power iterations to estimate spectral norm, then normalizes
        weights accordingly. Periodically updates the stored singular vector.
        """
        # Perform power iterations to estimate spectral norm
        weight_composite = self._composite_weight_matrix()
        u = self.u
        
        for _ in range(self.n_power_iterations):
            u, sigma = self._power_iteration(weight_composite, u)
        
        # Update stored u (only if training to avoid modifying inference behavior)
        if self.training:
            self.u.copy_(u)
        
        # Normalize weights by spectral norm
        weight_real = getattr(self.module, self.weight_name)
        weight_imag = getattr(self.module, self.imag_weight_name)
        
        # Scale both real and imaginary parts by 1/sigma
        scale = 1.0 / (sigma + self.eps)
        
        # Directly scale the weights
        weight_real_scaled = weight_real * scale
        weight_imag_scaled = weight_imag * scale
        
        # Temporarily replace weights with normalized versions
        weight_real_orig = weight_real.data.clone()
        weight_imag_orig = weight_imag.data.clone()
        
        weight_real.data.copy_(weight_real_scaled)
        weight_imag.data.copy_(weight_imag_scaled)
        
        try:
            # Run forward pass with normalized weights
            output = self.module(*args, **kwargs)
        finally:
            # Always restore original weights
            weight_real.data.copy_(weight_real_orig)
            weight_imag.data.copy_(weight_imag_orig)
        
        return output
    
    def remove(self) -> None:
        """Remove spectral normalization from the module."""
        # This is a no-op for the wrapper approach; real removal would require
        # modifying the module itself. This is kept for API compatibility.
        pass


class SpectralNormalizedComplexLinear(nn.Module):
    """
    Complex-valued linear layer with built-in spectral normalization.
    
    This is a convenience wrapper that combines ComplexLinear with
    automatic spectral normalization of both real and imaginary weights.
    
    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool
        Whether to use bias terms. Default: True.
    use_spectral_norm : bool
        Whether to apply spectral normalization. Default: True.
    n_power_iterations : int
        Number of power iterations for spectral norm. Default: 1.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_spectral_norm: bool = True,
        n_power_iterations: int = 1,
    ) -> None:
        super().__init__()
        
        # Import here to avoid circular imports
        from .complex_decomposer import ComplexLinear
        
        self.complex_linear = ComplexLinear(in_features, out_features, bias=bias)
        self.use_spectral_norm = use_spectral_norm
        
        if use_spectral_norm:
            self.spectral_norm = ComplexSpectralNorm(
                self.complex_linear,
                weight_name='weight_real',
                imag_weight_name='weight_imag',
                n_power_iterations=n_power_iterations,
            )
        else:
            self.spectral_norm = None
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complex linear layer.
        
        Parameters
        ----------
        z : torch.Tensor
            Complex input tensor of shape (..., in_features).
        
        Returns
        -------
        torch.Tensor
            Complex output of shape (..., out_features).
        """
        if self.spectral_norm is not None:
            return self.spectral_norm(z)
        else:
            return self.complex_linear(z)


def apply_spectral_norm_to_module(
    module: nn.Module,
    n_power_iterations: int = 1,
) -> nn.Module:
    """
    Recursively apply spectral normalization to all ComplexLinear layers.
    
    Parameters
    ----------
    module : nn.Module
        Module tree to apply normalization to.
    n_power_iterations : int
        Power iterations for spectral norm computation.
    
    Returns
    -------
    nn.Module
        Modified module with spectral normalization applied.
    """
    from .complex_decomposer import ComplexLinear
    
    for name, child in module.named_children():
        if isinstance(child, ComplexLinear):
            # Replace ComplexLinear with SpectralNormalizedComplexLinear
            setattr(module, name, SpectralNormalizedComplexLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias_real is not None,
                use_spectral_norm=True,
                n_power_iterations=n_power_iterations,
            ))
        else:
            # Recurse
            apply_spectral_norm_to_module(child, n_power_iterations)
    
    return module
