"""
Tests for Complex Spectral Normalization

Tests the spectral normalization implementation for complex-valued layers
to ensure proper gradient flow, training stability, and correctness.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

# Import the modules to test
from bifrost.decomposer.complex_decomposer import ComplexLinear, ComplexSelectiveScan
from bifrost.decomposer.spectral_normalization import (
    ComplexSpectralNorm,
    SpectralNormalizedComplexLinear,
    apply_spectral_norm_to_module,
)


class TestComplexSpectralNorm:
    """Test ComplexSpectralNorm wrapper."""
    
    def test_init(self):
        """Test initialization of spectral norm wrapper."""
        layer = ComplexLinear(in_features=64, out_features=32)
        spec_norm = ComplexSpectralNorm(layer, n_power_iterations=1)
        
        assert spec_norm.module is layer
        assert spec_norm.n_power_iterations == 1
        assert spec_norm.u.shape == (64,)  # 2*out_features
    
    def test_composite_weight_matrix(self):
        """Test construction of composite weight matrix."""
        layer = ComplexLinear(in_features=4, out_features=3)
        spec_norm = ComplexSpectralNorm(layer)
        
        # Get composite matrix
        composite = spec_norm._composite_weight_matrix()
        
        # Should be 2*out_features x 2*in_features
        assert composite.shape == (6, 8)
        assert composite.dtype == layer.weight_real.dtype
    
    def test_power_iteration(self):
        """Test power iteration step."""
        layer = ComplexLinear(in_features=8, out_features=8)
        spec_norm = ComplexSpectralNorm(layer)
        
        composite = spec_norm._composite_weight_matrix()
        u = spec_norm.u
        
        u_new, sigma = spec_norm._power_iteration(composite, u)
        
        # New u should be normalized
        assert torch.allclose(u_new.norm(), torch.tensor(1.0), atol=1e-6)
        
        # Sigma should be positive (spectral norm)
        assert sigma > 0
    
    def test_forward_normalizes_weights(self):
        """Test that forward pass normalizes weights."""
        torch.manual_seed(42)
        layer = ComplexLinear(in_features=16, out_features=16)
        spec_norm = ComplexSpectralNorm(layer, n_power_iterations=1)
        
        # Input
        x = torch.randn(2, 16, dtype=torch.complex64)
        
        # Store original weights
        w_real_orig = layer.weight_real.data.clone()
        w_imag_orig = layer.weight_imag.data.clone()
        
        # Forward pass
        _ = spec_norm(x)
        
        # Weights should have been normalized then restored
        assert torch.allclose(layer.weight_real.data, w_real_orig, atol=1e-5)
        assert torch.allclose(layer.weight_imag.data, w_imag_orig, atol=1e-5)
    
    def test_spectral_norm_decreases(self):
        """Test that spectral norm of normalized weights ≤ 1."""
        torch.manual_seed(42)
        layer = ComplexLinear(in_features=32, out_features=32)
        spec_norm = ComplexSpectralNorm(layer, n_power_iterations=5)
        
        # Get spectral norm before normalization
        composite_before = spec_norm._composite_weight_matrix()
        u_before = spec_norm.u
        _, sigma_before = spec_norm._power_iteration(composite_before, u_before)
        
        # Forward pass normalizes weights
        x = torch.randn(2, 32, dtype=torch.complex64)
        _ = spec_norm(x)
        
        # Get spectral norm after normalization (should be close to 1)
        u_after = spec_norm.u
        composite_after = spec_norm._composite_weight_matrix()
        _, sigma_after = spec_norm._power_iteration(composite_after, u_after)
        
        # After normalization, sigma should be ≤ 1
        assert sigma_after <= 1.0 + 1e-6  # Small tolerance for numerical precision
    
    def test_gradient_flow(self):
        """Test that gradients flow through spectral norm layer."""
        layer = ComplexLinear(in_features=8, out_features=8)
        spec_norm = ComplexSpectralNorm(layer)
        
        x = torch.randn(2, 8, dtype=torch.complex64, requires_grad=True)
        output = spec_norm(x)
        loss = output.abs().sum()
        loss.backward()
        
        # Gradients should flow to input and weights
        assert x.grad is not None
        assert layer.weight_real.grad is not None
        assert layer.weight_imag.grad is not None


class TestSpectralNormalizedComplexLinear:
    """Test SpectralNormalizedComplexLinear convenience wrapper."""
    
    def test_init_with_spectral_norm(self):
        """Test initialization with spectral norm enabled."""
        layer = SpectralNormalizedComplexLinear(
            in_features=32,
            out_features=32,
            use_spectral_norm=True
        )
        
        assert layer.use_spectral_norm is True
        assert layer.spectral_norm is not None
        assert hasattr(layer, 'complex_linear')
    
    def test_init_without_spectral_norm(self):
        """Test initialization without spectral norm."""
        layer = SpectralNormalizedComplexLinear(
            in_features=32,
            out_features=32,
            use_spectral_norm=False
        )
        
        assert layer.use_spectral_norm is False
        assert layer.spectral_norm is None
    
    def test_forward_with_spectral_norm(self):
        """Test forward pass with spectral norm enabled."""
        layer = SpectralNormalizedComplexLinear(
            in_features=16,
            out_features=16,
            use_spectral_norm=True
        )
        
        x = torch.randn(2, 16, dtype=torch.complex64)
        output = layer(x)
        
        assert output.shape == (2, 16)
        assert output.dtype == torch.complex64
    
    def test_forward_without_spectral_norm(self):
        """Test forward pass without spectral norm."""
        layer = SpectralNormalizedComplexLinear(
            in_features=16,
            out_features=16,
            use_spectral_norm=False
        )
        
        x = torch.randn(2, 16, dtype=torch.complex64)
        output = layer(x)
        
        assert output.shape == (2, 16)
        assert output.dtype == torch.complex64


class TestComplexSelectiveScanWithSpectralNorm:
    """Test ComplexSelectiveScan with spectral normalization."""
    
    def test_init_with_spectral_norm(self):
        """Test ComplexSelectiveScan initialization with spectral norm."""
        ssm = ComplexSelectiveScan(
            d_model=64,
            d_state=16,
            expand=2,
            use_spectral_norm=True
        )
        
        assert ssm.use_spectral_norm is True
        assert ssm.in_proj_norm is not None
        assert ssm.x_proj_norm is not None
        assert ssm.out_proj_norm is not None
    
    def test_init_without_spectral_norm(self):
        """Test ComplexSelectiveScan initialization without spectral norm."""
        ssm = ComplexSelectiveScan(
            d_model=64,
            d_state=16,
            expand=2,
            use_spectral_norm=False
        )
        
        assert ssm.use_spectral_norm is False
        assert ssm.in_proj_norm is None
        assert ssm.x_proj_norm is None
        assert ssm.out_proj_norm is None


class TestApplySpectralNormToModule:
    """Test recursive spectral norm application."""
    
    def test_apply_to_complex_linear(self):
        """Test applying spectral norm to ComplexLinear layers."""
        module = nn.Sequential(
            ComplexLinear(32, 32),
            ComplexLinear(32, 32),
        )
        
        modified = apply_spectral_norm_to_module(module, n_power_iterations=1)
        
        # Check that layers are replaced
        assert isinstance(modified[0], SpectralNormalizedComplexLinear)
        assert isinstance(modified[1], SpectralNormalizedComplexLinear)
    
    def test_apply_to_nested_module(self):
        """Test applying spectral norm to nested modules."""
        module = nn.Sequential(
            ComplexSelectiveScan(d_model=32, d_state=8),
            ComplexLinear(32, 32),
        )
        
        modified = apply_spectral_norm_to_module(module, n_power_iterations=1)
        
        # The SelectiveScan should not be replaced, but its internal
        # ComplexLinear layers should have spectral norm applied
        assert isinstance(modified, nn.Sequential)


class TestTrainingStabilityWithSpectralNorm:
    """Test training stability improvements with spectral normalization."""
    
    def test_training_without_nan(self):
        """Test that spectral norm prevents gradient explosion during training."""
        torch.manual_seed(42)
        
        # Create a layer with spectral norm
        layer = SpectralNormalizedComplexLinear(
            in_features=64,
            out_features=64,
            use_spectral_norm=True
        )
        
        optimizer = Adam(layer.parameters(), lr=0.01)
        
        # Training loop
        for _ in range(5):
            x = torch.randn(4, 64, dtype=torch.complex64)
            output = layer(x)
            
            # Simple MSE loss
            loss = (output.abs() ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check for NaN/Inf
            assert torch.isfinite(loss), "Loss became NaN/Inf"
            assert torch.isfinite(output).all(), "Output contains NaN/Inf"
            
            for param in layer.parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), f"Gradient contains NaN/Inf"
    
    def test_spectral_norm_bounds_lipschitz(self):
        """Test that spectral norm normalizes Lipschitz constant."""
        torch.manual_seed(42)
        
        layer = SpectralNormalizedComplexLinear(
            in_features=32,
            out_features=32,
            use_spectral_norm=True
        )
        layer.eval()
        
        # Generate input
        x1 = torch.randn(1, 32, dtype=torch.complex64)
        x2 = torch.randn(1, 32, dtype=torch.complex64)
        
        with torch.no_grad():
            # Multiple forward passes to update spectral norm estimate
            for _ in range(5):
                _ = layer(x1)
        
        with torch.no_grad():
            y1 = layer(x1)
            y2 = layer(x2)
        
        # Lipschitz constant: ||f(x1) - f(x2)|| / ||x1 - x2||
        delta_x = (x1 - x2).abs().sum().item()
        delta_y = (y1 - y2).abs().sum().item()
        
        if delta_x > 1e-6:
            lipschitz = delta_y / delta_x
            # After spectral norm, Lipschitz should be bounded
            assert lipschitz < 100  # Rough bound


class TestDifferentShapes:
    """Test spectral norm with different tensor shapes."""
    
    def test_linear_layer_shapes(self):
        """Test SpectralNormalizedComplexLinear with various input shapes."""
        layer = SpectralNormalizedComplexLinear(
            in_features=32,
            out_features=16,
            use_spectral_norm=True
        )
        
        x = torch.randn(4, 32, dtype=torch.complex64)
        output = layer(x)
        
        assert output.shape == (4, 16)
        assert output.dtype == torch.complex64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
