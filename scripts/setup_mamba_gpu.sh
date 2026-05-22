#!/bin/bash
# Mamba-SSM GPU Setup Script for RunPod / Cloud GPU Instances
# NVIDIA H200 / H100 / A100 / RTX 4090 compatible

set -e

echo "=== Mamba-SSM GPU Setup ==="
echo "GPU Detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "CUDA Version: $CUDA_VERSION"

# Install system dependencies
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    git \
    build-essential \
    ninja-build \
    python3-dev \
    python3-pip \
    libaio-dev \
    cuda-toolkit-12-8 \
    2>/dev/null || true

# Upgrade pip and install build tools
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel ninja

# Install PyTorch with CUDA support (matches CUDA 12.8)
echo "Installing PyTorch with CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install causal-conv1d (mamba-ssm dependency)
echo "Installing causal-conv1d..."
pip install causal-conv1d>=1.4.0

# Install mamba-ssm with CUDA extensions
echo "Installing mamba-ssm..."
# Set CUDA architecture for H200 (sm_90)
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/usr/local/cuda

# Try pip install first (pre-built wheels)
pip install mamba-ssm>=2.2.4 || {
    echo "Pre-built wheel not available, building from source..."
    pip install git+https://github.com/state-spaces/mamba.git@main
}

# Install additional dependencies
echo "Installing additional FBC dependencies..."
pip install \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    PyWavelets>=1.5.0 \
    einops>=0.7.0 \
    triton>=2.3.0

# Verify installation
echo ""
echo "=== Verifying Mamba-SSM Installation ==="
python3 -c "
import torch
import mamba_ssm
from mamba_ssm import Mamba

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Mamba-SSM: {mamba_ssm.__version__}')

# Test Mamba layer on GPU
if torch.cuda.is_available():
    device = 'cuda'
    batch, seq, dim = 2, 64, 128
    x = torch.randn(batch, seq, dim).to(device)
    model = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2).to(device)
    y = model(x)
    print(f'Test forward pass: {x.shape} -> {y.shape}')
    print('Mamba-SSM GPU installation successful!')
else:
    print('WARNING: CUDA not available, using CPU fallback')
"

echo ""
echo "=== Setup Complete ==="
echo "To activate in FBC: FBCPipeline(use_mamba=True)"
