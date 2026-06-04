# Cloud GPU Setup Guide

Quick deployment guide for running Bifrost on cloud GPU instances (RunPod, Lambda Labs, etc.).

## Hardware Requirements

- **GPU**: NVIDIA H200, H100, A100, or RTX 4090 (sm_90/sm_89/sm_80)
- **CUDA**: 12.4 or higher
- **Memory**: 40GB+ VRAM recommended for large models
- **Storage**: 20GB+ for dependencies and checkpoints

## Quick Start (RunPod)

### 1. Create Pod
- Template: PyTorch 2.3 + CUDA 12.8
- GPU: NVIDIA H200 (or H100/A100)
- Container Disk: 20GB
- Volume Disk: 50GB (for data)

### 2. SSH into Pod
```bash
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/<your-key>
```

### 3. Clone Repository
```bash
cd /workspace
git clone https://github.com/QuantumindSSI/Bifrost.git bifrost
cd bifrost
```

### 4. Run Setup Script
```bash
chmod +x scripts/setup_mamba_gpu.sh
./scripts/setup_mamba_gpu.sh
```

This installs:
- PyTorch with CUDA 12.8
- causal-conv1d (Mamba dependency)
- mamba-ssm with CUDA kernels
- Bifrost dependencies

### 5. Verify Installation
```bash
python3 scripts/verify_mamba_gpu.py
```

Expected output:
```
GPU 0: NVIDIA H200
  Compute: 9.0 (sm_90)
  Memory: 141.0 GB

Mamba-SSM: 2.2.4
Mamba-SSM GPU installation successful!

FBC Mamba-3 integration: SUCCESS
```

## Manual Installation (if script fails)

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Mamba dependencies
pip install causal-conv1d>=1.4.0 --no-build-isolation

# Install Mamba-SSM (build from source for H200)
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/usr/local/cuda
pip install mamba-ssm>=2.2.4 --no-build-isolation

# Install FBC
pip install -e .
```

## Testing Mamba-3 in FBC

```python
import torch
from fbc.pipeline import FBCPipeline

# Create pipeline with Mamba-3
pipeline = FBCPipeline(
    n_fft_s0=1024,
    n_fft_s1=512,
    d_model=256,
    n_scales=6,
    n_heads=8,
    use_mamba=True,  # Enable Mamba-3 SSM
).cuda()

# Process audio
signal = torch.randn(1, 16000).cuda()
bound_st, coherence = pipeline(signal, {"sample_rate": 16000})

print(f"Output: {bound_st.amplitude.shape}")
print(f"Mamba-3 active: {pipeline.s1.use_mamba}")
```

## Performance Benchmarks

| GPU | Batch | Seq Length | Time/iter | Tokens/sec |
|-----|-------|-----------|-----------|------------|
| H200 | 4 | 512 | ~2.1ms | ~975K |
| H200 | 8 | 1024 | ~3.8ms | ~2.1M |
| H100 | 4 | 512 | ~2.3ms | ~890K |
| A100 | 4 | 512 | ~2.8ms | ~730K |

## Troubleshooting

### Build fails with "no kernel image"
Set correct CUDA arch:
```bash
export TORCH_CUDA_ARCH_LIST="9.0"  # H200/H100
# or
export TORCH_CUDA_ARCH_LIST="8.0"  # A100
# or
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090
```

### ImportError: libcudart.so.12
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Out of memory
Reduce batch size or d_model:
```python
pipeline = FBCPipeline(d_model=128, n_heads=4)  # Smaller model
```

## Docker Alternative

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git ninja-build

WORKDIR /workspace
COPY . /workspace/fbc-core
RUN pip install mamba-ssm causal-conv1d
RUN pip install -e /workspace/fbc-core
```

## Next Steps

- [SKG Persistence Setup](./SKG_PERSISTENCE.md)
- [Multi-GPU Training](./MULTI_GPU.md)
- [Cross-Modal Retrieval](../examples/cross_modal_retrieval.py)
