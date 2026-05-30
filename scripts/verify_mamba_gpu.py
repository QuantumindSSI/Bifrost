#!/usr/bin/env python3
"""
Mamba-SSM GPU Verification Script for FBC
Tests Mamba-3 integration on cloud GPU instances.
"""

import sys
import torch
import time

def check_cuda():
    """Check CUDA availability and GPU info."""
    print("=" * 60)
    print("CUDA / GPU Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"CUDNN: {torch.backends.cudnn.version()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute: {props.major}.{props.minor} (sm_{props.major}{props.minor})")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")

    return True


def check_mamba_ssm():
    """Check mamba-ssm installation."""
    print("\n" + "=" * 60)
    print("Mamba-SSM Check")
    print("=" * 60)

    try:
        import mamba_ssm
        from mamba_ssm import Mamba
        print(f"Mamba-SSM: {mamba_ssm.__version__}")
        return True
    except ImportError as e:
        print(f"ERROR: mamba-ssm not installed: {e}")
        return False


def test_mamba_forward():
    """Test Mamba forward pass on GPU."""
    print("\n" + "=" * 60)
    print("Mamba Forward Pass Test")
    print("=" * 60)

    from mamba_ssm import Mamba

    device = torch.device("cuda")
    batch, seq_len, d_model = 4, 512, 256

    # Create model
    model = Mamba(
        d_model=d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        device=device,
        dtype=torch.float32,
    )

    # Test data
    x = torch.randn(batch, seq_len, d_model, device=device)

    # Warmup
    for _ in range(3):
        _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        y = model(x)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Time per iteration: {elapsed/iterations*1000:.2f} ms")
    print(f"Throughput: {batch * seq_len * iterations / elapsed:.0f} tokens/sec")

    return True


def test_fbc_mamba_integration():
    """Test FBC S1 with Mamba-3 enabled."""
    print("\n" + "=" * 60)
    print("FBC Mamba-3 Integration Test")
    print("=" * 60)

    sys.path.insert(0, "/workspace/fbc-core/src")

    from bifrost.pipeline import BifrostPipeline

    # Create pipeline with Mamba enabled
    pipeline = BifrostPipeline(
        n_fft_s0=512,
        n_fft_s1=256,
        d_model=128,
        n_scales=4,
        n_heads=4,
        use_mamba=True,  # Enable Mamba-3
        preserve_frames=True,
    ).cuda()

    print(f"Pipeline on device: {next(pipeline.parameters()).device}")

    # Test signal
    signal = torch.randn(2, 1, 8000, device="cuda")  # batch=2, channels=1, samples=8000

    # Forward pass
    with torch.no_grad():
        bound_st, coherence = pipeline(signal, {"sample_rate": 8000})

    print(f"Output amplitude shape: {bound_st.amplitude.shape}")
    print(f"Coherence shape: {coherence.shape}")
    print("FBC Mamba-3 integration: SUCCESS")

    return True


def main():
    """Run all verification tests."""
    print("Mamba-SSM GPU Verification for FBC")
    print("=" * 60)

    checks = [
        ("CUDA/GPU", check_cuda),
        ("Mamba-SSM Install", check_mamba_ssm),
        ("Mamba Forward Pass", test_mamba_forward),
        ("FBC Integration", test_fbc_mamba_integration),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r for _, r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
