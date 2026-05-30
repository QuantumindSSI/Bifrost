"""
Customer Usage Test for FBC Implementation

This test validates the complete FBC pipeline as a customer would use it:
- Audio processing with phase coherence
- Harmonic binding for musical signals (440Hz ↔ 880Hz)
- Multimodal support (audio, text, tensor)
- Training workflow with metrics

Run: python tests/test_customer_usage.py
"""

import torch
import numpy as np
from bifrost import (
    BifrostPipeline,
    create_multimodal_pipeline,
    HarmonicBinding,
    PhaseCoherenceMetrics,
    ComplexBifrostTrainer,
)
from bifrost.spectral_tensor import SpectralTensor


def test_basic_fbc_pipeline():
    """Test 1: Basic FBC pipeline with default complex SSM."""
    print("\n" + "="*60)
    print("TEST 1: Basic FBC Pipeline (Complex SSM Default)")
    print("="*60)

    # Create pipeline (complex SSM is now default!)
    pipeline = BifrostPipeline(
        n_fft_s0=1024,
        n_fft_s1=512,
        d_model=128,
    )

    print(f"SSM Type: {pipeline.ssm_type}")
    assert "ComplexSpectralDecomposer" in pipeline.ssm_type, "Complex SSM should be default"

    # Generate test audio: 440Hz sine wave (A4 note)
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # Batch=1

    print(f"Input: {audio.shape} (1 second of 440Hz tone)")

    # Process through pipeline
    bound, coherence = pipeline(audio, {'sample_rate': sample_rate})

    print(f"Output amplitude: {bound.amplitude.shape}")
    print(f"Output phase: {bound.phase.shape}")
    print(f"Coherence shape: {coherence.shape}")

    # Verify phase coherence metadata
    assert bound.metadata.get('phase_coherence') == 'learned_via_complex_ssm', \
        "Phase should be processed through complex SSM"

    print("✅ TEST 1 PASSED: Complex SSM pipeline working")
    return True


def test_harmonic_binding_musical():
    """Test 2: Harmonic binding with musical chord (C-E-G = 1:1.25:1.5 ratio)."""
    print("\n" + "="*60)
    print("TEST 2: Harmonic Binding (Musical Chord)")
    print("="*60)

    # Create harmonic binding with A4=440Hz base
    harmonic_binding = HarmonicBinding(
        d_model=128,
        n_heads=4,
        n_freq=257,
        base_freq=440.0,  # A4
        sample_rate=16000.0,
    )

    # Generate C major chord: C (261.63Hz), E (329.63Hz), G (392.00Hz)
    # These have approximate ratios 1 : 1.26 : 1.50
    sample_rate = 16000
    duration = 0.5
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # C major chord with harmonics
    c_freq, e_freq, g_freq = 261.63, 329.63, 392.00
    chord = (
        torch.sin(2 * np.pi * c_freq * t) +
        torch.sin(2 * np.pi * e_freq * t) * 0.8 +
        torch.sin(2 * np.pi * g_freq * t) * 0.6
    ).unsqueeze(0)

    # Add overtones (2f, 3f) for each note
    for fundamental in [c_freq, e_freq, g_freq]:
        for overtone in [2, 3]:
            chord += torch.sin(2 * np.pi * fundamental * overtone * t) * 0.3

    print(f"Input chord: C={c_freq}Hz, E={e_freq}Hz, G={g_freq}Hz + overtones")

    # Convert to SpectralTensor
    n_fft = 512
    stft = torch.stft(chord.squeeze(0), n_fft=n_fft, return_complex=True)
    amplitude = stft.abs().unsqueeze(0).transpose(-2, -1)  # (1, T, n_freq)
    phase = stft.angle().unsqueeze(0).transpose(-2, -1)

    # Ensure correct shape
    if amplitude.shape[-1] != 257:
        amplitude = torch.nn.functional.interpolate(
            amplitude.transpose(-2, -1), size=257, mode='linear'
        ).transpose(-2, -1)
        phase = torch.nn.functional.interpolate(
            phase.transpose(-2, -1), size=257, mode='linear'
        ).transpose(-2, -1)

    print(f"Spectral amplitude: {amplitude.shape}")

    # Apply harmonic binding
    bound, attn = harmonic_binding(amplitude, phase)

    print(f"Harmonic bins detected: {len(harmonic_binding.harmonic_grid.get_harmonic_bins())}")
    print(f"Attention mean: {attn.mean():.4f}, std: {attn.std():.4f}")

    # Check that attention is non-uniform (harmonic structure detected)
    assert attn.std() > 0.001, "Attention should be non-uniform for harmonic input"

    print("✅ TEST 2 PASSED: Harmonic binding detects musical structure")
    return True


def test_multimodal_usage():
    """Test 3: Multimodal pipeline (audio, text, tensor)."""
    print("\n" + "="*60)
    print("TEST 3: Multimodal Pipeline")
    print("="*60)

    # Test AUDIO
    print("\n--- Audio Modality ---")
    audio_pipe = create_multimodal_pipeline(
        modality='audio',
        n_fft=1024,
        d_model=128,
    )
    print(f"SSM: {audio_pipe.ssm_type}")

    sample_rate = 16000
    t = torch.linspace(0, 0.5, int(sample_rate * 0.5))
    audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
    bound, coherence = audio_pipe(audio)
    print(f"Audio: {audio.shape} -> {bound.amplitude.shape}")

    # Test TEXT
    print("\n--- Text Modality ---")
    text_pipe = create_multimodal_pipeline(
        modality='text',
        n_fft=512,
        d_model=128,
    )
    print(f"SSM: {text_pipe.ssm_type}")

    tokens = torch.randint(0, 50000, (1, 128))
    bound, coherence = text_pipe(tokens)
    print(f"Text: {tokens.shape} -> {bound.amplitude.shape}")

    # Test TENSOR
    print("\n--- Tensor Modality ---")
    tensor_pipe = create_multimodal_pipeline(
        modality='tensor',
        n_fft=1024,
        d_model=128,
    )
    print(f"SSM: {tensor_pipe.ssm_type}")

    # 2D spatial tensor
    tensor = torch.randn(2, 64, 64)
    bound, coherence = tensor_pipe(tensor)
    print(f"Tensor (2D spatial): {tensor.shape} -> {bound.amplitude.shape}")
    assert bound.metadata.get('detected_structure') == '2d_spatial', \
        "Should detect 2D spatial structure"

    # 1D temporal tensor
    temporal = torch.randn(2, 500)
    bound, coherence = tensor_pipe(temporal)
    print(f"Tensor (1D temporal): {temporal.shape} -> {bound.amplitude.shape}")

    print("\n✅ TEST 3 PASSED: All modalities working with complex SSM")
    return True


def test_phase_coherence_metrics():
    """Test 4: Phase coherence metrics validation."""
    print("\n" + "="*60)
    print("TEST 4: Phase Coherence Metrics")
    print("="*60)

    # Create coherent phase (smooth progression)
    phase_coherent = torch.cumsum(torch.randn(2, 32, 128) * 0.1, dim=1)

    # Create random phase (incoherent)
    phase_random = torch.randn(2, 32, 128)

    # Measure smoothness
    smooth_coherent = PhaseCoherenceMetrics.phase_gradient_smoothness(phase_coherent)
    smooth_random = PhaseCoherenceMetrics.phase_gradient_smoothness(phase_random)

    print(f"Coherent phase smoothness: {smooth_coherent:.2f}")
    print(f"Random phase smoothness: {smooth_random:.2f}")

    # Coherent phase should be smoother (higher smoothness value)
    assert smooth_coherent > smooth_random, \
        f"Coherent phase should be smoother: {smooth_coherent} vs {smooth_random}"

    # Test diagonal coherence ratio
    # Use positive values (like softmax attention weights)
    coherence = torch.rand(2, 4, 32, 32) * 0.5 + 0.1  # All positive, [0.1, 0.6]
    # Make diagonal stronger
    for b in range(2):
        for h in range(4):
            coherence[b, h] = coherence[b, h] + torch.eye(32) * 1.5  # Boost diagonal

    ratio = PhaseCoherenceMetrics.diagonal_coherence_ratio(coherence)
    print(f"Diagonal coherence ratio: {ratio:.3f} (should be > 1.0)")

    assert ratio > 1.0, f"Strong diagonal should have ratio > 1.0, got {ratio}"

    # Test complex correlation
    z1 = torch.randn(2, 32, 128) + 1j * torch.randn(2, 32, 128)
    z2 = z1 * 0.9 + torch.randn(2, 32, 128) * 0.1 + 1j * torch.randn(2, 32, 128) * 0.1
    corr = PhaseCoherenceMetrics.complex_state_correlation(z1, z2)
    print(f"Complex correlation: {corr:.3f} (should be high for similar states)")

    assert corr > 0.5, f"Similar complex states should have high correlation: {corr}"

    print("\n✅ TEST 4 PASSED: Phase coherence metrics working correctly")
    return True


def test_training_workflow():
    """Test 5: Complex SSM training workflow."""
    print("\n" + "="*60)
    print("TEST 5: Training Workflow (Complex SSM)")
    print("="*60)

    from bifrost import ComplexSpectralDecomposer, ComplexNextStepLoss

    # Use n_fft=256, so n_freq = 129. Set d_model=129 to match.
    n_fft = 256
    n_freq = n_fft // 2 + 1  # 129
    d_model = n_freq  # 129 - matching dimensions

    # Create decomposer
    decomposer = ComplexSpectralDecomposer(
        n_fft=n_fft,
        d_model=d_model,
        n_frames=8,
        d_state=8,
    )

    # Create loss function
    criterion = ComplexNextStepLoss()

    # Create synthetic training data: complex spectral tensor
    B, T = 2, 8

    # Generate coherent complex sequence
    z_input = torch.randn(B, n_freq) + 1j * torch.randn(B, n_freq)
    z_input = z_input.unsqueeze(1).expand(B, T, n_freq)  # (B, T, n_freq)

    # Add temporal structure
    for t in range(1, T):
        phase_shift = torch.exp(1j * torch.tensor(0.1 * t))
        z_input[:, t, :] = z_input[:, t, :] * phase_shift + torch.randn(B, n_freq) * 0.1

    # Create SpectralTensor
    batch = SpectralTensor(
        amplitude=z_input.abs()[:, 0, :],  # First frame
        phase=z_input.angle()[:, 0, :],
        scale=torch.linspace(0, 8000, n_freq).expand(B, -1),
        uncertainty=torch.ones(B, n_freq) * 0.1,
    )

    # Forward pass
    print("Testing forward pass...")
    decomposed, _ = decomposer(batch)
    print(f"Decomposed: {decomposed.amplitude.shape}")

    # Compute loss manually - both should be (B, T-1, d_model)
    pred_complex = torch.complex(
        decomposed.amplitude[:, :-1, :],
        decomposed.phase[:, :-1, :]
    )
    target_z = z_input[:, 1:, :]  # Shift by one frame

    print(f"Pred: {pred_complex.shape}, Target: {target_z.shape}")
    
    # Check for NaN/Inf in inputs before loss computation
    assert torch.isfinite(pred_complex).all(), "Pred contains NaN/Inf"
    assert torch.isfinite(target_z).all(), "Target contains NaN/Inf"
    
    loss = criterion(pred_complex, target_z)
    print(f"Loss: {loss.item():.4f}")

    # Verify loss is finite
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    print("\n✅ TEST 5 PASSED: Training workflow functional")
    return True


def run_all_tests():
    """Run all customer usage tests."""
    print("\n" + "="*60)
    print("FBC CUSTOMER USAGE TEST SUITE")
    print("="*60)
    print("\nThis test validates FBC as a customer would use it:")
    print("- Complex SSM for phase coherence (now default)")
    print("- Harmonic binding for musical signals (440Hz ↔ 880Hz)")
    print("- Multimodal support (audio, text, tensor)")
    print("- Phase coherence metrics and training")

    tests = [
        test_basic_fbc_pipeline,
        test_harmonic_binding_musical,
        test_multimodal_usage,
        test_phase_coherence_metrics,
        test_training_workflow,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! FBC is ready for customer use.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
