"""
Empirical Validation of Harmonic/Projected Coherence Blend Ratio

Validates the optimal blend ratio between original-phase coherence (harmonic-preserving)
and projected-phase coherence (learned) using ground-truth harmonic signals.

Per Agentic CTO-Persona Policy:
- C-01: Full documentation
- C-02: Explicit error handling
- C-04: Complexity ≤10
- G1-G5: Complete, executable, correct
"""

import argparse
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

from bifrost.pipeline import BifrostPipeline
from bifrost.spectral_tensor import SpectralTensor


def generate_harmonic_chord(
    base_freq: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Generate harmonic chord: f, 2f, 3f (440Hz + 880Hz + 1320Hz).
    
    Args:
        base_freq: Fundamental frequency (Hz)
        duration_s: Signal duration in seconds
        sample_rate: Sampling rate (Hz)
        
    Returns:
        Tensor of shape (1, n_samples) containing harmonic chord
    """
    n_samples = int(sample_rate * duration_s)
    t = torch.linspace(0.0, duration_s, n_samples)
    
    # Fundamental + 2nd harmonic + 3rd harmonic with decreasing amplitudes
    signal = (
        torch.sin(2.0 * math.pi * base_freq * t)
        + 0.6 * torch.sin(2.0 * math.pi * 2.0 * base_freq * t)
        + 0.4 * torch.sin(2.0 * math.pi * 3.0 * base_freq * t)
    )
    
    return signal.unsqueeze(0)


def generate_inharmonic_chord(
    freq1: float = 440.0,
    freq2: float = 700.0,  # Not harmonic with 440
    freq3: float = 1120.0,  # Not harmonic
    duration_s: float = 1.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Generate inharmonic chord (non-integer frequency ratios).
    
    Args:
        freq1, freq2, freq3: Three non-harmonic frequencies
        duration_s: Signal duration in seconds
        sample_rate: Sampling rate (Hz)
        
    Returns:
        Tensor of shape (1, n_samples) containing inharmonic chord
    """
    n_samples = int(sample_rate * duration_s)
    t = torch.linspace(0.0, duration_s, n_samples)
    
    signal = (
        torch.sin(2.0 * math.pi * freq1 * t)
        + 0.6 * torch.sin(2.0 * math.pi * freq2 * t)
        + 0.4 * torch.sin(2.0 * math.pi * freq3 * t)
    )
    
    return signal.unsqueeze(0)


def measure_harmonic_preservation(
    spectral: SpectralTensor,
    base_freq: float,
    sample_rate: int,
    n_fft: int = 512,
) -> Dict[str, float]:
    """
    Measure how well spectral representation preserves harmonic structure.
    
    Metrics:
    - harmonic_peak_ratio: Ratio of energy at harmonic vs non-harmonic bins
    - octave_coherence: Phase coherence between f and 2f
    - inharmonic_rejection: Ability to suppress inharmonic frequencies
    
    Args:
        spectral: SpectralTensor with amplitude and phase
        base_freq: Expected fundamental frequency (Hz)
        sample_rate: Audio sampling rate
        n_fft: FFT size used in canonicalizer
        
    Returns:
        Dictionary of preservation metrics (higher = better harmonic preservation)
    """
    # Frequency bin resolution
    freq_resolution = sample_rate / n_fft
    
    # Get frequency bins
    n_freq = spectral.amplitude.shape[-1]
    freq_bins = torch.linspace(0, sample_rate / 2, n_freq)
    
    # Find harmonic bins (integer multiples of base_freq)
    harmonic_bins = []
    harmonic_mask = torch.zeros(n_freq, dtype=torch.bool)
    
    for h in range(1, 6):  # 1st to 5th harmonic
        target_freq = base_freq * h
        bin_idx = int(target_freq / freq_resolution)
        if bin_idx < n_freq:
            harmonic_bins.append(bin_idx)
            # Mark nearby bins as harmonic (±1 bin tolerance)
            for offset in [-1, 0, 1]:
                idx = bin_idx + offset
                if 0 <= idx < n_freq:
                    harmonic_mask[idx] = True
    
    # Find non-harmonic bins (excluding DC and Nyquist)
    non_harmonic_mask = ~harmonic_mask
    non_harmonic_mask[0] = False  # Exclude DC
    non_harmonic_mask[-1] = False  # Exclude Nyquist edge
    
    # Amplitude-based harmonic preservation
    amp = spectral.amplitude.abs().mean(dim=(0, 1))  # Average over batch and time
    
    harmonic_energy = amp[harmonic_mask].sum().item()
    non_harmonic_energy = amp[non_harmonic_mask].sum().item()
    
    harmonic_peak_ratio = harmonic_energy / (non_harmonic_energy + 1e-8)
    
    # Phase coherence between adjacent harmonics (f and 2f)
    octave_coherence = 0.0
    if len(harmonic_bins) >= 2:
        # Get phase at fundamental and 2nd harmonic
        phase = spectral.phase.mean(dim=(0, 1))  # Average over batch and time
        phase_f = phase[harmonic_bins[0]]
        phase_2f = phase[harmonic_bins[1]]
        
        # Phase coherence = cos(2*phase_f - phase_2f) 
        # For perfect harmonic relationship, phase_2f ≈ 2*phase_f
        phase_diff = 2 * phase_f - phase_2f
        octave_coherence = torch.cos(phase_diff).item()
    
    return {
        "harmonic_peak_ratio": harmonic_peak_ratio,
        "octave_coherence": octave_coherence,
        "harmonic_energy": harmonic_energy,
        "non_harmonic_energy": non_harmonic_energy,
    }


def test_blend_ratio(
    pipeline: BifrostPipeline,
    blend_ratio: float,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Test a specific blend ratio on harmonic and inharmonic signals.
    
    Args:
        pipeline: BifrostPipeline instance
        blend_ratio: Ratio for original-phase coherence (0.0 to 1.0)
        device: Device for computation
        
    Returns:
        Dictionary of performance metrics for this blend ratio
    """
    # Temporarily set blend ratio in binding
    if hasattr(pipeline.binding, 'harmonic_blend_ratio'):
        original_ratio = pipeline.binding.harmonic_blend_ratio
        pipeline.binding.harmonic_blend_ratio = blend_ratio
    else:
        # Store as attribute for testing
        pipeline.binding.harmonic_blend_ratio = blend_ratio
        original_ratio = None
    
    try:
        # Test on harmonic signal (should preserve structure)
        harmonic_signal = generate_harmonic_chord(base_freq=440.0).to(device)
        
        with torch.no_grad():
            harmonic_spectral, _ = pipeline(harmonic_signal)
        
        harmonic_metrics = measure_harmonic_preservation(
            harmonic_spectral, base_freq=440.0, sample_rate=16000
        )
        
        # Test on inharmonic signal (should show different response)
        inharmonic_signal = generate_inharmonic_chord().to(device)
        
        with torch.no_grad():
            inharmonic_spectral, _ = pipeline(inharmonic_signal)
        
        inharmonic_metrics = measure_harmonic_preservation(
            inharmonic_spectral, base_freq=440.0, sample_rate=16000
        )
        
        # Discrimination score: how well does it separate harmonic from inharmonic?
        # A good blend should: 
        # 1. High harmonic_peak_ratio for harmonic signals
        # 2. Low harmonic_peak_ratio for inharmonic signals
        discrimination = (
            harmonic_metrics["harmonic_peak_ratio"] 
            - inharmonic_metrics["harmonic_peak_ratio"]
        )
        
        return {
            "blend_ratio": blend_ratio,
            "harmonic_peak_ratio": harmonic_metrics["harmonic_peak_ratio"],
            "octave_coherence": harmonic_metrics["octave_coherence"],
            "inharmonic_peak_ratio": inharmonic_metrics["harmonic_peak_ratio"],
            "discrimination": discrimination,
        }
        
    finally:
        # Restore original ratio
        if original_ratio is not None:
            pipeline.binding.harmonic_blend_ratio = original_ratio


def find_optimal_blend_ratio(
    d_model: int = 128,
    n_fft: int = 512,
    device: str = "cpu",
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """
    Empirically find the optimal blend ratio by testing multiple values.
    
    Args:
        d_model: Model dimension
        n_fft: FFT size
        device: Device for computation
        
    Returns:
        Tuple of (optimal_ratio, all_results_dict)
    """
    print("=" * 60)
    print("EMPIRICAL BLEND RATIO VALIDATION")
    print("=" * 60)
    print(f"Testing blend ratios: 0.5, 0.6, 0.7, 0.8, 0.9")
    print(f"Device: {device}")
    print()
    
    # NOTE: Blend ratio validation requires use_complex_ssm=False
    # The harmonic_blend_ratio only affects the dual-stream SSM path where
    # use_original_phase=True (n_freq_in is set, triggering projection).
    # With complex SSM, the decomposer already outputs d_model dims, so
    # no projection is needed and the blend ratio has no effect.
    pipeline = BifrostPipeline(
        d_model=d_model,
        n_fft_s0=n_fft,
        use_complex_ssm=False,  # Required for blend ratio testing
    ).to(device)
    pipeline.eval()
    
    blend_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    for ratio in blend_ratios:
        print(f"Testing blend_ratio = {ratio:.1f}...")
        
        metrics = test_blend_ratio(pipeline, ratio, device)
        results[ratio] = metrics
        
        print(f"  Harmonic peak ratio: {metrics['harmonic_peak_ratio']:.3f}")
        print(f"  Octave coherence:    {metrics['octave_coherence']:.3f}")
        print(f"  Inharmonic peak:     {metrics['inharmonic_peak_ratio']:.3f}")
        print(f"  Discrimination:      {metrics['discrimination']:.3f}")
        print()
    
    # Find optimal ratio based on discrimination (best harmonic/inharmonic separation)
    optimal_ratio = max(results.keys(), key=lambda r: results[r]["discrimination"])
    
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Ratio':<8} {'Harmonic':<10} {'Inharmonic':<12} {'Discrim':<10}")
    print("-" * 60)
    
    for ratio in blend_ratios:
        r = results[ratio]
        marker = " ***" if ratio == optimal_ratio else ""
        print(
            f"{ratio:<8.1f} "
            f"{r['harmonic_peak_ratio']:<10.3f} "
            f"{r['inharmonic_peak_ratio']:<12.3f} "
            f"{r['discrimination']:<10.3f}{marker}"
        )
    
    print("-" * 60)
    print(f"\nOPTIMAL BLEND RATIO: {optimal_ratio:.1f}")
    print(f"  (Best discrimination between harmonic and inharmonic signals)")
    print()
    
    # Recommendation
    if optimal_ratio != 0.7:
        print(f"RECOMMENDATION: Update binding.py line 226 from 0.7 to {optimal_ratio}")
        print(f"  Current: coherence = 0.7 * coherence_orig + 0.3 * coherence")
        print(f"  Optimal: coherence = {optimal_ratio:.1f} * coherence_orig + {1-optimal_ratio:.1f} * coherence")
    else:
        print("RECOMMENDATION: Current 0.7/0.3 ratio is empirically validated.")
    
    return optimal_ratio, results


def main():
    parser = argparse.ArgumentParser(
        description="Empirical validation of harmonic/projected coherence blend ratio"
    )
    parser.add_argument(
        "--d-model", type=int, default=128,
        help="Model dimension (default: 128)"
    )
    parser.add_argument(
        "--n-fft", type=int, default=512,
        help="FFT size (default: 512)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for computation (cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    optimal_ratio, results = find_optimal_blend_ratio(
        d_model=args.d_model,
        n_fft=args.n_fft,
        device=args.device,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
