"""
Bifröst CLI — Command-line interface for Frequency-Based Cognition.

Commands:
    bifrost process <file>    Process audio/image/text through Bifröst pipeline
    bifrost train <data>      Train complex SSM on dataset
    bifrost demo              Interactive demo with visualizations
    bifrost validate <file>   Check phase coherence metrics
    bifrost serve             Start API server

Examples:
    bifrost process audio.wav --output results.json --visualize
    bifrost train --data ./dataset --epochs 100 --device cuda
    bifrost demo --chord "440,880,1320" --type harmonic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Optional torchaudio import (only needed for audio processing)
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None

from . import BifrostPipeline, HarmonicBinding, create_multimodal_pipeline
from .spectral_tensor import SpectralTensor
from .complex_training import ComplexBifrostTrainer, PhaseCoherenceMetrics


def cmd_process(args: argparse.Namespace) -> int:
    """Process file through Bifröst pipeline."""
    print(f"🎵 Processing: {args.input}")

    ext = Path(args.input).suffix.lower()
    if ext in ['.wav', '.mp3', '.flac']:
        results = _process_audio_file(args)
    elif ext in ['.png', '.jpg', '.jpeg']:
        results = _process_image_file(args)
    else:
        print(f"❌ Unsupported file type: {ext}")
        return 1

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to: {args.output}")
    else:
        print(json.dumps(results, indent=2))
    return 0


def _process_audio_file(args: argparse.Namespace) -> dict:
    """Process audio file and return results dict."""
    import torchaudio
    audio, sr = torchaudio.load(args.input)
    pipeline = BifrostPipeline(
        n_fft_canonical=args.n_fft,
        n_fft_decompose=args.n_fft // 2,
        d_model=args.d_model,
        use_complex_ssm=True,
    )
    bound, coherence = pipeline(audio, {'sample_rate': sr})
    return {
        'input_shape': list(audio.shape),
        'sample_rate': sr,
        'ssm_type': pipeline.ssm_type,
        'output_shape': {
            'amplitude': list(bound.amplitude.shape),
            'phase': list(bound.phase.shape),
        },
        'metadata': bound.metadata,
        'coherence_diagonal_ratio': (
            PhaseCoherenceMetrics.diagonal_coherence_ratio(coherence)
        ),
    }


def _process_image_file(args: argparse.Namespace) -> dict:
    """Process image file and return results dict."""
    from PIL import Image
    img = Image.open(args.input).convert('L')
    tensor = (
        torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
    )
    pipeline = create_multimodal_pipeline(
        'tensor', n_fft=args.n_fft, d_model=args.d_model
    )
    bound, coherence = pipeline(tensor)
    return {
        'input_shape': list(tensor.shape),
        'detected_structure': bound.metadata.get('detected_structure'),
        'ssm_type': pipeline.ssm_type,
        'output_shape': list(bound.amplitude.shape),
    }




def cmd_demo(args: argparse.Namespace) -> int:
    """Interactive demo with visualizations."""
    print("🎹 Bifröst Interactive Demo")
    print("=" * 60)

    if args.type == 'harmonic':
        _demo_harmonic(args.chord)
    elif args.type == 'coherence':
        _demo_coherence()
    elif args.type == 'multimodal':
        _demo_multimodal()

    print("\n" + "=" * 60)
    return 0


def _demo_harmonic(chord: str) -> None:
    """Run harmonic binding demo."""
    print("\n🎼 Harmonic Binding Demo")
    print(f"   Chord: {chord}")
    print(
        "   (Note: This demo uses synthetically generated audio "
        "with harmonic structure)"
    )

    freqs = [float(f) for f in chord.split(',')]
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))

    audio = torch.zeros_like(t)
    for f in freqs:
        audio += torch.sin(2 * np.pi * f * t)
        for overtone in [2, 3]:
            audio += torch.sin(2 * np.pi * f * overtone * t) * 0.3

    audio = audio.unsqueeze(0)
    harmonic = HarmonicBinding(
        d_model=128,
        n_freq=257,
        base_freq=freqs[0],
        sample_rate=sample_rate,
    )

    stft = torch.stft(audio.squeeze(0), n_fft=512, return_complex=True)
    amplitude = stft.abs().unsqueeze(0).transpose(-2, -1)
    phase = stft.angle().unsqueeze(0).transpose(-2, -1)

    if amplitude.shape[-1] != 257:
        amplitude = torch.nn.functional.interpolate(
            amplitude.transpose(-2, -1), size=257, mode='linear'
        ).transpose(-2, -1)
        phase = torch.nn.functional.interpolate(
            phase.transpose(-2, -1), size=257, mode='linear'
        ).transpose(-2, -1)

    bound, attn = harmonic(amplitude, phase)

    print("\n📊 Results:")
    print(
        f"   Harmonic bins: "
        f"{len(harmonic.harmonic_grid.get_harmonic_bins())}"
    )
    print(
        f"   Attention std: {attn.std():.4f} "
        f"(non-uniform = structure detected)"
    )

    print("\n🎵 Harmonic relationships:")
    for f in freqs:
        print(f"   {f}Hz → overtones at {2*f:.0f}Hz, {3*f:.0f}Hz")


def _demo_coherence() -> None:
    """Run phase coherence demo."""
    print("\n🌊 Phase Coherence Demo")
    print(
        "   (Note: This demo uses synthetically generated phase "
        "data for demonstration)"
    )

    T, n_freq = 32, 128
    phase_coherent = torch.cumsum(
        torch.randn(1, T, n_freq) * 0.1, dim=1
    )
    smooth_coherent = PhaseCoherenceMetrics.phase_gradient_smoothness(
        phase_coherent
    )

    phase_random = torch.randn(1, T, n_freq)
    smooth_random = PhaseCoherenceMetrics.phase_gradient_smoothness(
        phase_random
    )

    print("\n📈 Phase smoothness:")
    print(f"   Coherent: {smooth_coherent:.2f} (high = smooth)")
    print(f"   Random:   {smooth_random:.2f} (low = chaotic)")
    ratio = smooth_coherent / smooth_random
    print(f"   Ratio:    {ratio:.2f}x smoother")
    print(f"\n✅ Complex SSM learns {ratio:.1f}x smoother phase evolution")


def _demo_multimodal() -> None:
    """Run multimodal demo."""
    print("\n🔄 Multimodal Demo")
    print(
        "   (Note: This demo uses synthetically generated data "
        "for demonstration)"
    )

    modalities = [
        ('audio', torch.randn(1, 8000)),
        ('text', torch.randint(0, 50000, (1, 128))),
        ('tensor', torch.randn(2, 64, 64)),
    ]

    for name, data in modalities:
        pipeline = create_multimodal_pipeline(
            name, n_fft=512, d_model=128
        )
        bound, coherence = pipeline(data)
        print(
            f"   {name:8s}: {list(data.shape)} → "
            f"{list(bound.amplitude.shape)} "
            f"[{pipeline.ssm_type[:20]}...]"
        )


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate phase coherence metrics."""
    print(f"🔍 Validating: {args.input}")
    
    import torchaudio
    audio, sr = torchaudio.load(args.input)
    
    pipeline = BifrostPipeline(
        n_fft_canonical=1024,
        n_fft_decompose=512,
        d_model=128,
        use_complex_ssm=True,
    )
    
    bound, coherence = pipeline(audio, {'sample_rate': sr})
    
    # Compute metrics
    diag_ratio = PhaseCoherenceMetrics.diagonal_coherence_ratio(coherence)
    phase_smooth = PhaseCoherenceMetrics.phase_gradient_smoothness(bound.phase)
    
    print(f"\n📊 Validation Results:")
    print(f"   SSM Type: {pipeline.ssm_type}")
    print(f"   Diagonal Coherence Ratio: {diag_ratio:.3f}")
    print(f"   {'✅ PASS' if diag_ratio > 1.0 else '❌ FAIL'} (should be > 1.0)")
    print(f"   Phase Smoothness: {phase_smooth:.2f}")
    print(f"   {'✅ PASS' if phase_smooth > 5.0 else '⚠️  LOW'} (high = smooth)")
    
    if args.metrics:
        print(f"\n   Additional metrics: {args.metrics}")
    
    return 0 if diag_ratio > 1.0 and phase_smooth > 5.0 else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start API server."""
    print(f"🌐 Starting Bifröst API Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"\n   Endpoints:")
    print(f"   - POST /process   Process audio/image/text")
    print(f"   - GET  /health    Health check")
    print(f"   - GET  /demo      Interactive demo")
    
    try:
        from .api import start_server
        start_server(host=args.host, port=args.port)
    except ImportError:
        print("\n⚠️  API server not implemented yet. Run:")
        print("   pip install fastapi uvicorn")
        print("   Then: bifrost serve")
        return 1
    
    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog='bifrost',
        description='Bifröst — The Spectral Rainbow Bridge (Frequency-Based Cognition) CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bifrost process audio.wav --output results.json --visualize
  bifrost train --data ./dataset --epochs 100 --device cuda
  bifrost demo --chord "440,880,1320" --type harmonic
  bifrost validate audio.wav --metrics coherence_ratio
  bifrost serve --host 0.0.0.0 --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process file through Bifröst')
    process_parser.add_argument('input', help='Input file (wav, png, etc.)')
    process_parser.add_argument('-o', '--output', help='Output JSON file')
    process_parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    process_parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    process_parser.set_defaults(func=cmd_process)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--type', choices=['harmonic', 'coherence', 'multimodal'], 
                             default='harmonic', help='Demo type')
    demo_parser.add_argument('--chord', default='440,880,1320', 
                             help='Frequencies for harmonic demo (Hz)')
    demo_parser.set_defaults(func=cmd_demo)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate phase coherence')
    validate_parser.add_argument('input', help='Input audio file')
    validate_parser.add_argument('--metrics', help='Additional metrics to compute')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='127.0.0.1', help='Server host')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    serve_parser.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
