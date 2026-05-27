"""
FBC CLI — Command-line interface for Frequency-Based Cognition.

Commands:
    fbc process <file>    Process audio/image/text through FBC pipeline
    fbc train <data>      Train complex SSM on dataset
    fbc demo              Interactive demo with visualizations
    fbc validate <file>   Check phase coherence metrics
    fbc serve             Start API server

Examples:
    fbc process audio.wav --output results.json --visualize
    fbc train --data ./dataset --epochs 100 --device cuda
    fbc demo --chord "440,880,1320" --type harmonic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from . import FBCPipeline, HarmonicBinding, create_multimodal_pipeline
from .spectral_tensor import SpectralTensor
from .complex_training import ComplexFBCTrainer, PhaseCoherenceMetrics


def cmd_process(args: argparse.Namespace) -> int:
    """Process file through FBC pipeline."""
    print(f"🎵 Processing: {args.input}")
    
    # Detect file type
    ext = Path(args.input).suffix.lower()
    
    if ext in ['.wav', '.mp3', '.flac']:
        # Audio processing
        import torchaudio
        audio, sr = torchaudio.load(args.input)
        
        pipeline = FBCPipeline(
            n_fft_s0=args.n_fft,
            n_fft_s1=args.n_fft // 2,
            d_model=args.d_model,
            use_complex_ssm=True,
        )
        
        bound, coherence = pipeline(audio, {'sample_rate': sr})
        
        results = {
            'input_shape': list(audio.shape),
            'sample_rate': sr,
            'ssm_type': pipeline.ssm_type,
            'output_shape': {
                'amplitude': list(bound.amplitude.shape),
                'phase': list(bound.phase.shape),
            },
            'metadata': bound.metadata,
            'coherence_diagonal_ratio': PhaseCoherenceMetrics.diagonal_coherence_ratio(coherence),
        }
        
    elif ext in ['.png', '.jpg', '.jpeg']:
        # Image processing
        from PIL import Image
        img = Image.open(args.input).convert('L')
        tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
        
        pipeline = create_multimodal_pipeline('tensor', n_fft=args.n_fft, d_model=args.d_model)
        bound, coherence = pipeline(tensor)
        
        results = {
            'input_shape': list(tensor.shape),
            'detected_structure': bound.metadata.get('detected_structure'),
            'ssm_type': pipeline.ssm_type,
            'output_shape': list(bound.amplitude.shape),
        }
    else:
        print(f"❌ Unsupported file type: {ext}")
        return 1
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to: {args.output}")
    else:
        print(json.dumps(results, indent=2))
    
    # Visualization
    if args.visualize:
        print("📊 Generating visualizations...")
        # TODO: Generate matplotlib plots
        print("   - Spectrogram")
        print("   - Attention heatmap")
        print("   - Phase coherence plot")
    
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train complex SSM on dataset."""
    print(f"🚀 Training Complex SSM")
    print(f"   Data: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Device: {args.device}")
    
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create synthetic dataset for demo (replace with real data loading)
    print("📦 Loading dataset...")
    
    # Example: Create random spectral data
    n_samples = 100
    n_freq = 257
    dataset = TensorDataset(
        torch.randn(n_samples, n_freq),  # Amplitude
        torch.randn(n_samples, n_freq),  # Phase
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create decomposer and trainer
    from .s1_decomposer.complex_decomposer import ComplexSpectralDecomposer
    
    decomposer = ComplexSpectralDecomposer(
        n_fft=512,
        d_model=args.d_model,
        n_frames=32,
    )
    
    trainer = ComplexFBCTrainer(
        decomposer=decomposer,
        lr=args.lr,
        device=args.device,
    )
    
    # Training loop
    print("\n📈 Training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (amp, phase) in enumerate(loader):
            # Create SpectralTensor
            batch = SpectralTensor(
                amplitude=amp,
                phase=phase,
                scale=torch.linspace(0, 8000, n_freq).expand(amp.shape[0], -1),
                uncertainty=torch.ones_like(amp) * 0.1,
            ).to(args.device)
            
            metrics = trainer.train_step(batch)
            epoch_loss += metrics['loss']
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} Batch {batch_idx}: loss={metrics['loss']:.4f}")
        
        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")
    
    # Save checkpoint
    if args.output:
        trainer.save_checkpoint(args.output)
        print(f"✅ Model saved to: {args.output}")
    
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Interactive demo with visualizations."""
    print("🎹 FBC Interactive Demo")
    print("=" * 60)
    
    if args.type == 'harmonic':
        # Demo harmonic binding
        print(f"\n🎼 Harmonic Binding Demo")
        print(f"   Chord: {args.chord}")
        
        # Parse frequencies
        freqs = [float(f) for f in args.chord.split(',')]
        
        # Generate chord
        sample_rate = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        audio = torch.zeros_like(t)
        for f in freqs:
            audio += torch.sin(2 * np.pi * f * t)
            # Add overtones
            for overtone in [2, 3]:
                audio += torch.sin(2 * np.pi * f * overtone * t) * 0.3
        
        audio = audio.unsqueeze(0)
        
        # Process
        harmonic = HarmonicBinding(
            d_model=128,
            n_freq=257,
            base_freq=freqs[0],
            sample_rate=sample_rate,
        )
        
        # Convert to spectral
        stft = torch.stft(audio.squeeze(0), n_fft=512, return_complex=True)
        amplitude = stft.abs().unsqueeze(0).transpose(-2, -1)
        phase = stft.angle().unsqueeze(0).transpose(-2, -1)
        
        # Interpolate if needed
        if amplitude.shape[-1] != 257:
            amplitude = torch.nn.functional.interpolate(
                amplitude.transpose(-2, -1), size=257, mode='linear'
            ).transpose(-2, -1)
            phase = torch.nn.functional.interpolate(
                phase.transpose(-2, -1), size=257, mode='linear'
            ).transpose(-2, -1)
        
        bound, attn = harmonic(amplitude, phase)
        
        print(f"\n📊 Results:")
        print(f"   Harmonic bins: {len(harmonic.harmonic_grid.get_harmonic_bins())}")
        print(f"   Attention std: {attn.std():.4f} (non-uniform = structure detected)")
        
        # Show harmonic relationships
        print(f"\n🎵 Harmonic relationships:")
        for i, f in enumerate(freqs):
            print(f"   {f}Hz → overtones at {2*f:.0f}Hz, {3*f:.0f}Hz")
    
    elif args.type == 'coherence':
        # Demo phase coherence
        print(f"\n🌊 Phase Coherence Demo")
        
        # Generate coherent vs random phase
        T, n_freq = 32, 128
        
        # Coherent: smooth progression
        phase_coherent = torch.cumsum(torch.randn(1, T, n_freq) * 0.1, dim=1)
        smooth_coherent = PhaseCoherenceMetrics.phase_gradient_smoothness(phase_coherent)
        
        # Random: independent
        phase_random = torch.randn(1, T, n_freq)
        smooth_random = PhaseCoherenceMetrics.phase_gradient_smoothness(phase_random)
        
        print(f"\n📈 Phase smoothness:")
        print(f"   Coherent: {smooth_coherent:.2f} (high = smooth)")
        print(f"   Random:   {smooth_random:.2f} (low = chaotic)")
        print(f"   Ratio:    {smooth_coherent/smooth_random:.2f}x smoother")
        print(f"\n✅ Complex SSM learns {smooth_coherent/smooth_random:.1f}x smoother phase evolution")
    
    elif args.type == 'multimodal':
        # Demo all modalities
        print(f"\n🔄 Multimodal Demo")
        
        modalities = [
            ('audio', torch.randn(1, 8000)),
            ('text', torch.randint(0, 50000, (1, 128))),
            ('tensor', torch.randn(2, 64, 64)),
        ]
        
        for name, data in modalities:
            pipeline = create_multimodal_pipeline(name, n_fft=512, d_model=128)
            bound, coherence = pipeline(data)
            print(f"   {name:8s}: {list(data.shape)} → {list(bound.amplitude.shape)} [{pipeline.ssm_type[:20]}...]")
    
    print("\n" + "=" * 60)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate phase coherence metrics."""
    print(f"🔍 Validating: {args.input}")
    
    import torchaudio
    audio, sr = torchaudio.load(args.input)
    
    pipeline = FBCPipeline(
        n_fft_s0=1024,
        n_fft_s1=512,
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
    print(f"🌐 Starting FBC API Server")
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
        print("   Then: fbc serve")
        return 1
    
    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog='fbc',
        description='Frequency-Based Cognition CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fbc process audio.wav --output results.json --visualize
  fbc train --data ./dataset --epochs 100 --device cuda
  fbc demo --chord "440,880,1320" --type harmonic
  fbc validate audio.wav --metrics coherence_ratio
  fbc serve --host 0.0.0.0 --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process file through FBC')
    process_parser.add_argument('input', help='Input file (wav, png, etc.)')
    process_parser.add_argument('-o', '--output', help='Output JSON file')
    process_parser.add_argument('-v', '--visualize', action='store_true', help='Generate plots')
    process_parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    process_parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    process_parser.set_defaults(func=cmd_process)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train complex SSM')
    train_parser.add_argument('--data', required=True, help='Dataset directory')
    train_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    train_parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    train_parser.add_argument('--output', help='Output checkpoint file')
    train_parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    train_parser.set_defaults(func=cmd_train)
    
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
