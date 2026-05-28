"""Bifröst CLI main entry point."""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore", message="mamba-ssm not available", category=UserWarning)
warnings.filterwarnings("ignore", message="SelectiveScanBlock stand-in", category=UserWarning)

from bifrost.cli import __version__


def cmd_demo(args: argparse.Namespace) -> int:
    """Run atomic demos."""
    import subprocess
    
    demo_scripts = {
        "1": "demo_1_antiphase.py",
        "2": "demo_2_harmonic_binding.py",
        "3": "demo_3_cross_modal_retrieval.py",
    }
    
    if args.demo_id == "all":
        ids = ["1", "2", "3"]
    else:
        ids = [args.demo_id]
    
    for demo_id in ids:
        if demo_id not in demo_scripts:
            print(f"Error: Unknown demo '{demo_id}'. Available: 1, 2, 3, all")
            return 1
        
        script = Path(__file__).parent.parent.parent.parent / "demos" / demo_scripts[demo_id]
        if not script.exists():
            print(f"\n⚠️  Demo script not found: {script}")
            print(f"\nTo run demos, ensure demo files are in the demos/ directory:")
            print(f"  cp dev-docs/demo_*.py demos/")
            print(f"\nOr run without demos:")
            print(f"  bifrost --help")
            return 1
        
        print(f"\n{'='*60}")
        print(f"Running Demo {demo_id}")
        print(f"{'='*60}\n")
        
        result = subprocess.run([sys.executable, str(script)], cwd=str(script.parent))
        if result.returncode != 0:
            print(f"\nDemo {demo_id} failed with exit code {result.returncode}")
            return result.returncode
    
    print(f"\n{'='*60}")
    print("All demos completed successfully!")
    print(f"{'='*60}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    import subprocess
    
    bench_scripts = {
        "attention": "bench_attention.py",
        "realistic": "bench_realistic.py",
    }
    
    if args.bench_name not in bench_scripts:
        print(f"Error: Unknown benchmark '{args.bench_name}'. Available: {list(bench_scripts.keys())}")
        return 1
    
    script = Path(__file__).parent.parent.parent.parent / "benchmarks" / bench_scripts[args.bench_name]
    if not script.exists():
        print(f"Error: Benchmark script not found: {script}")
        return 1
    
    print(f"Running benchmark: {args.bench_name}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(script.parent))
    return result.returncode


def cmd_process(args: argparse.Namespace) -> int:
    """Process audio through Bifröst pipeline."""
    import numpy as np
    import torch

    from bifrost.data import load_sample_audio
    from bifrost.data.loader import SAMPLES
    from bifrost.canonicalizer import SpectralCanonicalizer
    from bifrost.decomposer import SpectralDecomposer
    from bifrost.resonance_attention import SpectralBinding

    # Load audio
    if args.input in SAMPLES["audio"]:
        audio, sr = load_sample_audio(args.input)
        print(f"Loaded sample: {args.input} ({audio.shape} @ {sr} Hz)")
    else:
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(args.input)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            audio = torch.from_numpy(data)
            print(f"Loaded file: {args.input} ({audio.shape} @ {sr} Hz)")
        except Exception as e:
            print(f"Error loading audio: {e}")
            return 1

    n_fft = args.n_fft
    n_freq = n_fft // 2 + 1
    d_model = args.d_model
    n_frames = args.n_frames
    d_state = args.d_state
    expand = args.expand
    d_conv = args.d_conv

    canonicalizer = SpectralCanonicalizer(n_fft=n_fft)
    decomposer = SpectralDecomposer(
        n_fft=n_fft, n_scales=4, d_model=d_model, n_frames=n_frames,
        d_state=d_state, expand=expand, d_conv=d_conv
    )
    binding = SpectralBinding(d_model=d_model, n_heads=4, n_bands=8, dropout=0.0, n_freq_in=n_freq)
    canonicalizer.eval(); decomposer.eval(); binding.eval()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2 and audio.shape[1] <= 2:
        audio = audio.unsqueeze(0)

    print(f"Processing through Bifröst pipeline (SSM: {decomposer.ssm_type})...")
    with torch.no_grad():
        canonical = canonicalizer(audio, metadata={"sample_rate": float(sr)})
        decomposed = decomposer(canonical)
        bound, coherence = binding(decomposed)

    B, T, D = bound.amplitude.shape if bound.amplitude.dim() == 3 else (1, 1, bound.amplitude.shape[-1])
    attn_entropy = -(coherence * coherence.clamp(min=1e-8).log()).sum(dim=-1).mean()

    print(f"\nPipeline output:")
    print(f"  SSM type              : {decomposer.ssm_type}")
    print(f"  Time frames (T)       : {T}")
    print(f"  Spectral tensor shape : {bound.amplitude.shape}  (B, T, d_model)")
    print(f"  Attention map shape   : {coherence.shape}  (B, heads, T, T)")
    print(f"  Mean attention        : {coherence.mean():.4f}")
    print(f"  Max attention         : {coherence.max():.4f}")
    print(f"  Attention entropy     : {attn_entropy:.4f}  (higher = more distributed)")

    if args.output:
        torch.save({
            "spectral": bound,
            "attention": coherence,
            "sample_rate": sr,
        }, args.output)
        print(f"\nSaved results to: {args.output}")

    return 0


def cmd_samples(args: argparse.Namespace) -> int:
    """List available sample data."""
    from bifrost.data import list_samples

    AUDIO_DESC = {
        "mono_16khz":   "440 Hz tone, mono, 16 kHz, 2s",
        "mono_8khz":    "220 Hz tone, mono, 8 kHz, 2s",
        "stereo_44khz": "440+880 Hz tones, stereo, 44.1 kHz, 2s",
        "speech_synth": "Synthetic vowel /a/ (formant synthesis), 16 kHz, 2s",
        "music_chord":  "A-major chord (440+554+659 Hz), 44.1 kHz, 2s",
        "noise_pink":   "Pink noise (1/f spectrum), 16 kHz, 2s",
    }
    IMAGE_DESC = {
        "gray":         "16×16 grayscale gradient",
        "rgb":          "16×16 RGB color quadrants",
        "rgb_large":    "32×32 RGB color gradient",
        "spectrum_vis": "64×64 spectral visualization",
        "gradient_rgb": "64×64 smooth HSV-like gradient",
    }

    samples = list_samples()
    print("Available sample data:")
    print()
    print("Audio samples (use with: bifrost process <name>):")
    for name in samples["audio"]:
        desc = AUDIO_DESC.get(name, "")
        print(f"  {name:<16}  {desc}")
    print()
    print("Image samples:")
    for name in samples["image"]:
        desc = IMAGE_DESC.get(name, "")
        print(f"  {name:<16}  {desc}")
    print()
    print("Usage:")
    print("  bifrost process <sample_name>          # process bundled sample")
    print("  bifrost process myfile.wav -o out.pt   # process your own file")
    return 0


def cmd_attractors(args: argparse.Namespace) -> int:
    """Extract FrequencyAttractors from audio (S3)."""
    import numpy as np
    import torch

    from bifrost.data import load_sample_audio
    from bifrost.canonicalizer import SpectralCanonicalizer
    from bifrost.decomposer import SpectralDecomposer
    from bifrost.resonance_attention import SpectralBinding
    from bifrost.phase_lock_bridge import PhaseLockBridge

    from bifrost.data.loader import SAMPLES

    # Load audio
    if args.input in SAMPLES["audio"]:
        audio, sr = load_sample_audio(args.input)
        print(f"Loaded sample: {args.input} ({audio.shape} @ {sr} Hz)")
    else:
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(args.input)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            audio = torch.from_numpy(data)
            print(f"Loaded file: {args.input} ({audio.shape} @ {sr} Hz)")
        except Exception as e:
            print(f"Error loading audio: {e}")
            return 1

    # Build pipeline
    n_fft = args.n_fft
    n_freq = n_fft // 2 + 1
    d_model = args.d_model

    canonicalizer = SpectralCanonicalizer(n_fft=n_fft)
    decomposer = SpectralDecomposer(n_fft=n_fft, n_scales=4, d_model=n_freq)
    binding = SpectralBinding(d_model=d_model, n_heads=4, n_bands=args.n_bands, dropout=0.0)

    canonicalizer.eval()
    decomposer.eval()
    binding.eval()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2 and audio.shape[1] <= 2:
        audio = audio.unsqueeze(0)

    print(f"\nExtracting attractors...")
    with torch.no_grad():
        canonical = canonicalizer(audio, metadata={"sample_rate": float(sr)})
        decomposed = decomposer(canonical)
        bound, coherence = binding(decomposed)

        # Extract attractors from binding output
        attractors = PhaseLockBridge.extract_attractors_from_s2(
            bound, n_bands=args.n_bands, domain=args.domain, prefix=args.prefix
        )

    print(f"\nExtracted {len(attractors)} attractors:")
    print(f"  Domain: {args.domain}")
    print(f"  Bands per attractor: {args.n_bands}")
    print(f"  d_model: {attractors[0].d_model if attractors else 'N/A'}")
    print()

    for att in attractors[:args.max_display]:
        energy = att.spectral_energy()
        print(f"  {att.attractor_id}: stability={att.stability:.3f}, energy={energy:.4f}")

    if len(attractors) > args.max_display:
        print(f"  ... and {len(attractors) - args.max_display} more")

    if args.output:
        torch.save({
            "attractors": attractors,
            "spectral": bound,
            "sample_rate": sr,
            "n_bands": args.n_bands,
            "domain": args.domain,
        }, args.output)
        print(f"\nSaved {len(attractors)} attractors to: {args.output}")

    return 0


def cmd_bridge(args: argparse.Namespace) -> int:
    """Evaluate Phase-Lock Bridge between two attractor sets (S4)."""
    import torch
    from bifrost.phase_lock_bridge import PhaseLockBridge

    # Load attractor files
    try:
        data_a = torch.load(args.source)
        data_b = torch.load(args.target)
        attractors_a = data_a["attractors"]
        attractors_b = data_b["attractors"]
        print(f"Loaded {len(attractors_a)} attractors from {args.source}")
        print(f"Loaded {len(attractors_b)} attractors from {args.target}")
    except Exception as e:
        print(f"Error loading attractor files: {e}")
        print("Hint: Use 'bifrost attractors <audio>' to generate .pt files first")
        return 1

    # Initialize bridge
    bridge = PhaseLockBridge(
        n_bands=args.n_bands,
        min_locked_bands=args.min_locked,
        band_threshold=args.band_threshold,
        activation_threshold=args.activation_threshold,
    )
    bridge.eval()

    print(f"\nPhase-Lock Bridge evaluation:")
    print(f"  Min locked bands: {args.min_locked}")
    print(f"  Band threshold: {args.band_threshold}")
    print(f"  Activation threshold: {args.activation_threshold}")
    print()

    # Find all bridge candidates
    with torch.no_grad():
        candidates = bridge.find_bridges(attractors_a, attractors_b)

    # Display results
    activated = [c for c in candidates if c.is_activated]

    print(f"Evaluated {len(attractors_a) * len(attractors_b)} pairs")
    print(f"Found {len(activated)} activated bridges ({len(candidates)} candidates total)")
    print()

    if activated:
        print("Activated bridges (sorted by score):")
        print(f"{'Source':>12} → {'Target':>12} │ {'Score':>8} │ {'Locked':>6} │ {'Bands':>20}")
        print("─" * 70)
        for c in activated[:args.max_display]:
            band_str = ",".join(f"{b:.2f}" for b in c.band_coherences[:4]) + "..."
            print(f"{c.source.attractor_id:>12} → {c.target.attractor_id:>12} │ "
                  f"{c.activation_score:>8.4f} │ {c.n_locked_bands:>6} │ {band_str}")

        if len(activated) > args.max_display:
            print(f"\n... and {len(activated) - args.max_display} more activated bridges")

    # Summary statistics
    if candidates:
        scores = [c.activation_score for c in candidates]
        print(f"\nScore statistics:")
        print(f"  Mean: {sum(scores)/len(scores):.4f}")
        print(f"  Max:  {max(scores):.4f}")
        print(f"  Min:  {min(scores):.4f}")

    if args.output:
        torch.save({
            "candidates": candidates,
            "activated": activated,
            "config": {
                "n_bands": args.n_bands,
                "min_locked_bands": args.min_locked,
                "band_threshold": args.band_threshold,
                "activation_threshold": args.activation_threshold,
            },
        }, args.output)
        print(f"\nSaved bridge evaluation to: {args.output}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="bifrost",
        description="Bifröst — The Spectral Rainbow Bridge (Frequency-Based Cognition) CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bifrost demo 1                     # Run anti-phase discrimination demo
  bifrost demo all                   # Run all demos
  bifrost bench attention            # Run attention benchmark
  bifrost process mono_16khz         # Process sample audio (S0-S2)
  bifrost process myfile.wav -o out.pt  # Process file, save results
  bifrost samples                    # List available samples
  bifrost attractors mono_16khz -o att.pt   # Extract attractors (S3)
  bifrost bridge att_a.pt att_b.pt -o bridges.pt  # Phase-lock evaluation (S4)
        """
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run atomic demos")
    demo_parser.add_argument("demo_id", choices=["1", "2", "3", "all"], help="Demo to run")
    demo_parser.set_defaults(func=cmd_demo)
    
    # Benchmark command
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument("bench_name", choices=["attention", "realistic"], help="Benchmark to run")
    bench_parser.set_defaults(func=cmd_bench)
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process audio through Bifröst pipeline")
    process_parser.add_argument("input", help="Audio file or sample name (e.g., 'mono_16khz')")
    process_parser.add_argument("-o", "--output", help="Output file for results (.pt)")
    process_parser.add_argument("--n-fft", type=int, default=1024, help="FFT size (default: 1024)")
    process_parser.add_argument("--d-model", type=int, default=128, help="Model dimension (default: 128)")
    process_parser.add_argument("--n-frames", type=int, default=32, help="Number of time frames for SSM (default: 32)")
    process_parser.add_argument("--d-state", type=int, default=16, help="SSM state size (default: 16)")
    process_parser.add_argument("--expand", type=int, default=2, help="SSM expansion factor (default: 2)")
    process_parser.add_argument("--d-conv", type=int, default=4, help="SSM conv kernel size (default: 4)")
    process_parser.set_defaults(func=cmd_process)
    
    # Samples command
    samples_parser = subparsers.add_parser("samples", help="List available sample data")
    samples_parser.set_defaults(func=cmd_samples)

    # Attractors command (S3)
    att_parser = subparsers.add_parser("attractors", help="Extract FrequencyAttractors from audio (S3)")
    att_parser.add_argument("input", help="Audio file or sample name (e.g., 'mono_16khz')")
    att_parser.add_argument("-o", "--output", help="Output file for attractors (.pt)")
    att_parser.add_argument("-n", "--n-bands", type=int, default=8, help="Number of spectral bands (default: 8)")
    att_parser.add_argument("--n-fft", type=int, default=1024, help="FFT size (default: 1024)")
    att_parser.add_argument("--d-model", type=int, default=128, help="Model dimension (default: 128)")
    att_parser.add_argument("--domain", default="audio", help="Domain label (default: audio)")
    att_parser.add_argument("--prefix", default="att", help="Attractor ID prefix (default: att)")
    att_parser.add_argument("--max-display", type=int, default=10, help="Max attractors to display (default: 10)")
    att_parser.set_defaults(func=cmd_attractors)

    # Bridge command (S4)
    bridge_parser = subparsers.add_parser("bridge", help="Evaluate Phase-Lock Bridge between attractor sets (S4)")
    bridge_parser.add_argument("source", help="Source attractor file (.pt)")
    bridge_parser.add_argument("target", help="Target attractor file (.pt)")
    bridge_parser.add_argument("-o", "--output", help="Output file for bridge results (.pt)")
    bridge_parser.add_argument("-n", "--n-bands", type=int, default=8, help="Number of spectral bands (default: 8)")
    bridge_parser.add_argument("--min-locked", type=int, default=3, help="Min locked bands threshold (default: 3)")
    bridge_parser.add_argument("--band-threshold", type=float, default=0.5, help="Per-band coherence threshold (default: 0.5)")
    bridge_parser.add_argument("--activation-threshold", type=float, default=0.6, help="Activation score threshold (default: 0.6)")
    bridge_parser.add_argument("--max-display", type=int, default=10, help="Max bridges to display (default: 10)")
    bridge_parser.set_defaults(func=cmd_bridge)

    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
