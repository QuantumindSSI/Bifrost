#!/usr/bin/env python3
"""
Phase 1 Sample Data Testing Script
====================================

Demonstrates how to:
1. Test with generated sample files (WAV, PNG)
2. Test with your own local files
3. Batch ingest multiple files
4. Handle errors gracefully

Usage:
    python test_with_samples.py --generate    # Create sample files
    python test_with_samples.py --test-samples # Test samples
    python test_with_samples.py --files file1 file2  # Test your files
    python test_with_samples.py --both        # All of the above
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spectral_encoder.ingest.pipeline import IngestPipeline, Modality
from spectral_encoder.ingest.validation.exceptions import ValidationError, DecodingError
import numpy as np
from scipy.io import wavfile
from PIL import Image


def create_sample_files(output_dir="./sample_data"):
    """Create sample audio and image files for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENERATING SAMPLE DATA FILES")
    print(f"{'='*70}\n")
    
    # Sample 1: Mono audio (16kHz, 2 seconds)
    print("  [1/4] Creating sample mono audio (mono_16khz.wav)...")
    sr = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    # 440 Hz sine wave + 880 Hz sine wave
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t))
    audio_int16 = np.int16(audio * 32767)
    wavfile.write(f"{output_dir}/mono_16khz.wav", sr, audio_int16)
    
    # Sample 2: Stereo audio (44.1kHz, 1 second)
    print("  [2/4] Creating sample stereo audio (stereo_44khz.wav)...")
    sr = 44100
    duration = 1
    t = np.linspace(0, duration, int(sr * duration))
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 660 * t)
    stereo = np.column_stack([left, right])
    stereo_int16 = np.int16(stereo * 32767)
    wavfile.write(f"{output_dir}/stereo_44khz.wav", sr, stereo_int16)
    
    # Sample 3: RGB image (256x256)
    print("  [3/4] Creating sample RGB image (rgb_image.png)...")
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[64:192, 64:192, 0] = 255  # Red square
    rgb[96:160, 96:160, 1] = 255  # Green square (overlaps)
    Image.fromarray(rgb).save(f"{output_dir}/rgb_image.png")
    
    # Sample 4: Grayscale image (128x128)
    print("  [4/4] Creating sample grayscale image (gray_image.png)...")
    gray = np.zeros((128, 128), dtype=np.uint8)
    gray[32:96, 32:96] = 200  # Light gray square
    Image.fromarray(gray).save(f"{output_dir}/gray_image.png")
    
    print(f"\n✅ Sample files created in: {output_dir}/\n")
    return output_dir


def detect_modality(filepath):
    """Auto-detect modality from file extension."""
    ext = Path(filepath).suffix.lower()
    if ext in ['.wav', '.mp3', '.flac', '.ogg']:
        return Modality.AUDIO
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return Modality.IMAGE
    return None


def test_sample_files(sample_dir="./sample_data"):
    """Test the pipeline with generated sample files."""
    print(f"\n{'='*70}")
    print("TESTING WITH SAMPLE FILES")
    print(f"{'='*70}\n")
    
    pipeline = IngestPipeline(strict_validation=False)
    
    files = {
        "Audio (mono, 16kHz)": f"{sample_dir}/mono_16khz.wav",
        "Audio (stereo, 44.1kHz)": f"{sample_dir}/stereo_44khz.wav",
        "Image (RGB)": f"{sample_dir}/rgb_image.png",
        "Image (Grayscale)": f"{sample_dir}/gray_image.png",
    }
    
    results = {}
    
    for label, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"  ❌ {label}")
            print(f"     File not found: {filepath}\n")
            continue
        
        try:
            modality = detect_modality(filepath)
            if not modality:
                print(f"  ❌ {label} - Unknown format\n")
                continue
            
            print(f"  ✓ Testing: {label}")
            data, metadata = pipeline.ingest_from_file(filepath, modality)
            
            # Display result details
            print(f"    • Modality: {modality.value}")
            print(f"    • Dtype: {data.dtype}")
            print(f"    • Shape: {data.shape}")
            
            if modality == Modality.AUDIO:
                print(f"    • Min: {data.min():.4f}, Max: {data.max():.4f}")
                print(f"    • Range check: {data.min() >= -1.0 and data.max() <= 1.0} ✓")
            elif modality == Modality.IMAGE:
                print(f"    • Min: {data.min():.4f}, Max: {data.max():.4f}")
                print(f"    • Range check: {data.min() >= 0.0 and data.max() <= 1.0} ✓")
            
            results[label] = {"status": "PASS", "shape": str(data.shape)}
            print()
            
        except (ValidationError, DecodingError) as e:
            print(f"    ❌ Error: {e}\n")
            results[label] = {"status": "FAIL", "error": str(e)}
    
    return results


def test_your_files(file_list):
    """Test the pipeline with user-provided files."""
    print(f"\n{'='*70}")
    print("TESTING WITH YOUR FILES")
    print(f"{'='*70}\n")
    
    pipeline = IngestPipeline(strict_validation=False)
    results = {}
    
    for filepath in file_list:
        filepath = os.path.expanduser(filepath)
        
        if not os.path.exists(filepath):
            print(f"  ❌ File not found: {filepath}\n")
            results[filepath] = {"status": "FAIL", "error": "File not found"}
            continue
        
        try:
            modality = detect_modality(filepath)
            if not modality:
                print(f"  ❌ {os.path.basename(filepath)} - Unknown format\n")
                results[filepath] = {"status": "FAIL", "error": "Unknown format"}
                continue
            
            print(f"  ✓ Testing: {os.path.basename(filepath)}")
            data, metadata = pipeline.ingest_from_file(filepath, modality)
            
            print(f"    • Modality: {modality.value}")
            print(f"    • Shape: {data.shape}")
            print(f"    • Dtype: {data.dtype}")
            print(f"    • Data min/max: [{data.min():.4f}, {data.max():.4f}]")
            print()
            
            results[filepath] = {
                "status": "PASS",
                "modality": modality.value,
                "shape": str(data.shape)
            }
            
        except (ValidationError, DecodingError) as e:
            print(f"    ❌ Error: {e}\n")
            results[filepath] = {"status": "FAIL", "error": str(e)}
    
    return results


def batch_test(sample_dir="./sample_data"):
    """Test batch ingestion."""
    print(f"\n{'='*70}")
    print("BATCH INGESTION TEST")
    print(f"{'='*70}\n")
    
    pipeline = IngestPipeline(strict_validation=False)
    
    # Get all files from sample directory
    files = list(Path(sample_dir).glob("*.*"))
    
    if not files:
        print("  ⚠️  No files found in sample directory\n")
        return {}
    
    print(f"  Batch ingesting {len(files)} files...\n")
    
    successful = []
    failed = []
    
    for filepath in files:
        modality = detect_modality(str(filepath))
        if not modality:
            failed.append({"filename": filepath.name, "error": "Unknown format"})
            continue
        
        try:
            data, metadata = pipeline.ingest_from_file(str(filepath), modality)
            successful.append({
                "filename": filepath.name,
                "modality": modality.value,
                "shape": data.shape
            })
        except Exception as e:
            failed.append({"filename": filepath.name, "error": str(e)})
    
    print(f"  ✅ Successfully ingested {len(successful)} files")
    if failed:
        print(f"  ⚠️  Failed {len(failed)} files\n")
    
    print("  Successful:")
    for result in successful:
        print(f"    • {result['filename']}: {result['modality']} {result['shape']}")
    
    if failed:
        print("\n  Failed:")
        for error in failed:
            print(f"    • {error['filename']}: {error['error']}")
    
    print()
    return {"successful": successful, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 with sample data")
    parser.add_argument("--generate", action="store_true", help="Generate sample files")
    parser.add_argument("--test-samples", action="store_true", help="Test with samples")
    parser.add_argument("--files", nargs="+", help="Test specific files (e.g., --files audio.wav image.png)")
    parser.add_argument("--batch", action="store_true", help="Test batch ingestion")
    parser.add_argument("--both", action="store_true", help="Generate samples AND test with them")
    parser.add_argument("--sample-dir", default="./sample_data", help="Sample data directory")
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any([args.generate, args.test_samples, args.files, args.batch, args.both]):
        parser.print_help()
        return
    
    # Handle --both flag
    if args.both:
        args.generate = True
        args.test_samples = True
        args.batch = True
    
    sample_dir = args.sample_dir
    
    # Generate samples if requested
    if args.generate:
        sample_dir = create_sample_files(sample_dir)
    
    # Test with generated samples
    if args.test_samples and os.path.exists(sample_dir):
        test_sample_files(sample_dir)
    
    # Test with user files
    if args.files:
        test_your_files(args.files)
    
    # Batch test
    if args.batch and os.path.exists(sample_dir):
        batch_test(sample_dir)
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
