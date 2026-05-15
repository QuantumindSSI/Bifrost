"""Example: Audio ingest and canonicalization."""

from spectral_encoder.ingest.pipeline import IngestPipeline, Modality
import numpy as np
from scipy.io import wavfile
import io


def example_synthetic_audio():
    """Generate and ingest synthetic audio."""
    print("=" * 60)
    print("EXAMPLE 1: Synthetic Audio Ingestion")
    print("=" * 60)
    
    # Generate synthetic audio (1 second, 16kHz, 440 Hz sine wave)
    sr = 16000
    duration = 1.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    frequency = 440  # A4 note
    audio_signal = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    # Encode to WAV bytes
    bio = io.BytesIO()
    wavfile.write(bio, sr, audio_signal)
    wav_bytes = bio.getvalue()
    
    # Ingest through pipeline
    pipeline = IngestPipeline(strict_validation=False)
    audio, metadata = pipeline.ingest(wav_bytes, Modality.AUDIO, "wav")
    
    print(f"✓ Decoded WAV: {metadata['num_samples']} samples")
    print(f"  Sample rate: {metadata['sample_rate']} Hz")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Duration: {metadata['duration_sec']:.2f}s")
    print(f"  Audio dtype: {audio.dtype}")
    print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    print()


def example_image():
    """Generate and ingest synthetic image."""
    print("=" * 60)
    print("EXAMPLE 2: Synthetic Image Ingestion")
    print("=" * 60)
    
    try:
        from PIL import Image
    except ImportError:
        print("❌ Pillow not installed; skipping image example")
        return
    
    # Create a simple RGB gradient image
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        img_array[i, :, 0] = int(255 * i / 224)  # Red gradient
        img_array[:, i, 1] = int(255 * i / 224)  # Green gradient
    img_array[:, :, 2] = 128  # Blue constant
    
    # Encode to PNG bytes
    img = Image.fromarray(img_array, mode="RGB")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    png_bytes = bio.getvalue()
    
    # Ingest through pipeline
    pipeline = IngestPipeline(strict_validation=False)
    image, metadata = pipeline.ingest(png_bytes, Modality.IMAGE, "png")
    
    print(f"✓ Decoded PNG: {metadata['width']}×{metadata['height']}")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Bit depth: {metadata['bit_depth']}")
    print(f"  Color space: {metadata['color_space']}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print()


def example_stereo_audio():
    """Ingest stereo audio."""
    print("=" * 60)
    print("EXAMPLE 3: Stereo Audio Ingestion")
    print("=" * 60)
    
    # Generate stereo: left=440Hz, right=880Hz
    sr = 16000
    duration = 2.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    stereo = np.column_stack([left, right])
    
    # Encode to WAV
    bio = io.BytesIO()
    wavfile.write(bio, sr, stereo)
    wav_bytes = bio.getvalue()
    
    # Ingest
    pipeline = IngestPipeline(strict_validation=False)
    audio, metadata = pipeline.ingest(wav_bytes, Modality.AUDIO, "wav")
    
    print(f"✓ Decoded Stereo WAV:")
    print(f"  Shape: {audio.shape}")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Sample rate: {metadata['sample_rate']} Hz")
    print(f"  Duration: {metadata['duration_sec']:.2f}s")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPECTRAL ENCODER: PHASE 1 INGESTION EXAMPLES")
    print("=" * 60 + "\n")
    
    example_synthetic_audio()
    example_image()
    example_stereo_audio()
    
    print("=" * 60)
    print("All examples completed successfully! ✓")
    print("=" * 60)
