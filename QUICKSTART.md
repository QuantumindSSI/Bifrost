# Phase 1: Spectral Encoder Data Ingestion — QUICKSTART

## Overview

**Phase 1 delivers:** Audio/image deserialization, validation, and normalization.

- ✅ **Audio Decoder** (WAV, MP3, FLAC, OGG → float32 in [-1, 1])
- ✅ **Image Decoder** (PNG, JPG, TIFF, BMP → float32 in [0, 1])
- ✅ **Validation** (schema checks, NaN/Inf detection)
- ✅ **Normalization** (type conversion, scaling)
- ✅ **Pipeline** (unified entry point)
- ✅ **Tests** (unit + integration, 90%+ coverage target)

**Status:** Development-ready. 16 modules, 27 test cases.

---

## Installation

### Option 1: Development Install

```bash
pip install -e ".[dev]"
```

### Option 2: Production Install

```bash
pip install -r requirements.txt
```

---

## Quick Test

```bash
# Run all tests
pytest spectral_encoder/tests/ -v

# Run with coverage
pytest spectral_encoder/tests/ --cov=spectral_encoder --cov-report=html

# Run specific test file
pytest spectral_encoder/tests/unit/test_decoders.py -v
```

**Expected result:** All tests pass (green ✓).

---

## Basic Usage

### Example 1: Decode & Normalize Audio

```python
from spectral_encoder.ingest.pipeline import IngestPipeline, Modality

# Initialize pipeline
pipeline = IngestPipeline(strict_validation=False)

# Load WAV file
with open("song.wav", "rb") as f:
    wav_bytes = f.read()

# Ingest through pipeline
audio, metadata = pipeline.ingest(wav_bytes, Modality.AUDIO, "wav")

# Output: audio is float32 in [-1.0, 1.0]
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {metadata['sample_rate']} Hz")
print(f"Duration: {metadata['duration_sec']:.2f}s")
```

### Example 2: Decode & Normalize Image

```python
from spectral_encoder.ingest.pipeline import IngestPipeline, Modality

pipeline = IngestPipeline(strict_validation=False)

# Load PNG file
with open("image.png", "rb") as f:
    png_bytes = f.read()

# Ingest through pipeline
image, metadata = pipeline.ingest(png_bytes, Modality.IMAGE, "png")

# Output: image is float32 in [0.0, 1.0]
print(f"Image shape: {image.shape}")
print(f"Dimensions: {metadata['width']}×{metadata['height']}")
print(f"Channels: {metadata['channels']}")
```

### Example 3: Batch Ingest

```python
from spectral_encoder.ingest.pipeline import IngestPipeline, Modality
import glob

pipeline = IngestPipeline(strict_validation=False)

# Get all WAV files
paths = glob.glob("data/audio/*.wav")

# Batch ingest
results = pipeline.batch_ingest(paths, Modality.AUDIO)

for audio, metadata in results:
    print(f"✓ Loaded: {metadata['num_samples']} samples @ {metadata['sample_rate']} Hz")
```

---

## Module Architecture

```
spectral_encoder/
├── ingest/
│   ├── decoders/
│   │   ├── base.py          # Decoder ABC
│   │   ├── audio.py         # AudioDecoder (WAV, MP3, FLAC, OGG)
│   │   └── image.py         # ImageDecoder (PNG, JPG, TIFF, BMP)
│   ├── validation/
│   │   ├── exceptions.py    # Custom exceptions
│   │   ├── audio.py         # AudioValidator
│   │   └── image.py         # ImageValidator
│   ├── normalize.py         # Type conversion & scaling
│   ├── pipeline.py          # Unified IngestPipeline
│   └── __init__.py
├── tests/
│   ├── conftest.py          # Fixtures
│   ├── unit/
│   │   ├── test_decoders.py
│   │   ├── test_validation.py
│   │   └── test_normalization.py
│   └── integration/
│       └── test_integration.py
├── examples/
│   └── basic_ingest.py
└── __init__.py
```

---

## API Reference

### IngestPipeline

**Constructor:**
```python
pipeline = IngestPipeline(
    strict_validation=False,  # False=warn; True=reject
    repair_on_error=True      # Attempt to fix data issues
)
```

**Methods:**

#### `ingest(data: bytes, modality: Modality, format_str: str) → Tuple[ndarray, Dict]`
Ingest raw bytes through complete pipeline.
- **Args:** raw bytes, modality (AUDIO/IMAGE), format ("wav", "png", etc.)
- **Returns:** (normalized array, metadata dict)
- **Raises:** DecodingError, ValidationError (strict mode)

#### `ingest_from_file(file_path: str, modality: Modality) → Tuple[ndarray, Dict]`
Ingest from file path. Format auto-detected from extension.

#### `batch_ingest(file_paths: List[str], modality: Modality) → List[Tuple[ndarray, Dict]]`
Ingest multiple files. Skips failures, returns successful results.

---

### AudioDecoder

```python
from spectral_encoder.ingest.decoders.audio import AudioDecoder

decoder = AudioDecoder()

# Supported formats
assert decoder.supports_format("wav")
assert decoder.supports_format("mp3")
assert decoder.supports_format("flac")
assert decoder.supports_format("ogg")

# Decode
audio, metadata = decoder.decode(wav_bytes, "wav")
# audio: float32, shape (channels, samples) or (samples,)
# metadata: dict with sample_rate, channels, duration_sec, etc.
```

**Metadata Dict:**
```python
{
    "format": "wav",
    "sample_rate": 16000,
    "bit_depth": 16,
    "channels": 1,
    "num_samples": 16000,
    "duration_sec": 1.0,
}
```

---

### ImageDecoder

```python
from spectral_encoder.ingest.decoders.image import ImageDecoder

decoder = ImageDecoder()

# Supported formats
assert decoder.supports_format("png")
assert decoder.supports_format("jpg")
assert decoder.supports_format("tiff")

# Decode
image, metadata = decoder.decode(png_bytes, "png")
# image: float32, shape (H, W) or (H, W, C)
# metadata: dict with width, height, channels, color_space, etc.
```

**Metadata Dict:**
```python
{
    "format": "png",
    "width": 224,
    "height": 224,
    "channels": 3,
    "bit_depth": 8,
    "color_space": "rgb",
    "size_bytes": 12345,
}
```

---

## Validation Modes

### Strict Mode (Production)

```python
pipeline = IngestPipeline(strict_validation=True)

# Rejects invalid data, raises ValidationError
audio, _ = pipeline.ingest(wav_bytes, Modality.AUDIO, "wav")
```

### Lenient Mode (Development)

```python
pipeline = IngestPipeline(strict_validation=False)

# Warns on invalid data, attempts repair, continues
audio, _ = pipeline.ingest(wav_bytes, Modality.AUDIO, "wav")
# Prints: ⚠️  Audio validation warning: ...
```

---

## Constraints & Limits

### Audio

| Property | Min | Max |
|----------|-----|-----|
| Sample rate | 8 kHz | 48 kHz |
| Bit depth | 8, 16, 24, 32 bits |
| Channels | 1–8 |
| Duration | 10 ms | 3600 s (1 hour) |

### Image

| Property | Min | Max |
|----------|-----|-----|
| Width | 16 px | 8192 px |
| Height | 16 px | 8192 px |
| Channels | 1 (gray), 3 (RGB), 4 (RGBA) |
| Bit depth | 8, 16 bits |

---

## Normalization

**Audio:** All output is float32 in **[-1.0, 1.0]**
- int16 → ÷ 32768
- uint8 → (x - 128) / 128
- float32 → clip to [-1, 1]

**Image:** All output is float32 in **[0.0, 1.0]**
- uint8 → ÷ 255
- uint16 → ÷ 65535
- float32 → clip to [0, 1]

---

## Troubleshooting

### "DecodingError: Failed to decode wav"
- File may be corrupted. Check file size > 44 bytes.
- If compressed (MP3), ensure `librosa` installed: `pip install librosa`

### "ValidationError: Audio is invalid"
- Duration too short? Minimum 10 ms.
- Sample rate out of range? Must be 8–48 kHz.
- Contains NaN/Inf? Check source signal.
- Use `strict_validation=False` to see warnings instead.

### "ImportError: No module named 'PIL'"
- Install Pillow: `pip install Pillow`

### "ImportError: No module named 'librosa'"
- Install librosa: `pip install librosa`

---

## Next Steps (Phase 2)

- [ ] Text deserialization (CSV, JSON, Parquet)
- [ ] Tensor loading (NPZ, HDF5, Zarr)
- [ ] Buffering & rate control (RingBuffer)
- [ ] Source connectors (file watcher, HTTP, Kafka)
- [ ] Error recovery (dead-letter queue, retry logic)

---

## References

- **Plan:** `spectral_encoder_data_ingestion_plan.md`
- **Language Analysis:** `language_analysis.md`
- **FBC Architecture:** `README.md` (Phase 1 deliverables)

---

## Support

For issues or questions, refer to:
- Test cases: `spectral_encoder/tests/`
- Examples: `spectral_encoder/examples/basic_ingest.py`
- Architecture: `spectral_encoder_data_ingestion_plan.md`

**Version:** 0.1.0 (Alpha)  
**Last Updated:** May 2026  
**Status:** Development-Ready, Pre-Production
