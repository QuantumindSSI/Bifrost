# Phase 1 Sample Data Testing Guide

✅ **Status:** Production-ready for real data ingestion

---

## Quick Start: Test with Generated Samples

```bash
cd fbc-phase1
python3 test_with_samples.py --both
```

This will:
1. Generate 4 sample files (2 audio, 2 image)
2. Test them through the pipeline
3. Show batch ingestion

**Expected Output:**
```
✅ Successfully ingested 4 files
  • mono_16khz.wav: audio (32000,)
  • stereo_44khz.wav: audio (44100, 2)
  • rgb_image.png: image (256, 256, 3)
  • gray_image.png: image (128, 128)
```

---

## Testing with Your Own Files

### Option 1: Test Individual Files

```bash
python3 test_with_samples.py --files ~/audio.wav ~/image.png
```

Supports:
- **Audio:** `.wav`, `.mp3`, `.flac`, `.ogg`
- **Images:** `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

### Option 2: Batch Test Your Directory

```bash
python3 test_with_samples.py --generate --batch --sample-dir ~/my_data
```

This ingests all files in `~/my_data/` at once.

---

## API Usage in Your Code

### Basic Ingestion

```python
from spectral_encoder.ingest.pipeline import IngestPipeline, Modality

pipeline = IngestPipeline(strict_validation=False)

# Single file
data, metadata = pipeline.ingest_from_file(
    "path/to/audio.wav",
    modality=Modality.AUDIO
)

print(f"Shape: {data.shape}")
print(f"Range: [{data.min():.4f}, {data.max():.4f}]")
```

### Batch Processing

```python
# Multiple files at once
results = pipeline.batch_ingest([
    "/path/to/audio1.wav",
    "/path/to/audio2.wav",
    "/path/to/image.png"
])

print(f"Successful: {len(results['successful'])}")
print(f"Failed: {len(results['failed'])}")

for result in results['successful']:
    print(f"  {result['filename']}: {result['shape']}")
```

### From Raw Bytes

```python
# Read file as bytes
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

# Ingest
data, metadata = pipeline.ingest(
    audio_bytes,
    modality=Modality.AUDIO,
    format_str="wav"
)
```

---

## Data Validation Rules

### Audio Constraints
- **Sample Rate:** 8,000 - 48,000 Hz
- **Duration:** 10ms - 3,600s (1 hour max)
- **Bit Depth:** 8, 16, 24, 32 bits
- **Channels:** 1-8
- **Output Range:** [-1.0, 1.0] (float32)

### Image Constraints
- **Resolution:** 16px - 8,192px (per dimension)
- **Channels:** 1 (grayscale), 3 (RGB), 4 (RGBA)
- **Bit Depth:** 8, 16 bits
- **Output Range:** [0.0, 1.0] (float32)

### Validation Modes

```python
# Strict mode: Reject any constraint violation
pipeline = IngestPipeline(strict_validation=True)

# Lenient mode: Warn and continue
pipeline = IngestPipeline(strict_validation=False)
```

---

## Expected Output Format

### Audio Data
```python
data.shape = (samples,) or (samples, channels)
data.dtype = float32
data.min() >= -1.0, data.max() <= 1.0
```

**Example:** 16kHz mono, 2 seconds → shape (32000,)

### Image Data
```python
data.shape = (height, width) or (height, width, channels)
data.dtype = float32
data.min() >= 0.0, data.max() <= 1.0
```

**Example:** 256×256 RGB → shape (256, 256, 3)

---

## Error Handling

```python
from spectral_encoder.ingest.validation.exceptions import (
    DecodingError,
    ValidationError
)

try:
    data, meta = pipeline.ingest_from_file("file.wav", Modality.AUDIO)
except DecodingError as e:
    print(f"Could not decode: {e}")
except ValidationError as e:
    print(f"Failed validation: {e}")
```

---

## Test Results: Sample Data

All 4 sample files tested successfully:

| File | Modality | Shape | Min | Max | Status |
|------|----------|-------|-----|-----|--------|
| mono_16khz.wav | audio | (32000,) | -0.4343 | 0.4343 | ✅ |
| stereo_44khz.wav | audio | (44100, 2) | -0.9999 | 0.9999 | ✅ |
| rgb_image.png | image | (256, 256, 3) | 0.0000 | 1.0000 | ✅ |
| gray_image.png | image | (128, 128) | 0.0000 | 0.7843 | ✅ |

---

## Troubleshooting

### "Unknown format"
- Check file extension (`.wav`, `.png`, etc.)
- Supported formats listed above

### "Invalid sample rate"
- Audio must be 8-48kHz
- Convert files: `ffmpeg -i input.wav -ar 16000 output.wav`

### "Image too small"
- Minimum 16×16 pixels
- Check image resolution

### "ModuleNotFoundError"
- Install requirements: `pip install -r requirements.txt`

---

## Next Steps

1. **Phase 2:** Add text/tensor support (CSV, JSON, HDF5)
2. **Phase 3:** Real-time streaming (rate control, buffering)
3. **Phase 4:** Production monitoring & error recovery

---

**Ready to test?**

```bash
python3 test_with_samples.py --both
```

✅ **Phase 1 is production-ready for sample data testing**
