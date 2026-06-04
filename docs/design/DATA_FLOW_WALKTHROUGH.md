# Data Ingestion Flow - Complete Walkthrough

**Purpose:** Understand how raw files (audio, images) flow through the Bifrost pipeline to become standardized float32 tensors.

---

## Quick Overview

```
Raw File (bytes)
    ↓
[DECODER]       Extract format-specific data (int16/uint8)
    ↓
[VALIDATOR]     Check schema & constraints (pass/fail)
    ↓
[NORMALIZER]    Convert types & scale to canonical ranges
    ↓
Standardized float32 Tensor (ready for ML models)
```

---

## Step 1: Decoder - Extract Raw Data

### Audio Decoder

**Input:** WAV/MP3/FLAC/OGG bytes

**Process:**
1. Parse audio format header (sample rate, channels, bit depth)
2. Extract audio samples as integers (int16, int24, int32)
3. Stack channels if stereo/surround

**Output:** NumPy int16 array

**Example:** mono_16khz.wav
```
File: mono_16khz.wav (64 KB)
  Sample rate: 16,000 Hz
  Channels: 1 (mono)
  Bit depth: 16-bit
  Duration: 2 seconds
  Total samples: 32,000

Output array:
  Shape: (32000,)
  Dtype: int16
  Range: [-32768, 32767]
  Data: [2048, 4096, 6144, 8192, ...]
```

### Image Decoder

**Input:** PNG/JPG/TIFF/BMP bytes

**Process:**
1. Parse image format header (resolution, channels, bit depth)
2. Read pixel data as uint8/uint16
3. Stack RGB channels if present

**Output:** NumPy uint8 array

**Example:** rgb_image.png
```
File: rgb_image.png (0.5 KB)
  Width: 256 pixels
  Height: 256 pixels
  Channels: 3 (RGB)
  Bit depth: 8-bit
  
Output array:
  Shape: (256, 256, 3)
  Dtype: uint8
  Range: [0, 255]
  Data: [[[255, 0, 0],      (red)
          [0, 255, 0],      (green)
          [255, 255, 0]], ...] (yellow)
```

---

## Step 2: Validator - Check Constraints

### Audio Validator

Checks that audio meets these constraints:

| Constraint | Min | Max | Notes |
|-----------|-----|-----|-------|
| Sample Rate | 8 kHz | 48 kHz | Telephony to professional |
| Duration | 10 ms | 3600 s | 1 hour max |
| Channels | 1 | 8 | Mono to 7.1 surround |
| Bit Depth | 8 bits | 32 bits | All standard depths |

Additional checks:
- ✓ No NaN/Inf values
- ✓ Correct shape (1D for mono, 2D for multi-channel)

**Example validation (mono_16khz.wav):**

Input from decoder:
```
shape=(32000,), dtype=int16, sample_rate=16000Hz
```

Checks performed:
```
✓ Shape (32000,) is 1D?           YES (mono)
✓ Sample rate 16000 Hz?           YES (8-48kHz valid)
✓ Duration = 32000/16000 = 2s?    YES (10ms-3600s valid)
✓ Bit depth 16?                   YES (8/16/24/32 valid)
✓ Has NaN/Inf?                    NO
```

Result: **✓ PASS** → Proceed to normalization

**Validation failure example:**

Input sample rate: 4000 Hz

Check: `4000 in [8000, 48000]`?

Result: **✗ FAIL** → Reject this audio (too low sample rate)

### Image Validator

Checks that images meet these constraints:

| Constraint | Min | Max | Notes |
|-----------|-----|-----|-------|
| Width | 16 px | 8192 px | Thumbnail to 8K |
| Height | 16 px | 8192 px | Thumbnail to 8K |
| Channels | 1 | 4 | Grayscale, RGB, RGBA |
| Bit Depth | 8 bits | 16 bits | Standard depths |

Additional checks:
- ✓ No NaN/Inf values
- ✓ Correct shape (2D for grayscale, 3D for color)

---

## Step 3: Normalizer - Type Conversion & Scaling

### Audio Normalization

**Input:** int16 array, range [-32768, 32767]

**Process:**
```python
audio_float32 = audio_int16 / 32768.0
```

**Output:** float32 array, range [-1.0, 1.0]

**Why 32768 and not 32767?**
- int16 min: -32768
- int16 max: +32767 (asymmetric!)
- Dividing by 32768 makes range symmetric: [-1.0, 1.0]
- Industry standard in audio processing

**Example calculation:**
```
int16 value:    2048
Convert:        2048.0 / 32768.0
Result:         0.0625 (float32)
✓ In range [-1.0, 1.0]
```

**Array transformation:**
```
Before:  [2048, 4096, 6144, 8192, ...] (int16)
÷32768:  [0.0625, 0.125, 0.1875, 0.25, ...] (float32)
After:   [0.0625, 0.125, 0.1875, 0.25, ...] (canonical)
```

### Image Normalization

**Input:** uint8 array, range [0, 255]

**Process:**
```python
image_float32 = image_uint8 / 255.0
```

**Output:** float32 array, range [0.0, 1.0]

**Why 255 and not 256?**
- uint8 min: 0
- uint8 max: 255
- Dividing by 255 maps full range to [0.0, 1.0]
- Industry standard in image processing

**Example calculation:**
```
uint8 value:    128
Convert:        128.0 / 255.0
Result:         0.502 (float32)
✓ In range [0.0, 1.0]

uint8 value:    255
Convert:        255.0 / 255.0
Result:         1.0 (float32)
✓ At boundary
```

**Array transformation (RGB):**
```
Before:  [[[255, 0, 0],      [0, 255, 0],      [255, 255, 0]]] (uint8)
÷255:    [[[1.0, 0.0, 0.0],  [0.0, 1.0, 0.0],  [1.0, 1.0, 0.0]]] (float32)
After:   [[[1.0, 0.0, 0.0],  [0.0, 1.0, 0.0],  [1.0, 1.0, 0.0]]] (canonical)
```

---

## Complete Example 1: mono_16khz.wav

### Step 1: INPUT
```
File:   mono_16khz.wav
Specs:  16-bit PCM WAV
Rate:   16,000 Hz
Channels: 1 (mono)
Duration: 2 seconds
Size:   64 KB
```

### Step 2: DECODER
```
Method:  scipy.io.wavfile.read()
Output:  array([2048, 4096, 6144, ...], dtype=int16)
Shape:   (32000,)
Range:   [-32768, 32767]
```

### Step 3: VALIDATOR
```
Checks:
  ✓ Shape (32000,) is 1D?           YES
  ✓ Sample rate 16000 Hz?           YES
  ✓ Duration 2 seconds?             YES
  ✓ No NaN/Inf?                     NO
  
Status: PASS ✓
```

### Step 4: NORMALIZER
```
Input:   [2048, 4096, 6144, ...] int16
÷32768:  [0.0625, 0.125, 0.1875, ...] float32
Clip:    (all within [-1.0, 1.0], no clipping needed)
Output:  [0.0625, 0.125, 0.1875, ...] float32
```

### FINAL OUTPUT
```
Shape:     (32000,)
Dtype:     float32
Range:     [-0.434296, 0.434296]  (actual data)
Canonical: YES ✓
Ready:     YES ✓ (can feed to ML model)
```

---

## Complete Example 2: rgb_image.png

### Step 1: INPUT
```
File:   rgb_image.png
Specs:  8-bit RGB PNG
Width:  256 pixels
Height: 256 pixels
Size:   0.5 KB
```

### Step 2: DECODER
```
Method:  PIL.Image.open()
Output:  array([[[255, 0, 0], ...]], dtype=uint8)
Shape:   (256, 256, 3)
Range:   [0, 255]
```

### Step 3: VALIDATOR
```
Checks:
  ✓ Shape (256, 256, 3) is 3D?              YES
  ✓ Width 256 in [16, 8192]?                YES
  ✓ Height 256 in [16, 8192]?               YES
  ✓ Channels 3?                             YES
  ✓ No NaN/Inf?                             NO
  
Status: PASS ✓
```

### Step 4: NORMALIZER
```
Input:   [[[255, 0, 0], [0, 255, 0], ...]] uint8
÷255:    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], ...]] float32
Clip:    (all within [0.0, 1.0], no clipping needed)
Output:  [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], ...]] float32
```

### FINAL OUTPUT
```
Shape:     (256, 256, 3)
Dtype:     float32
Range:     [0.0, 1.0]  (all pixels normalized)
Canonical: YES ✓
Ready:     YES ✓ (can feed to ML model)
```

---

## Batch Processing

### Input
```
["audio1.wav", "image1.png", "audio2.wav"]
```

### Processing (Lenient Mode)

```
File 1: audio1.wav
  Decode ✓  →  Validate ✓  →  Normalize ✓  →  SUCCESS

File 2: image1.png
  Decode ✓  →  Validate ✓  →  Normalize ✓  →  SUCCESS

File 3: audio2.wav
  Decode ✓  →  Validate ✗ (sample rate 4kHz < 8kHz)  →  FAIL (skip)
```

### Output
```python
{
  'successful': [
    {'filename': 'audio1.wav', 'shape': (32000,), 'data': array(...)},
    {'filename': 'image1.png', 'shape': (256, 256, 3), 'data': array(...)}
  ],
  'failed': [
    {'filename': 'audio2.wav', 'error': 'Invalid sample rate 4kHz'}
  ]
}
```

### Processing (Strict Mode)
```
File 1: audio1.wav
  Decode ✓  →  Validate ✓  →  Normalize ✓  →  SUCCESS

File 2: image1.png
  Decode ✓  →  Validate ✓  →  Normalize ✓  →  SUCCESS

File 3: audio2.wav
  Decode ✓  →  Validate ✗  →  STOP, raise ValidationError
  
(Remaining files not processed)
```

---

## Error Handling

### Possible Errors

**DecodingError** (Decoder step):
- File not found
- Unsupported format
- Corrupted file headers
- Missing required library

**ValidationError** (Validator step):
- Sample rate out of range
- Resolution too small/large
- Invalid number of channels
- Contains NaN/Inf values

**NormalizationError** (Normalizer step):
- Type conversion failed
- Scale overflow

### Client Code
```python
from bifrost.exceptions import (
    DecodingError,
    ValidationError
)

try:
    data, meta = pipeline.ingest_from_file(path, Modality.AUDIO)
except DecodingError as e:
    print(f"Could not decode: {e}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

---

## Summary Table

| Stage | Input Type | Input Range | Output Type | Output Range | Purpose |
|-------|-----------|-------------|-------------|--------------|---------|
| **Decoder** | bytes | file | int/uint | format-specific | Extract |
| **Validator** | int/uint | varies | int/uint | same | Verify |
| **Normalizer** | int/uint | format-specific | float32 | **canonical** | Standardize |

**Canonical ranges:**
- **Audio:** float32, [-1.0, 1.0]
- **Image:** float32, [0.0, 1.0]

---

## Key Insights

1. **Decoding** extracts format-specific data
   - Different libraries for different formats
   - Output always NumPy int/uint array

2. **Validation** enforces business constraints
   - Sample rate range (audio)
   - Resolution range (image)
   - Data quality (no NaN/Inf)

3. **Normalization** standardizes to ML-ready format
   - Type conversion: int/uint → float32
   - Scaling: format-specific range → canonical range
   - Audio: ÷32768 → [-1.0, 1.0]
   - Image: ÷255 → [0.0, 1.0]

4. **Batch processing** handles multiple files efficiently
   - Lenient mode: Skip failures, continue
   - Strict mode: Stop on first error

5. **Error handling** provides clear feedback
   - Custom exception types (Decoding, Validation, Normalization)
   - Detailed error messages
   - Success/failure reporting for batch operations

---

**This pipeline ensures all input data is canonicalized into a consistent float32 tensor format suitable for downstream ML models.**
