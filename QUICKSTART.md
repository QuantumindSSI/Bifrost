# Quickstart

## Install

```bash
git clone https://github.com/quantumind/bifrost.git
cd bifrost
pip install -e .
```

---

## Process a signal

```python
import torch
from bifrost.pipeline import BifrostPipeline

pipe = BifrostPipeline(d_model=256, use_complex_ssm=True, use_s3_attractor=True)

# Any 1D signal — audio, sensor, time series
signal = torch.randn(1, 16000)       # batch=1, 16000 samples
output, coherence = pipe(signal)

print(output.amplitude.shape)        # (1, T, F) spectral embedding
print(coherence.mean().item())       # phase coherence score ∈ [0, 1]
```

---

## Load audio or image from file

```python
from bifrost.ingest import IngestPipeline, Modality

pipe = IngestPipeline()

# Audio
audio, meta = pipe.ingest_from_file("recording.wav", Modality.AUDIO)
# audio: float32 array, shape (channels, samples), range [-1, 1]
# meta: {"sample_rate": 16000, "duration_sec": 3.2, ...}

# Image
image, meta = pipe.ingest_from_file("photo.png", Modality.IMAGE)
# image: float32 array, shape (H, W, C), range [0, 1]
```

---

## Multi-modal pipeline

```python
from bifrost.multimodal_pipeline import create_multimodal_pipeline, Modality
import torch

pipe = create_multimodal_pipeline(d_model=256)

audio_signal = torch.randn(1, 16000)
image_signal = torch.randn(1, 3, 224, 224)

audio_out = pipe(audio_signal, modality=Modality.AUDIO)
image_out = pipe(image_signal, modality=Modality.IMAGE)
```

---

## Streaming inference

The SSM carries state across chunk boundaries — no context is lost at chunk edges.

```python
h = None
for chunk in stream:                             # chunk: (1, chunk_size)
    output, coherence, h = pipe.forward_stateful(chunk, h_0=h)
```

---

## Run the CLI

```bash
bifrost process audio.wav -o output.pt           # full pipeline
bifrost attractors audio.wav -o attractors.pt    # attractor extraction
bifrost bridge a.pt b.pt --min-locked 3          # cross-modal bridge
bifrost demo 1                                   # anti-phase demo
bifrost bench attention                          # attention benchmark
```

---

## Run tests

```bash
pytest tests/ -v
pytest tests/ --cov=bifrost --cov-report=html
```

---

## Supported input formats

| Modality | Formats |
|---|---|
| Audio | WAV, MP3, FLAC, OGG |
| Image | PNG, JPG, TIFF, BMP |
| Text | raw string → character-level STFT |
| Sensor | any float32 tensor |

**Audio constraints:** sample rate 8–48 kHz, duration 10 ms–3600 s  
**Image constraints:** 16–8192 px per side, 1/3/4 channels
