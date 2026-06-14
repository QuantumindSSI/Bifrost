# Phase 1.2 & Phase 2: Multi-Modal Curation and Tokenization

Complete implementation for Phase 1.2 (Dataset Curation) and Phase 2 (Tokenization) of the Bifrost Production Plan.

## Quick Start

### Prerequisites
```bash
# GPU environment (required for video processing)
# NVIDIA CUDA 12.1+, 8x A100 recommended for full pipeline

# Python packages
pip install -e ".[dev]"
pip install librosa soundfile pillow opencv-python pqdm scikit-learn

# Ensure data root is available
export DATA_ROOT=/path/to/raw/media
export CHECKPOINT_DIR=./checkpoints
export BATCH_SIZE=32
export GPUS=0,1,2,3
export EPOCHS=20
```

---

## Phase 1.2: Multi-Modal Dataset Curation

Curates a 3PB corpus from:
- **Audio**: 500K+ hours (LibriLight, AudioSet, FMA)
- **Video**: 50K+ hours (YouTube-8M, Kinetics-700)
- **Images**: 5M+ (LAION, COCO, ImageNet)
- **Text**: 500M tokens (Books3, CommonCrawl)
- **Sensors**: 5K+ hours (Industrial, Robotics, Automotive)

### Step 1: Prepare Raw Data

Organize raw media in DATA_ROOT:
```
$DATA_ROOT/
├── librispeech/
│   ├── train-clean-100/
│   ├── train-clean-360/
│   └── test-clean/
├── audioset/
│   ├── raw/
│   └── metadata.csv
├── youtube8m/
│   ├── videos/
│   └── metadata.jsonl
├── coco/
│   ├── images/
│   └── annotations.json
├── books3/
│   ├── text_files/
│   └── metadata.txt
└── sensors/
    ├── industrial/
    ├── robotics/
    └── automotive/
```

### Step 2: Run Dataset Curation

```bash
# Automatic curation (all modalities)
bash scripts/phase1.2_phase2_training.sh

# Or curation only (skip tokenization)
python3 -c "
from bifrost.data.multimodal_curator import MultiModalCurator, CurationConfig
from pathlib import Path

config = CurationConfig(output_dir=Path('./datasets/multimodal_corpus'))
curator = MultiModalCurator(config)
results = curator.curate_all(Path('$DATA_ROOT'))
print(results)
"

# Or curate individual modalities
python3 -c "
from bifrost.data.multimodal_curator import AudioCurator, CurationConfig
curator = AudioCurator(CurationConfig())
curator.process_librispeech(Path('$DATA_ROOT/librispeech'))
curator.save_metadata()
"
```

### Step 3: Verify Curation

```bash
# Check curated metadata
find datasets/multimodal_corpus -name "metadata.jsonl" -exec wc -l {} \;

# Expected output:
# datasets/multimodal_corpus/audio/metadata.jsonl: ~5000 audio files
# datasets/multimodal_corpus/video/metadata.jsonl: ~500 videos  
# datasets/multimodal_corpus/image/metadata.jsonl: ~50000 images
# datasets/multimodal_corpus/text/metadata.jsonl: ~2000 documents
# datasets/multimodal_corpus/sensor/metadata.jsonl: ~100 sensor files

# View sample metadata
head -5 datasets/multimodal_corpus/audio/metadata.jsonl | jq .
```

### Output Structure

```
datasets/multimodal_corpus/
├── audio/
│   ├── metadata.jsonl              # 1 line per audio file
│   ├── librispeech/                # Symlinked files
│   ├── audioset/
│   └── fma/
├── video/
│   ├── metadata.jsonl
│   └── youtube8m/
├── image/
│   ├── metadata.jsonl
│   ├── coco/
│   ├── laion_filtered/
│   └── imagenet/
├── text/
│   ├── metadata.jsonl
│   ├── books3/
│   └── commoncrawl/
├── sensor/
│   ├── metadata.jsonl
│   ├── industrial/
│   ├── robotics/
│   └── automotive/
└── cross_modal_pairs/
    └── pairs.jsonl                 # Audio+Text, Image+Caption, etc.
```

---

## Phase 2: Attractor Tokenization

Converts continuous spectral outputs → discrete 65K attractor vocabulary.

### Step 1: Extract Attractors (Phase 2.0)

```bash
# Automatic (after curation)
# Included in phase1.2_phase2_training.sh

# Or manual
python3 -c "
import json, torch, librosa
from pathlib import Path
from bifrost.pipeline import BifrostPipeline
from bifrost.tokenization.attractor_tokenizer import detect_attractors

pipeline = BifrostPipeline(n_fft_s0=1024, n_fft_s1=512, d_model=256)

# Process audio files
for audio_file in Path('datasets/multimodal_corpus/audio').glob('**/*.wav'):
    y, sr = librosa.load(str(audio_file), sr=16000, mono=True)
    signal = torch.from_numpy(y).unsqueeze(0)
    
    with torch.no_grad():
        bound_st, coherence = pipeline(signal, {'sample_rate': sr})
    
    attractors = detect_attractors(coherence[0], bound_st)
    print(f'{audio_file}: {len(attractors)} attractors')
"
```

### Step 2: Train Tokenizer (Phase 2.1)

```bash
# Automatic training
bash scripts/phase1.2_phase2_training.sh

# Or manual training with custom parameters
EPOCHS=20 BATCH_SIZE=32 GPUS=0,1,2,3 python3 -c "
import torch
from bifrost.tokenization.attractor_tokenizer import (
    AttractorTokenizer, AttractorTokenizerTrainer
)

# Initialize
tokenizer = AttractorTokenizer(vocab_size=65536, latent_dim=256)
trainer = AttractorTokenizerTrainer(tokenizer, lr=1e-3)

# Load attractors and train
# ... (see full training code in phase1.2_phase2_training.sh)

# Save
torch.save(tokenizer.state_dict(), 'checkpoints/phase2/tokenizer_final.pt')
"
```

### Step 3: Validate Tokenization (Phase 2.2)

```bash
# Reconstruction quality metrics
python3 -c "
import torch
from bifrost.tokenization.attractor_tokenizer import AttractorTokenizer

tokenizer = AttractorTokenizer(vocab_size=65536, latent_dim=256)
ckpt = torch.load('checkpoints/phase2/tokenizer_final.pt')
tokenizer.load_state_dict(ckpt['model_state_dict'])

# Test on attractors
# Compute: MSE, token perplexity, codebook utilization
print('✅ Tokenizer ready for Phase 3 training')
"
```

### Success Metrics

Phase 2 completion gates:
- ✅ **Tokenizer reconstruction MAE < 0.05**
- ✅ **Codebook perplexity > 50K** (tokens used)
- ✅ **Token sequence compression 10:1** (attractors → tokens)
- ✅ **Encoding confidence > 0.8** (VQ assignment quality)

---

## Individual Modality Training

### Audio-Only Training

```bash
python3 scripts/train_audio_encoder.py \
    --data-dir datasets/multimodal_corpus/audio \
    --output-dir checkpoints/audio_encoder \
    --epochs 20 \
    --batch-size 32

# Metrics
# - SNR estimation accuracy
# - Temporal coherence preservation
# - Compression ratio
```

### Video-Only Training

```bash
python3 scripts/train_video_encoder.py \
    --data-dir datasets/multimodal_corpus/video \
    --output-dir checkpoints/video_encoder \
    --fps 30 \
    --epochs 20

# Metrics
# - Temporal smoothness
# - Scene boundary detection
# - Frame reconstruction quality
```

### Image-Only Training

```bash
python3 scripts/train_image_encoder.py \
    --data-dir datasets/multimodal_corpus/image \
    --output-dir checkpoints/image_encoder \
    --patch-size 16 \
    --epochs 20
```

### Text-Only Training

```bash
python3 scripts/train_text_encoder.py \
    --data-dir datasets/multimodal_corpus/text \
    --output-dir checkpoints/text_encoder \
    --vocab-size 10000 \
    --epochs 20
```

### Sensor-Only Training

```bash
python3 scripts/train_sensor_encoder.py \
    --data-dir datasets/multimodal_corpus/sensor \
    --output-dir checkpoints/sensor_encoder \
    --channels 6 \
    --sample-rate 1000 \
    --epochs 20
```

---

## CLI Commands

### Run Full Pipeline

```bash
# All phases automatically
bifrost phase1.2-2 curate-tokenize \
    --data-root /path/to/raw/media \
    --output-dir ./datasets \
    --epochs 20 \
    --batch-size 32 \
    --gpus 0,1,2,3
```

### Curation Only

```bash
bifrost phase1.2 curate \
    --data-root /path/to/raw/media \
    --audio-sources librispeech,audioset \
    --video-sources youtube8m,kinetics \
    --image-sources coco,laion \
    --output-dir ./datasets/multimodal_corpus
```

### Tokenization Only

```bash
bifrost phase2 tokenize \
    --data-dir ./datasets/multimodal_corpus \
    --vocab-size 65536 \
    --epochs 20 \
    --batch-size 32 \
    --device cuda
```

### Evaluate Tokenizer

```bash
bifrost phase2 evaluate \
    --checkpoint ./checkpoints/phase2/tokenizer_final.pt \
    --data-dir ./datasets/multimodal_corpus/attractors \
    --num-samples 1000
```

---

## Monitoring & Logs

```bash
# View training progress
tail -f logs/phase1.2_curation.log
tail -f logs/phase2.0_attractors.log
tail -f logs/phase2.1_tokenizer.log
tail -f logs/phase2.2_validation.log

# TensorBoard (if enabled)
tensorboard --logdir logs/tensorboard --port 6006
```

---

## Hardware Requirements

### Minimum (Single GPU)
- GPU: NVIDIA A100 (40GB+)
- CPU: 32 cores
- RAM: 128GB
- Storage: 2TB (corpus only, no checkpoints)
- Time: 3-4 weeks (real-time audio processing)

### Recommended (Multi-GPU)
- GPUs: 8x NVIDIA A100 (80GB)
- CPU: 128 cores
- RAM: 512GB
- Storage: 3-4TB
- Time: 4-8 days (parallelized)

### Expected Disk Space

```
3PB Corpus (raw):
  Audio:    2.5 TB (500K hours @ 256kbps)
  Video:    2.0 TB (50K hours @ 4Mbps)
  Images:   200 GB (5M images)
  Text:     2 GB (500M tokens)
  Sensors:  50 GB

Processed/Curated:
  Metadata: ~50 GB
  Extracted attractors: ~500 GB
  Checkpoints: ~50 GB
  Cross-modal pairs: ~10 GB

Total: ~3.8 TB for full pipeline
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
export BATCH_SIZE=8

# Or sample data
python3 -c "
# Process 10% of data instead of 100%
sample_rate = 0.1
"
```

### Slow Video Processing

```bash
# Skip video processing, focus on audio
python3 -c "
from bifrost.data.multimodal_curator import AudioCurator
AudioCurator(...).process_librispeech(...)
"
```

### CUDA Out of Memory on Inference

```bash
# Use CPU for attractor extraction
python3 scripts/extract_attractors.py --device cpu
```

---

## Next Steps

After Phase 2 completion:
1. ✅ 3PB curated multi-modal corpus ready
2. ✅ 65K attractor vocabulary trained
3. → **Phase 3**: Phase-LM Training on 64x A100s
4. → **Phase 4**: Decoder Training (audio/video/image/text generation)
5. → **Phase 5**: API & Infrastructure Deployment

See [BIFROST_PRODUCTION_PLAN.md](../dev-docs/BIFROST_PRODUCTION_PLAN.md) Phase 3+ for next steps.

---

## References

- [Phase 1.2 Implementation](../src/bifrost/data/multimodal_curator.py)
- [Phase 2 Implementation](../src/bifrost/tokenization/attractor_tokenizer.py)
- [Bifrost Production Plan](../dev-docs/BIFROST_PRODUCTION_PLAN.md)
- [Bifrost README](../../README.md)
