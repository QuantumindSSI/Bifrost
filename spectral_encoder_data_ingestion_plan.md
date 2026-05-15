# Spectral Encoder: Universal Data Ingestion Plan

## 1. Overview & Objectives

**Goal:** Build a robust, scalable data ingestion system for a universal spectral encoder that accepts:
- **Audio** (WAV, MP3, FLAC, OGG)
- **Images** (PNG, JPEG, TIFF, BMP)
- **Text** (TXT, CSV, JSON, Parquet)
- **Tensors** (NPZ, HDF5, Zarr, binary)

**From sources:**
- Local file systems
- Streaming APIs (HTTP, WebSocket, Kafka)
- Databases (SQL, MongoDB, S3, cloud storage)

**Key constraints:**
- Low latency (streaming)
- High throughput (batch processing)
- Memory efficiency (large datasets)
- Type safety & validation
- Graceful error handling & recovery

---

## 2. Architecture Overview

```
Data Sources
    │
    ├─ File System (Watcher)
    ├─ Streaming Endpoints (HTTP, WebSocket, Kafka)
    ├─ Databases (SQL, MongoDB, S3)
    │
    ▼
┌─────────────────────────┐
│  Source Connector Layer │  (Pluggable drivers)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Data Deserialization       │  (Format parsers)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Validation & Error Handler │  (Schema, integrity)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Type Conversion & Normalize│  (float32, range [0,1])
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Buffering & Rate Control   │  (Ring buffer, backpressure)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Metadata Extraction        │  (Shape, SR, channels, etc)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Queue & Load Balancing     │  (FIFO, priority, sharding)
└────────────┬────────────────┘
             │
             ▼
    Spectral Transform Stage
```

---

## 3. Granular Component Breakdown

### 3.1 Source Connector Layer

**Purpose:** Abstract different data sources under a unified interface.

#### Audio Sources
- **Local:** librosa, scipy.io.wavfile, pydub
- **Streaming:** sounddevice (live microphone), icecast (streaming radio)
- **APIs:** YouTube, Spotify (with OAuth), cloud storage (S3, GCS)

#### Image Sources
- **Local:** PIL/Pillow, OpenCV, scikit-image
- **Streaming:** HTTP range requests, MJPEG streams
- **APIs:** Azure Blob Storage, Google Cloud Storage, AWS S3

#### Text Sources
- **Local:** plain text, CSV, JSON, Parquet, Arrow
- **Streaming:** WebSocket text frames, Kafka topics
- **APIs:** REST endpoints returning JSON/CSV, database queries

#### Tensor Sources
- **Local:** NumPy (.npy, .npz), HDF5 (.h5), Zarr
- **Streaming:** gRPC, TensorFlow Serving, ONNX Runtime
- **APIs:** Cloud ML APIs (Vertex AI, SageMaker)

**Interface contract:**
```python
class DataSource(ABC):
    @abstractmethod
    async def read(self) -> bytes: ...
    
    @abstractmethod
    async def metadata(self) -> Dict: ...
    
    @abstractmethod
    async def close(self) -> None: ...
```

---

### 3.2 Format Deserialization

**Purpose:** Convert raw bytes → canonical in-memory representation.

#### Audio Decoding
- **WAV/PCM:** struct unpacking (little-endian, sample width 1-4 bytes)
- **MP3/AAC:** FFmpeg (ffmpeg-python), librosa backend
- **FLAC/OGG:** soundfile, librosa
- **Output:** float32, mono or multi-channel array, shape (samples,) or (channels, samples)

#### Image Decoding
- **PNG/JPEG/TIFF:** Pillow.Image.open → NumPy array
- **BMP:** struct parsing or PIL
- **MJPEG:** frame-by-frame extraction (OpenCV cv2.VideoCapture)
- **Output:** uint8 or float32, shape (H, W) or (H, W, C), C ∈ {1, 3, 4}

#### Text Parsing
- **Plain text:** str.split('\n') or iterator.readlines()
- **CSV:** pandas.read_csv, csv module
- **JSON:** json.load, streaming parser (ijson)
- **Parquet:** pyarrow.parquet.read_table
- **Output:** list of strings or pd.DataFrame, with schema tracking

#### Tensor Loading
- **NPY/NPZ:** numpy.load (memory-mapped for large files)
- **HDF5:** h5py.File (lazy loading, compression-aware)
- **Zarr:** zarr.open (supports chunking, parallel reads)
- **Output:** NumPy array or sparse matrix, with dtype & shape preserved

---

### 3.3 Validation & Error Handling

**Purpose:** Detect malformed data early, apply recovery strategies.

#### Audio Validation
- ✓ Sample rate in [8000, 48000] Hz (warn if outside)
- ✓ Bit depth in {8, 16, 24, 32} bits
- ✓ Channels in [1, 8]
- ✓ Duration > 10ms (reject if < 10ms)
- ✓ No NaN/Inf values (clamp or interpolate)
- Recovery: resample if SR mismatch, mix-down if >2 channels, zero-fill shorts

#### Image Validation
- ✓ Resolution > 16×16 pixels (reject if smaller)
- ✓ Channels in {1, 3, 4}
- ✓ No partial/truncated data (verify file size matches)
- ✓ Color space: assume sRGB for RGB, grayscale for single-channel
- Recovery: resize if too large (max 4K), pad if too small, fill missing channels

#### Text Validation
- ✓ Encoding: detect & convert to UTF-8 (chardet lib)
- ✓ Non-empty after whitespace strip
- ✓ Max length: 1M characters per document
- ✓ No null bytes (strip if present)
- Recovery: skip invalid records, log to error queue

#### Tensor Validation
- ✓ Shape not empty (ndim ≥ 1)
- ✓ dtype in {float32, float64, int32, int64, uint8, uint16}
- ✓ No NaN/Inf (clamp or mask)
- ✓ Memory footprint < available RAM
- Recovery: convert dtype if safe, sparse-ify if > 90% zeros

**Error Handling Strategies:**
1. **Strict:** Reject and log (default for production)
2. **Lenient:** Warn and continue with defaults (dev/testing)
3. **Repair:** Apply heuristic recovery (e.g., resample, impute)
4. **Dead-letter:** Queue for manual review (async reprocessing)

---

### 3.4 Type Conversion & Normalization

**Purpose:** Map all data to canonical float32 ranges for downstream processing.

#### Audio Normalization
```
Raw: int16 [-32768, 32767]  →  float32 [-1.0, 1.0]
Raw: uint8 [0, 255]          →  float32 [0.0, 1.0]
Raw: float32 (assumed [-1,1]) →  passthrough or clip
Recovery: if max(abs(audio)) < 1e-3, scale by 10x (likely silenced)
```

#### Image Normalization
```
Raw: uint8 [0, 255]   →  float32 [0.0, 1.0]  (divide by 255)
Raw: uint16 [0, 65535] →  float32 [0.0, 1.0] (divide by 65535)
Raw: float32           →  clip to [0.0, 1.0]
```

#### Text Normalization
```
Encoding: detect with chardet, decode to str (UTF-8)
Case: preserve (no forced lower)
Whitespace: strip leading/trailing, normalize internal (single space)
Special chars: keep, don't sanitize (preserve semantics)
Length: truncate to max_length if set, else warn if >1M chars
```

#### Tensor Normalization
```
Float types (float32, float64): detect scale
  - If max < 1.0: likely already normalized, passthrough
  - If max >> 1000: likely unnormalized, divide by max or log-scale
  
Integer types (int32, int64, uint8, uint16): convert to float32
  - uint8: divide by 255
  - uint16: divide by 65535
  - int32/int64: divide by max observed value or 2^31-1
  
Sparse: keep sparse format (COO, CSR) or convert to dense if small
```

---

### 3.5 Buffering & Rate Control

**Purpose:** Decouple source speed from downstream consumption, prevent memory overflow.

#### Ring Buffer (Streaming)
- **Size:** configurable (default 100 MB)
- **Overflow policy:** block producer (backpressure) or drop oldest (circular)
- **Threads:** dedicated producer & consumer threads
- **Use case:** real-time audio/video ingestion

```python
class RingBuffer:
    def __init__(self, capacity_mb=100):
        self.buffer = deque(maxlen=int(capacity_mb * 1e6 / 8))
    
    def put(self, data: np.ndarray, timeout=5.0):
        # blocks if full, respects timeout
        pass
    
    def get(self, timeout=1.0) -> np.ndarray:
        # returns batch or times out
        pass
```

#### Batch Accumulation
- **Batch size:** configurable (default 32 for images, 512 samples for audio)
- **Timeout:** max accumulation time (default 5s, prevents stale data)
- **Padding:** zero-pad if batch incomplete at timeout

#### Backpressure Strategy
```
If buffer fill > 80%:
  → Signal producer to slow down (sleep 100ms)
  → Or drop lowest-priority items from queue
If buffer fill > 95%:
  → Pause all producers
  → Log warning
```

---

### 3.6 Metadata Extraction

**Purpose:** Capture source characteristics for downstream use (normalization, model input sizing, logging).

#### Audio Metadata
```
{
  "format": "wav" | "mp3" | "flac",
  "sample_rate": int,  # Hz
  "channels": int,
  "bit_depth": int,  # 8, 16, 24, 32
  "duration_sec": float,
  "num_samples": int,
  "codec": str,
  "bitrate_kbps": int (optional),
  "source": str  # "file://" | "http://" | "kafka://"
}
```

#### Image Metadata
```
{
  "format": "png" | "jpeg" | "tiff",
  "width": int,
  "height": int,
  "channels": int,  # 1, 3, 4
  "bit_depth": int,  # 8, 16
  "color_space": "sRGB" | "grayscale" | "lab",
  "dpi": int (optional),
  "size_bytes": int,
  "source": str
}
```

#### Text Metadata
```
{
  "format": "plain" | "csv" | "json" | "parquet",
  "encoding": "utf-8",
  "num_lines": int,
  "num_chars": int,
  "num_tokens": int (optional),
  "language": str (optional, via langdetect),
  "schema": Dict (for structured data),
  "source": str
}
```

#### Tensor Metadata
```
{
  "format": "npy" | "npz" | "h5" | "zarr",
  "dtype": str,  # "float32", "int64", etc
  "shape": Tuple[int],
  "size_bytes": int,
  "is_sparse": bool,
  "sparsity": float (% zeros),
  "min_val": float,
  "max_val": float,
  "source": str
}
```

---

### 3.7 Queuing & Load Balancing

**Purpose:** Route processed batches to spectral transform stage, handle multiple concurrent sources.

#### Queue Architecture
```python
class DataQueue:
    def __init__(self, max_size=1000, priority=False):
        if priority:
            self.queue = PriorityQueue(maxsize=max_size)
        else:
            self.queue = Queue(maxsize=max_size)
    
    def put(self, item: Tuple[ndarray, Dict], priority=0):
        # item = (data, metadata)
        pass
    
    def get(self, timeout=1.0) -> Tuple[ndarray, Dict]:
        pass
```

#### Load Balancing (Multi-Source)
- **FIFO (default):** First-in, first-out by source order
- **Priority:** weight by source importance, data freshness
- **Round-robin:** alternate between sources (fairness)
- **Least-loaded:** pull from source with smallest buffer

#### Sharding Strategy (Large Files)
- Split large tensors/datasets by partition:
  - Audio: split by time (e.g., 30-second chunks)
  - Images: split by spatial tiles (e.g., 224×224 patches)
  - Text: split by line count or token count
  - Tensors: split by first dimension (batch or leading mode)
- Each shard queued independently, reassemble later if needed

---

## 4. Implementation Phases

### Phase 1: Core Serialization & Validation (Week 1-2)
**Deliverable:** Audio + image deserialization, validation, normalization

**Tasks:**
- [ ] Audio decoder (WAV via scipy, MP3 via librosa)
- [ ] Image decoder (PIL/Pillow)
- [ ] Validation pipeline (schema checks, error recovery)
- [ ] Type conversion (int16→float32, uint8→float32)
- [ ] Unit tests (malformed data, edge cases)
- [ ] Integration test (10 audio clips + 10 images)

### Phase 2: Text & Tensor Support (Week 2-3)
**Deliverable:** CSV, JSON, Parquet, NPZ/HDF5 loading

**Tasks:**
- [ ] CSV/JSON parser (pandas, ijson for streaming)
- [ ] Parquet reader (pyarrow)
- [ ] NPZ/HDF5 loaders (numpy.load, h5py)
- [ ] Zarr support (lazy chunked loading)
- [ ] Metadata extraction (all modalities)
- [ ] Tests: 5 text files + 5 tensor files

### Phase 3: Buffering & Rate Control (Week 3-4)
**Deliverable:** RingBuffer, batch accumulation, backpressure

**Tasks:**
- [ ] RingBuffer class (deque-based, thread-safe)
- [ ] Batch accumulator (configurable size, timeout)
- [ ] Backpressure logic (signal producer to slow down)
- [ ] Thread pool for concurrent producers
- [ ] Load test: 4 producers, 2 consumers, 100MB buffer

### Phase 4: Source Connectors (Week 4-5)
**Deliverable:** File system watcher, HTTP, Kafka, S3/GCS connectors

**Tasks:**
- [ ] File watcher (watchdog library, platform-agnostic)
- [ ] HTTP connector (requests + range requests for large files)
- [ ] Kafka consumer (confluent-kafka or aiokafka)
- [ ] S3 connector (boto3, async support)
- [ ] GCS connector (google-cloud-storage)
- [ ] Integration test: 10 files from 3 different sources

### Phase 5: Error Recovery & Monitoring (Week 5-6)
**Deliverable:** Dead-letter queue, retry logic, telemetry

**Tasks:**
- [ ] Dead-letter queue (async reprocessing)
- [ ] Retry policy (exponential backoff, max retries)
- [ ] Logging & metrics (total bytes, errors, latency)
- [ ] Dashboard (Prometheus + Grafana or similar)
- [ ] Stress test: 1000 concurrent files, >80% error injection

### Phase 6: Optimization & Documentation (Week 6+)
**Deliverable:** Production-ready system, API docs, examples

**Tasks:**
- [ ] Memory profiling (reduce peak usage)
- [ ] Parallelization tuning (number of workers, buffer size)
- [ ] API documentation (source connectors, queue interface)
- [ ] User guide + examples (audio, image, text, tensor ingestion)
- [ ] Performance benchmarks (throughput, latency, memory)

---

## 5. Code Structure (Proposed)

```
spectral_encoder/
├── ingest/
│   ├── __init__.py
│   ├── sources/
│   │   ├── file.py          # FileSource (local FS)
│   │   ├── http.py          # HTTPSource
│   │   ├── kafka.py         # KafkaSource
│   │   ├── s3.py            # S3Source
│   │   ├── gcs.py           # GCSSource
│   │   └── base.py          # DataSource (ABC)
│   ├── decoders/
│   │   ├── audio.py         # Audio deserialization
│   │   ├── image.py         # Image deserialization
│   │   ├── text.py          # Text deserialization
│   │   ├── tensor.py        # Tensor deserialization
│   │   └── base.py          # Decoder (ABC)
│   ├── validation/
│   │   ├── audio.py
│   │   ├── image.py
│   │   ├── text.py
│   │   ├── tensor.py
│   │   ├── schema.py        # Schema validator
│   │   └── exceptions.py    # ValidationError, DecodingError
│   ├── normalize.py         # Type conversion & normalization
│   ├── metadata.py          # Metadata extraction
│   ├── buffer.py            # RingBuffer, batch accumulation
│   ├── queue.py             # DataQueue, load balancing
│   ├── retry.py             # Retry logic, dead-letter queue
│   └── pipeline.py          # Orchestrator (ties all together)
├── tests/
│   ├── conftest.py          # Fixtures
│   ├── unit/
│   │   ├── test_decoders.py
│   │   ├── test_validation.py
│   │   ├── test_normalize.py
│   │   └── test_buffer.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_sources.py
│   │   └── test_stress.py
│   └── fixtures/
│       ├── sample.wav
│       ├── sample.png
│       ├── sample.csv
│       └── sample.npz
├── examples/
│   ├── audio_ingest.py
│   ├── image_ingest.py
│   ├── text_ingest.py
│   └── tensor_ingest.py
└── docs/
    ├── architecture.md
    ├── api.md
    └── benchmarks.md
```

---

## 6. Dependencies

```
Core:
- numpy>=1.20.0
- librosa>=0.9.0          (audio)
- Pillow>=9.0.0           (image)
- soundfile>=0.11.0       (FLAC/OGG)
- scipy>=1.7.0            (WAV, signal processing)

Structured data:
- pandas>=1.3.0
- pyarrow>=7.0.0          (Parquet, Arrow)
- ijson>=3.1.0            (JSON streaming)

ML frameworks (optional):
- tensorflow>=2.8.0       (tf.data.Dataset)
- torch>=1.10.0           (DataLoader)

Cloud:
- boto3>=1.20.0           (S3)
- google-cloud-storage>=2.0.0
- confluent-kafka>=1.7.0  (Kafka)

Monitoring:
- prometheus-client>=0.12.0
- structlog>=21.0.0       (structured logging)

Testing:
- pytest>=7.0.0
- pytest-asyncio>=0.18.0
- pytest-xdist>=2.5.0     (parallel tests)
```

---

## 7. Configuration Example

```yaml
# ingest_config.yaml
ingest:
  sources:
    - name: local_audio
      type: file
      path: /data/audio/*.wav
      buffer_size_mb: 100
      batch_size: 32
      
    - name: local_images
      type: file
      path: /data/images/*.png
      buffer_size_mb: 50
      batch_size: 32
      
    - name: s3_tensors
      type: s3
      bucket: my-bucket
      prefix: tensors/
      region: us-west-2
      buffer_size_mb: 200
      
    - name: kafka_events
      type: kafka
      bootstrap_servers: ["localhost:9092"]
      topic: sensor_data
      buffer_size_mb: 150

  validation:
    strategy: lenient  # strict | lenient | repair
    max_audio_duration_sec: 300
    max_image_resolution: [4096, 4096]
    max_text_length: 1000000
    max_tensor_size_mb: 1000

  queue:
    type: fifo          # fifo | priority
    max_items: 1000
    load_balance: round_robin  # round_robin | least_loaded

  error_handling:
    max_retries: 3
    retry_backoff: exponential
    dead_letter_queue: true
    dead_letter_path: /tmp/dead_letter/

  monitoring:
    enable_metrics: true
    log_level: INFO
    sample_rate: 0.1    # log 10% of items
```

---

## 8. API Preview

```python
from spectral_encoder.ingest import IngestPipeline, FileSource, HTTPSource

# Create pipeline
pipeline = IngestPipeline.from_config("ingest_config.yaml")

# Or manual setup
pipeline = IngestPipeline(
    buffer_size_mb=100,
    batch_size=32,
    validation_strategy="lenient"
)

# Register sources
pipeline.add_source(FileSource("/data/audio/*.wav", modality="audio"))
pipeline.add_source(HTTPSource("https://api.example.com/data", modality="image"))

# Consume data
for batch, metadata in pipeline.consume():
    # batch: np.ndarray, shape (batch_size, ...)
    # metadata: list of dicts with source, modality, etc.
    
    # Pass to spectral transform
    spectra = spectral_transform(batch, modality=metadata[0]['modality'])
```

---

## 9. Success Criteria

- ✓ Load 1000 audio files (1-300sec each) without memory overflow
- ✓ Ingest 10K images (640×480 or larger) with <1% error rate
- ✓ Parse 1M-line CSV files with <100MB peak memory
- ✓ Stream Kafka data at 1000 msgs/sec without dropping
- ✓ Detect & recover from 95% of malformed data
- ✓ Latency: <100ms from source read to queue output (p95)
- ✓ Throughput: ≥100 MB/sec aggregate across all sources
- ✓ Zero data loss under normal operation

---

## 10. Timeline & Staffing

| Phase | Duration | Tasks | Owner |
|-------|----------|-------|-------|
| 1: Core | 2 weeks | Audio, image, validation | You |
| 2: Extended | 1 week | Text, tensor support | You |
| 3: Buffering | 1 week | RingBuffer, backpressure | You |
| 4: Sources | 1 week | File, HTTP, Kafka, cloud | You |
| 5: Resilience | 1 week | Errors, retry, monitoring | You |
| 6: Polish | 1+ weeks | Docs, benchmarks, examples | You |

**Total:** ~7-8 weeks for production-ready system.

---

## 11. Next Steps

1. **Clarify priorities:** Which data modality/source is highest priority?
2. **Finalize config:** Review ingest_config.yaml structure; add custom validators?
3. **Setup repo:** Initialize Python project, install deps, create test fixtures
4. **Phase 1 kickoff:** Start with audio deserialization (shortest path to value)
