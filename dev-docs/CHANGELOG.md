# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-17

### Added — FBC Core Pipeline (S0 → S1 → S2)

- **`SpectralTensor`** dataclass: canonical spectral container with
  `amplitude`, `phase`, `scale`, `uncertainty`, plus utility methods
  (`validate`, `to`, `complex_spectrum`, `energy`, `detach`).
- **`S0Canonicalizer`** (`fbc/s0_canonicalizer/`): converts raw signals
  to `SpectralTensor` via windowed FFT + z-score normalisation; supports
  1-D / 2-D / batched inputs.
- **`S1SpectralDecomposer`** (`fbc/s1_decomposer/`): multi-resolution
  decomposition via learnable wavelet bank + selective scan block
  (Mamba stand-in for Phase 1).
- **`ResonanceAttention`** (`fbc/resonance_attention/attention.py`):
  multi-head phase-coherence attention replacing dot-product attention,
  with learnable temperature `tau` and band weights.
- **`S2SpectralBinding`**: wraps `ResonanceAttention` for `SpectralTensor`
  input/output; coherence-weighted uncertainty refinement.
- **`FBCPipeline`** (`fbc/pipeline.py`): end-to-end S0 → S1 → S2
  orchestrator with cross-stage projection bridges.
- **Ingest → S0 Bridge** (`fbc/bridge.py`): adapter that normalises
  channel-axis layout (handles scipy `(samples, channels)` vs librosa
  `(channels, samples)`), flattens images to `(C, H*W)`, and rejects
  non-numeric inputs early.
- **`PhaseLockBridge`** (`fbc/phase_lock_bridge/`): Phase 1 initial
  implementation of cross-domain attractor matching with multi-band
  coherence gating (≥ 3 bands per Engineering Script §3).
- **`FrequencyAttractor`**: data structure for stable spectral patterns
  consumed by the Phase-Lock Bridge.

### Added — Benchmarks & Tests

- **`benchmarks/bench_attention.py`**: ResonanceAttention vs vanilla
  dot-product attention (latency, memory, coherence quality).
- **86 tests** total across all FBC modules (was 30+):
  - 10 SpectralTensor, 10 S0, 6 S1, 10 S2 / ResonanceAttention
  - 7 end-to-end pipeline, 19 ingest bridge, 16 phase-lock bridge
  - Real sample data integration (audio + images)
  - Edge cases: silent / single-sample / very-short signals,
    high-channel inputs.

### Fixed
- Coherence-score → uncertainty broadcasting in `S2SpectralBinding`
  for both 2-D `(channels, n_freq)` and 3-D `(batch, channels, n_freq)`
  inputs.

## [1.0.0] - 2026-05-15

### Added
- Audio decoder (WAV, MP3, FLAC, OGG formats)
  - Support for mono/stereo/surround (1-8 channels)
  - 8-48kHz sample rate handling
  - 8/16/24/32-bit depth support
  - Normalized output to float32 [-1.0, 1.0]

- Image decoder (PNG, JPG, JPEG, TIFF, BMP formats)
  - Support for grayscale/RGB/RGBA (1/3/4 channels)
  - 16-8192px resolution handling
  - 8/16-bit depth support
  - Normalized output to float32 [0.0, 1.0]

- Data validation system
  - Schema validation (format, shape, dtype)
  - Constraint checking (sample rate, resolution, dimensions)
  - NaN/Inf detection
  - Strict/lenient validation modes

- Data normalization
  - Type conversion (int → float32)
  - Range clipping and scaling
  - Canonical range enforcement

- Unified IngestPipeline
  - Single entry point for all modalities
  - Batch processing support
  - Selective error recovery
  - Metadata tracking

- Comprehensive test suite
  - 30+ unit and integration tests
  - 85%+ code coverage
  - Example scripts and fixtures

- Complete documentation
  - Architecture guide (7-layer ingestion pipeline)
  - API reference and quickstart
  - Language tradeoff analysis
  - Data ingestion plan (6-phase rollout)

- Sample data testing harness
  - Automatic test file generation
  - Single-file and batch testing
  - Command-line and Python API

### Fixed
- Audio dtype conversion (int16 → float32 using ÷32768)
- Image dtype conversion (uint8 → float32 using ÷255)
- Lazy-loading issue in PIL (Image.load() after Image.open())
- Quiet audio handling (scale up 10× for signals <1e-3 max)

### Known Limitations
- Phase 1: Audio and image only (text/tensor in Phase 2)
- No real-time streaming (streaming in Phase 3)
- No error recovery/dead-letter queue (Phase 4)
- Python-only (C++/Rust optimization in Phase 3+)

## Planned Releases

### [1.1.0] - Phase 2 (Text & Tensor Support)
- Text decoder (CSV, JSON, Parquet)
- Tensor loader (NPZ, HDF5, Zarr)
- Metadata schema standardization

### [1.2.0] - Phase 3 (Streaming & Optimization)
- RingBuffer for concurrent producer/consumer
- Source connectors (file watcher, HTTP, Kafka)
- C++ performance optimization (if profiling needed)

### [1.3.0] - Phase 4 (Production Hardening)
- Error recovery and dead-letter queue
- Monitoring and observability
- Distributed deployment support

---

**Version:** 1.0.0  
**Release Date:** 2026-05-15  
**Status:** Production-Ready
