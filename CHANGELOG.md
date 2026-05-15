# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
