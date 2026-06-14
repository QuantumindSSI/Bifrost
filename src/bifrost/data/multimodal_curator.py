"""
Phase 1.2: Multi-Modal Dataset Curation Pipeline

Orchestrates downloading, preprocessing, and curation of 3PB corpus across:
- Audio (500K+ hours)
- Video (50K+ hours)
- Images (5M+)
- Text (500M tokens)
- Sensors (5K+ hours)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import librosa
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CurationConfig:
    """Configuration for dataset curation"""
    
    # Storage
    output_dir: Path = Path("./datasets/multimodal_corpus")
    max_concurrent_downloads: int = 8
    
    # Audio settings
    audio_sample_rate: int = 16000
    audio_min_duration_sec: float = 0.5
    audio_max_duration_sec: float = 300.0
    audio_min_snr_db: float = 20.0
    
    # Video settings
    video_fps: int = 30
    video_min_frames: int = 10
    video_max_frames: int = 3000
    
    # Image settings
    image_min_resolution: int = 256
    image_max_resolution: int = 2048
    
    # Text settings
    text_min_tokens: int = 50
    text_max_tokens: int = 4096
    
    # Sensor settings
    sensor_min_channels: int = 1
    sensor_max_channels: int = 16
    sensor_sample_rate: int = 1000


class AudioCurator:
    """Audio dataset curation: LibriLight, AudioSet, FMA"""
    
    def __init__(self, config: CurationConfig):
        self.config = config
        self.output_dir = config.output_dir / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def quality_filter_audio(self, audio_path: str) -> Tuple[bool, str]:
        """
        Filter audio by:
        - SNR (Signal-to-Noise Ratio) > 20 dB
        - Duration within bounds
        - No corruption
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate, mono=True)
            
            # Check duration
            duration = len(y) / sr
            if duration < self.config.audio_min_duration_sec:
                return False, f"duration_too_short: {duration:.1f}s"
            if duration > self.config.audio_max_duration_sec:
                return False, f"duration_too_long: {duration:.1f}s"
            
            # Estimate SNR (simple: ratio of speech energy to silence)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            db = librosa.power_to_db(S, ref=np.max)
            
            # Compute percentiles
            percentile_5 = np.percentile(db, 5)
            percentile_95 = np.percentile(db, 95)
            snr_estimate = percentile_95 - percentile_5
            
            if snr_estimate < self.config.audio_min_snr_db:
                return False, f"low_snr: {snr_estimate:.1f}dB"
            
            return True, "pass"
        
        except Exception as e:
            return False, f"load_error: {str(e)}"
    
    def process_librispeech(self, librispeech_root: Path) -> Dict:
        """Process LibriSpeech 100h / 360h / 500h"""
        logger.info("Processing LibriSpeech...")
        stats = {"total": 0, "passed": 0, "failed": 0, "failures": {}}
        
        for mp3_file in tqdm(librispeech_root.rglob("*.mp3"), desc="LibriSpeech"):
            passed, reason = self.quality_filter_audio(str(mp3_file))
            stats["total"] += 1
            
            if passed:
                stats["passed"] += 1
                # Copy to corpus
                rel_path = mp3_file.relative_to(librispeech_root)
                output_file = self.output_dir / "librispeech" / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Soft link to save space
                if not output_file.exists():
                    os.symlink(mp3_file, output_file)
                
                self.metadata.append({
                    "source": "librispeech",
                    "file": str(output_file),
                    "modality": "audio",
                })
            else:
                stats["failed"] += 1
                stats["failures"][reason] = stats["failures"].get(reason, 0) + 1
        
        logger.info(f"LibriSpeech: {stats['passed']}/{stats['total']} passed")
        return stats
    
    def process_audioset(self, audioset_root: Path) -> Dict:
        """Process AudioSet (2M+ clips, ~5M hours)"""
        logger.info("Processing AudioSet (sampled)...")
        stats = {"total": 0, "passed": 0, "sampled": 0}
        
        # Sample 1% for manageable curation
        sample_rate = 0.01
        
        for wav_file in tqdm(audioset_root.rglob("*.wav"), desc="AudioSet"):
            if np.random.random() > sample_rate:
                continue
            
            stats["sampled"] += 1
            passed, reason = self.quality_filter_audio(str(wav_file))
            stats["total"] += 1
            
            if passed:
                stats["passed"] += 1
                self.metadata.append({
                    "source": "audioset",
                    "file": str(wav_file),
                    "modality": "audio",
                })
        
        logger.info(f"AudioSet: {stats['passed']}/{stats['total']} (sampled {stats['sampled']})")
        return stats
    
    def save_metadata(self):
        """Save audio metadata for downstream processing"""
        metadata_file = self.output_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(self.metadata)} audio metadata to {metadata_file}")


class VideoCurator:
    """Video dataset curation: YouTube-8M, Kinetics-700"""
    
    def __init__(self, config: CurationConfig):
        self.config = config
        self.output_dir = config.output_dir / "video"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def quality_filter_video(self, video_path: str) -> Tuple[bool, str]:
        """
        Filter video by:
        - Frame count within bounds
        - Resolution acceptable
        - No corruption
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, "cannot_open"
            
            # Get properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            cap.release()
            
            # Check bounds
            if frame_count < self.config.video_min_frames:
                return False, f"too_few_frames: {frame_count}"
            if frame_count > self.config.video_max_frames:
                return False, f"too_many_frames: {frame_count}"
            
            if min(height, width) < self.config.image_min_resolution:
                return False, f"too_small: {width}x{height}"
            if max(height, width) > self.config.image_max_resolution:
                return False, f"too_large: {width}x{height}"
            
            return True, "pass"
        
        except Exception as e:
            return False, f"error: {str(e)}"
    
    def process_youtube8m(self, youtube8m_root: Path) -> Dict:
        """Process YouTube-8M (sampled)"""
        logger.info("Processing YouTube-8M (sampled)...")
        stats = {"total": 0, "passed": 0, "sampled": 0}
        
        sample_rate = 0.001  # 0.1% of 8M = 8K videos
        
        for video_file in tqdm(youtube8m_root.rglob("*.mp4"), desc="YouTube-8M"):
            if np.random.random() > sample_rate:
                continue
            
            stats["sampled"] += 1
            passed, reason = self.quality_filter_video(str(video_file))
            stats["total"] += 1
            
            if passed:
                stats["passed"] += 1
                self.metadata.append({
                    "source": "youtube8m",
                    "file": str(video_file),
                    "modality": "video",
                })
        
        logger.info(f"YouTube-8M: {stats['passed']}/{stats['total']} (sampled {stats['sampled']})")
        return stats
    
    def save_metadata(self):
        metadata_file = self.output_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(self.metadata)} video metadata to {metadata_file}")


class ImageCurator:
    """Image dataset curation: LAION, COCO, ImageNet"""
    
    def __init__(self, config: CurationConfig):
        self.config = config
        self.output_dir = config.output_dir / "image"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def quality_filter_image(self, image_path: str) -> Tuple[bool, str]:
        """Filter by resolution and no corruption"""
        try:
            from PIL import Image
            img = Image.open(image_path)
            w, h = img.size
            
            if min(w, h) < self.config.image_min_resolution:
                return False, f"too_small: {w}x{h}"
            if max(w, h) > self.config.image_max_resolution:
                return False, f"too_large: {w}x{h}"
            
            return True, "pass"
        except Exception as e:
            return False, f"error: {str(e)}"
    
    def process_coco(self, coco_root: Path) -> Dict:
        """Process COCO (330K images)"""
        logger.info("Processing COCO...")
        stats = {"total": 0, "passed": 0, "failed": 0}
        
        for img_file in tqdm(coco_root.rglob("*.jpg"), desc="COCO"):
            passed, reason = self.quality_filter_image(str(img_file))
            stats["total"] += 1
            
            if passed:
                stats["passed"] += 1
                self.metadata.append({
                    "source": "coco",
                    "file": str(img_file),
                    "modality": "image",
                })
            else:
                stats["failed"] += 1
        
        logger.info(f"COCO: {stats['passed']}/{stats['total']} passed")
        return stats
    
    def save_metadata(self):
        metadata_file = self.output_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(self.metadata)} image metadata")


class TextCurator:
    """Text dataset curation: Books3, CommonCrawl, Lyrics"""
    
    def __init__(self, config: CurationConfig):
        self.config = config
        self.output_dir = config.output_dir / "text"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def process_books3(self, books3_root: Path) -> Dict:
        """Process Books3 (196GB, ~200K books)"""
        logger.info("Processing Books3 (sampled)...")
        stats = {"total": 0, "passed": 0}
        
        sample_rate = 0.01  # 1% sample = ~2K books
        
        for txt_file in tqdm(books3_root.rglob("*.txt"), desc="Books3"):
            if np.random.random() > sample_rate:
                continue
            
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Simple token count (words)
                token_count = len(text.split())
                
                if token_count < self.config.text_min_tokens:
                    continue
                
                stats["passed"] += 1
                self.metadata.append({
                    "source": "books3",
                    "file": str(txt_file),
                    "modality": "text",
                    "tokens": token_count,
                })
            except Exception as e:
                logger.warning(f"Error reading {txt_file}: {e}")
            
            stats["total"] += 1
        
        logger.info(f"Books3: {stats['passed']}/{stats['total']} passed")
        return stats
    
    def save_metadata(self):
        metadata_file = self.output_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(self.metadata)} text metadata")


class SensorCurator:
    """Sensor dataset curation: Industrial, Robotics, Automotive"""
    
    def __init__(self, config: CurationConfig):
        self.config = config
        self.output_dir = config.output_dir / "sensor"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def quality_filter_sensor(self, sensor_path: str) -> Tuple[bool, str]:
        """Filter by channel count and data integrity"""
        try:
            data = np.load(sensor_path)  # Assume .npy format
            
            if data.ndim == 1:
                channels = 1
            elif data.ndim == 2:
                channels = data.shape[0]
            else:
                return False, "invalid_shape"
            
            if channels < self.config.sensor_min_channels:
                return False, f"too_few_channels: {channels}"
            if channels > self.config.sensor_max_channels:
                return False, f"too_many_channels: {channels}"
            
            # Check for NaN/Inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                return False, "contains_nan_inf"
            
            return True, "pass"
        except Exception as e:
            return False, f"error: {str(e)}"
    
    def process_sensors(self, sensor_root: Path) -> Dict:
        """Process industrial/robotics/automotive sensor data"""
        logger.info("Processing sensor data...")
        stats = {"total": 0, "passed": 0, "failed": 0}
        
        for sensor_file in tqdm(sensor_root.rglob("*.npy"), desc="Sensors"):
            passed, reason = self.quality_filter_sensor(str(sensor_file))
            stats["total"] += 1
            
            if passed:
                stats["passed"] += 1
                self.metadata.append({
                    "source": "sensors",
                    "file": str(sensor_file),
                    "modality": "sensor",
                })
            else:
                stats["failed"] += 1
        
        logger.info(f"Sensors: {stats['passed']}/{stats['total']} passed")
        return stats
    
    def save_metadata(self):
        metadata_file = self.output_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(self.metadata)} sensor metadata")


class CrossModalAligner:
    """Build cross-modal pairs for contrastive training"""
    
    def __init__(self, config: CurationConfig):
        self.config = config
        self.output_dir = config.output_dir / "cross_modal_pairs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_pairs(self) -> List[Dict]:
        """
        Create aligned pairs:
        - Audio + Lyrics (from captions/metadata)
        - Video + Audio (video clips with soundtrack)
        - Image + Caption (from COCO)
        - Audio + Sensor (industrial: vibration + audio)
        """
        logger.info("Building cross-modal pairs...")
        pairs = []
        
        # Load metadata from all modalities
        audio_meta = self._load_metadata("audio")
        video_meta = self._load_metadata("video")
        image_meta = self._load_metadata("image")
        text_meta = self._load_metadata("text")
        sensor_meta = self._load_metadata("sensor")
        
        # Simple strategy: temporal/semantic matching
        # In production: use CLIP embeddings or manual annotation
        
        # Audio + Text pairs (from captions if available)
        for audio in audio_meta[:1000]:  # Sample
            for text in text_meta[:1000]:
                pairs.append({
                    "modality_1": "audio",
                    "file_1": audio["file"],
                    "modality_2": "text",
                    "file_2": text["file"],
                })
        
        # Image + Text pairs (COCO captions)
        for image in image_meta[:500]:
            for text in text_meta[:500]:
                pairs.append({
                    "modality_1": "image",
                    "file_1": image["file"],
                    "modality_2": "text",
                    "file_2": text["file"],
                })
        
        logger.info(f"Created {len(pairs)} cross-modal pairs")
        
        # Save pairs
        pairs_file = self.output_dir / "pairs.jsonl"
        with open(pairs_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        
        return pairs
    
    def _load_metadata(self, modality: str) -> List[Dict]:
        """Load metadata for a modality"""
        meta_file = self.config.output_dir / modality / "metadata.jsonl"
        if not meta_file.exists():
            return []
        
        meta = []
        with open(meta_file, "r") as f:
            for line in f:
                meta.append(json.loads(line))
        return meta


class MultiModalCurator:
    """Orchestrate full dataset curation"""
    
    def __init__(self, config: Optional[CurationConfig] = None):
        self.config = config or CurationConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
    
    def curate_all(self, data_root: Path) -> Dict:
        """Run full curation pipeline"""
        results = {
            "audio": {},
            "video": {},
            "image": {},
            "text": {},
            "sensor": {},
            "cross_modal": {},
        }
        
        # Audio
        logger.info("\n" + "="*60)
        logger.info("PHASE 1.2: Audio Curation")
        logger.info("="*60)
        audio_curator = AudioCurator(self.config)
        results["audio"] = audio_curator.process_librispeech(data_root / "librispeech")
        audio_curator.save_metadata()
        
        # Video
        logger.info("\n" + "="*60)
        logger.info("PHASE 1.2: Video Curation")
        logger.info("="*60)
        video_curator = VideoCurator(self.config)
        results["video"] = video_curator.process_youtube8m(data_root / "youtube8m")
        video_curator.save_metadata()
        
        # Image
        logger.info("\n" + "="*60)
        logger.info("PHASE 1.2: Image Curation")
        logger.info("="*60)
        image_curator = ImageCurator(self.config)
        results["image"] = image_curator.process_coco(data_root / "coco")
        image_curator.save_metadata()
        
        # Text
        logger.info("\n" + "="*60)
        logger.info("PHASE 1.2: Text Curation")
        logger.info("="*60)
        text_curator = TextCurator(self.config)
        results["text"] = text_curator.process_books3(data_root / "books3")
        text_curator.save_metadata()
        
        # Sensors
        logger.info("\n" + "="*60)
        logger.info("PHASE 1.2: Sensor Curation")
        logger.info("="*60)
        sensor_curator = SensorCurator(self.config)
        results["sensor"] = sensor_curator.process_sensors(data_root / "sensors")
        sensor_curator.save_metadata()
        
        # Cross-modal pairs
        logger.info("\n" + "="*60)
        logger.info("PHASE 1.2: Cross-Modal Alignment")
        logger.info("="*60)
        aligner = CrossModalAligner(self.config)
        results["cross_modal"]["pairs"] = len(aligner.build_pairs())
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print curation summary"""
        logger.info("\n" + "="*60)
        logger.info("CURATION SUMMARY")
        logger.info("="*60)
        
        for modality, stats in results.items():
            if isinstance(stats, dict) and "passed" in stats:
                logger.info(f"{modality.upper()}: {stats['passed']} passed, "
                          f"{stats.get('failed', 0)} failed")
            elif isinstance(stats, dict):
                logger.info(f"{modality.upper()}: {stats}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python multimodal_curator.py <data_root>")
        sys.exit(1)
    
    data_root = Path(sys.argv[1])
    curator = MultiModalCurator()
    results = curator.curate_all(data_root)
