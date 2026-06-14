"""
Phase 1: Automated Checkpointing & Version Control

Intelligent checkpoint management with:
- Automatic versioning (e.g., checkpoint_v0001.pt, checkpoint_v0002.pt)
- Metrics tracking (loss, validation accuracy, coherence)
- Best checkpoint selection (based on validation metrics)
- Efficient rollback to previous versions
- Git integration for reproducibility
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
import subprocess

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata associated with a checkpoint"""
    
    version: int
    epoch: int
    global_step: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    model_size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary"""
        return cls(**data)


class CheckpointManager:
    """
    Manages checkpoint versioning and selection.
    
    Maintains:
    - Sequential version numbers
    - Metadata for each checkpoint
    - Best checkpoint tracking
    - Efficient cleanup of old checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        best_metric: str = "val_loss",
        best_metric_mode: str = "min",  # "min" or "max"
    ):
        """
        Initialize checkpoint manager.
        
        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory to store checkpoints
        max_checkpoints : int
            Maximum number of checkpoints to keep
        best_metric : str
            Metric name to use for best checkpoint selection
        best_metric_mode : str
            "min" if lower is better, "max" if higher is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode
        
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self.best_checkpoint_file = self.checkpoint_dir / "best_checkpoint.json"
        
        # Load existing metadata
        self.checkpoints: Dict[int, CheckpointMetadata] = self._load_metadata()
        self.best_checkpoint: Optional[CheckpointMetadata] = self._load_best_checkpoint()
        
        logger.info(
            f"CheckpointManager initialized: dir={self.checkpoint_dir}, "
            f"max_ckpts={max_checkpoints}, best_metric={best_metric} ({best_metric_mode})"
        )
    
    def _load_metadata(self) -> Dict[int, CheckpointMetadata]:
        """Load checkpoint metadata from disk"""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file) as f:
                data = json.load(f)
            
            return {
                int(v_str): CheckpointMetadata.from_dict(meta)
                for v_str, meta in data.items()
            }
        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")
            return {}
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk"""
        data = {
            str(version): meta.to_dict()
            for version, meta in self.checkpoints.items()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_best_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Load best checkpoint metadata"""
        if not self.best_checkpoint_file.exists():
            return None
        
        try:
            with open(self.best_checkpoint_file) as f:
                data = json.load(f)
            return CheckpointMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load best checkpoint metadata: {e}")
            return None
    
    def _save_best_checkpoint(self) -> None:
        """Save best checkpoint metadata"""
        if self.best_checkpoint is None:
            return
        
        with open(self.best_checkpoint_file, 'w') as f:
            json.dump(self.best_checkpoint.to_dict(), f, indent=2)
    
    def _get_next_version(self) -> int:
        """Get next checkpoint version number"""
        if not self.checkpoints:
            return 1
        return max(self.checkpoints.keys()) + 1
    
    def _is_better_checkpoint(self, new_metric: float, best_metric: float) -> bool:
        """Check if new metric is better than best"""
        if self.best_metric_mode == "min":
            return new_metric < best_metric
        else:  # "max"
            return new_metric > best_metric
    
    def save_checkpoint(
        self,
        model_state: Dict[str, torch.Tensor],
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        config_hash: Optional[str] = None,
    ) -> Path:
        """
        Save a new checkpoint with automatic versioning.
        
        Parameters
        ----------
        model_state : Dict[str, torch.Tensor]
            Model state dictionary
        epoch : int
            Current epoch
        global_step : int
            Global training step
        metrics : Dict[str, float]
            Metrics (loss, accuracy, coherence, etc.)
        optimizer_state : Optional[Dict]
            Optimizer state dictionary
        scheduler_state : Optional[Dict]
            Learning rate scheduler state
        config_hash : Optional[str]
            Hash of training config for reproducibility
        
        Returns
        -------
        Path
            Path to saved checkpoint
        """
        # Get next version
        version = self._get_next_version()
        
        # Create metadata
        metadata = CheckpointMetadata(
            version=version,
            epoch=epoch,
            global_step=global_step,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            git_commit=self._get_git_commit(),
            config_hash=config_hash,
        )
        
        # Create checkpoint file
        checkpoint = {
            'model_state_dict': model_state,
            'metadata': metadata.to_dict(),
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_v{version:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Calculate file size
        model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        metadata.model_size_mb = model_size_mb
        
        # Update metadata tracking
        self.checkpoints[version] = metadata
        self._save_metadata()
        
        # Check if this is the best checkpoint
        if self.best_metric in metrics:
            new_metric = metrics[self.best_metric]
            
            if self.best_checkpoint is None or self._is_better_checkpoint(new_metric, self.best_checkpoint.metrics[self.best_metric]):
                self.best_checkpoint = metadata
                self._save_best_checkpoint()
                logger.info(
                    f"New best checkpoint (v{version:04d}): "
                    f"{self.best_metric}={new_metric:.4f}"
                )
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(
            f"Checkpoint saved (v{version:04d}): epoch={epoch}, "
            f"step={global_step}, size={model_size_mb:.1f}MB"
        )
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, Any]:
        """
        Load checkpoint from disk.
        
        Parameters
        ----------
        checkpoint_path : str | Path
            Path to checkpoint file
        device : torch.device
            Device to load to
        
        Returns
        -------
        Dict[str, Any]
            Checkpoint contents (model_state, optimizer_state, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def load_best_checkpoint(
        self,
        device: torch.device = torch.device('cpu'),
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.
        
        Parameters
        ----------
        device : torch.device
            Device to load to
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Checkpoint contents, or None if no best checkpoint exists
        """
        if self.best_checkpoint is None:
            logger.warning("No best checkpoint found")
            return None
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_v{self.best_checkpoint.version:04d}.pt"
        return self.load_checkpoint(checkpoint_path, device=device)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit"""
        versions = sorted(self.checkpoints.keys())
        
        if len(versions) <= self.max_checkpoints:
            return
        
        # Keep best checkpoint + most recent
        versions_to_keep = set()
        
        if self.best_checkpoint is not None:
            versions_to_keep.add(self.best_checkpoint.version)
        
        # Keep most recent max_checkpoints - 1
        versions_to_keep.update(versions[-(self.max_checkpoints - 1):])
        
        # Remove others
        for version in versions:
            if version not in versions_to_keep:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_v{version:04d}.pt"
                
                try:
                    checkpoint_path.unlink()
                    del self.checkpoints[version]
                    logger.info(f"Removed old checkpoint v{version:04d}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint v{version:04d}: {e}")
        
        self._save_metadata()
    
    def get_checkpoint_list(self) -> List[CheckpointMetadata]:
        """Get list of all checkpoints, sorted by version"""
        return sorted(self.checkpoints.values(), key=lambda x: x.version)
    
    def get_best_checkpoint_info(self) -> Optional[CheckpointMetadata]:
        """Get metadata for best checkpoint"""
        return self.best_checkpoint
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.checkpoint_dir.parent,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def rollback_to_version(self, version: int) -> Path:
        """
        Get path to specific checkpoint version.
        
        Parameters
        ----------
        version : int
            Checkpoint version to rollback to
        
        Returns
        -------
        Path
            Path to checkpoint file
        
        Raises
        ------
        ValueError
            If version doesn't exist
        """
        if version not in self.checkpoints:
            raise ValueError(f"Checkpoint version {version} not found")
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_v{version:04d}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"Rolling back to checkpoint v{version:04d}")
        return checkpoint_path
