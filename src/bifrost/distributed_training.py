"""
Phase 1: Distributed Training Infrastructure

Multi-GPU training with PyTorch DistributedDataParallel (DDP).
- Supports 8x A100 clusters
- Automatic gradient synchronization
- Mixed precision training (FP16/FP32)
- Efficient communication with NCCL

Usage:
    # Multi-node setup
    torchrun --nproc_per_node=8 --nnodes=1 train.py
    
    # Or multi-node across 2 machines
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 train.py
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 train.py
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Callable, Any, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class DistributedTrainerConfig:
    """Configuration for distributed training"""
    
    def __init__(
        self,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        backend: str = "nccl",  # nccl, gloo, mpi
        find_unused_params: bool = False,
        gradient_as_bucket_view: bool = True,
        mixed_precision: bool = True,
        accumulation_steps: int = 1,
    ):
        """
        Initialize distributed training configuration.
        
        Parameters
        ----------
        world_size : Optional[int]
            Total number of processes (auto-detected from env)
        rank : Optional[int]
            Global rank of this process (auto-detected from env)
        local_rank : Optional[int]
            Local rank on this node (auto-detected from env)
        backend : str
            Distributed backend: 'nccl' (GPU), 'gloo' (CPU/GPU), 'mpi'
        find_unused_params : bool
            Allow unused parameters (useful for some architectures)
        gradient_as_bucket_view : bool
            More memory efficient gradient bucketing
        mixed_precision : bool
            Enable automatic mixed precision (FP16/FP32)
        accumulation_steps : int
            Gradient accumulation steps (for larger effective batch)
        """
        # Auto-detect from torch.distributed launch
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
        self.rank = rank or int(os.environ.get("RANK", 0))
        self.local_rank = local_rank or int(os.environ.get("LOCAL_RANK", 0))
        
        self.backend = backend
        self.find_unused_params = find_unused_params
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        
    def __repr__(self) -> str:
        return (
            f"DistributedTrainerConfig(\n"
            f"  world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank},\n"
            f"  is_distributed={self.is_distributed}, is_main={self.is_main_process},\n"
            f"  backend={self.backend}, mixed_precision={self.mixed_precision}\n"
            f")"
        )


class DistributedTrainer:
    """
    Wrapper for distributed training orchestration.
    
    Handles:
    - Process group initialization
    - Model wrapping with DDP
    - Data loader creation with DistributedSampler
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DistributedTrainerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize distributed trainer.
        
        Parameters
        ----------
        model : nn.Module
            Model to wrap for distributed training
        config : Optional[DistributedTrainerConfig]
            Configuration; auto-created if None
        device : Optional[torch.device]
            Device to use; auto-detected if None
        """
        self.config = config or DistributedTrainerConfig()
        
        # Auto-detect device
        if device is None:
            device = torch.device(f"cuda:{self.config.local_rank}" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        logger.info(f"DistributedTrainer initialized on {device}")
        logger.info(self.config)
        
        # Initialize distributed backend if needed
        if self.config.is_distributed:
            self._init_distributed_backend()
        
        # Move model to device and wrap with DDP
        self.model = model.to(device)
        self.wrapped_model = self._wrap_model()
        
        # Mixed precision setup
        self.scaler = None
        if self.config.mixed_precision:
            self.scaler = GradScaler()
    
    def _init_distributed_backend(self) -> None:
        """Initialize PyTorch distributed backend"""
        if not dist.is_available():
            raise RuntimeError("PyTorch distributed is not available")
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                rank=self.config.rank,
                world_size=self.config.world_size,
            )
            logger.info(f"Initialized {self.config.backend} process group: rank={self.config.rank}/{self.config.world_size}")
    
    def _wrap_model(self) -> nn.Module:
        """Wrap model with DDP or return as-is for single GPU"""
        if not self.config.is_distributed:
            return self.model
        
        wrapped = DDP(
            self.model,
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=self.config.find_unused_params,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
        )
        logger.info(f"Model wrapped with DDP on rank {self.config.rank}")
        return wrapped
    
    def get_model(self) -> nn.Module:
        """Get the underlying model (unwrapped for serialization)"""
        if isinstance(self.wrapped_model, DDP):
            return self.wrapped_model.module
        return self.wrapped_model
    
    def create_sampler(self, dataset, shuffle: bool = True, **kwargs) -> DistributedSampler:
        """
        Create a DistributedSampler for the dataset.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sample from
        shuffle : bool
            Whether to shuffle data
        **kwargs
            Additional arguments for DistributedSampler
        
        Returns
        -------
        DistributedSampler
            Sampler that handles distributed sampling
        """
        if not self.config.is_distributed:
            from torch.utils.data import RandomSampler, SequentialSampler
            return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        
        return DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle,
            **kwargs
        )
    
    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader with automatic DistributedSampler.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to load
        batch_size : int
            Batch size per process (not total)
        shuffle : bool
            Whether to shuffle data
        num_workers : int
            Number of worker processes
        pin_memory : bool
            Pin memory for faster GPU transfer
        **kwargs
            Additional arguments for DataLoader
        
        Returns
        -------
        DataLoader
            Data loader with distributed sampling
        """
        sampler = self.create_sampler(dataset, shuffle=shuffle)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    
    def forward_backward(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        accumulation_step: int = 1,
    ) -> Dict[str, float]:
        """
        Mixed precision forward-backward pass with gradient accumulation.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch
        loss_fn : Callable
            Loss function that takes batch and returns scalar loss
        optimizer : torch.optim.Optimizer
            Optimizer for parameter updates
        accumulation_step : int
            Current accumulation step (for scaling loss)
        
        Returns
        -------
        Dict[str, float]
            Loss values and metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.mixed_precision):
            loss, metrics = loss_fn(self.wrapped_model, batch)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.accumulation_steps
        
        # Backward pass with mixed precision
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Optionally synchronize gradients (handled by DDP automatically)
        metrics['loss'] = loss.item()
        
        return metrics
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer, scaler_unscale: bool = False) -> None:
        """
        Perform optimizer step with optional gradient clipping.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to step
        scaler_unscale : bool
            Whether to unscale gradients before clipping (for mixed precision)
        """
        if self.scaler is not None:
            if scaler_unscale:
                self.scaler.unscale_(optimizer)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        name: str = "checkpoint",
    ) -> None:
        """
        Save distributed checkpoint (only on main process).
        
        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory to save checkpoint
        optimizer : Optional[torch.optim.Optimizer]
            Optimizer state to save
        scheduler : Optional[Any]
            Learning rate scheduler state
        epoch : int
            Current epoch
        global_step : int
            Global training step
        metrics : Optional[Dict[str, float]]
            Metrics to save
        name : str
            Checkpoint name (without extension)
        """
        if not self.config.is_main_process:
            return  # Only save on rank 0
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.get_model().state_dict(),
            'config': {
                'world_size': self.config.world_size,
                'rank': self.config.rank,
            },
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if self.config.is_main_process:
            logger.info(f"Checkpoint saved: {checkpoint_path} (epoch={epoch}, step={global_step})")
    
    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load distributed checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str | Path
            Path to checkpoint file
        optimizer : Optional[torch.optim.Optimizer]
            Optimizer to load state into
        scheduler : Optional[Any]
            Scheduler to load state into
        strict : bool
            Require exact key matching
        
        Returns
        -------
        Dict[str, Any]
            Checkpoint metadata (epoch, global_step, metrics)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.get_model().load_state_dict(
            checkpoint['model_state_dict'],
            strict=strict
        )
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'metrics': checkpoint.get('metrics', {}),
        }
        
        if self.config.is_main_process:
            logger.info(f"Checkpoint loaded: {checkpoint_path} (epoch={metadata['epoch']}, step={metadata['global_step']})")
        
        return metadata
    
    def barrier(self) -> None:
        """Synchronize all processes (barrier)"""
        if self.config.is_distributed:
            dist.barrier()
    
    def cleanup(self) -> None:
        """Clean up distributed training"""
        if self.config.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
