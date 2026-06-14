"""
Phase 1 Example: Distributed Training with BifröstPipeline

This script demonstrates how to use the new distributed training infrastructure
for multi-GPU training on 8x A100 clusters.

Usage:
    # Single GPU (for testing)
    python train_phase1.py --batch_size=32 --epochs=10
    
    # Multi-GPU (8x A100)
    torchrun --nproc_per_node=8 train_phase1.py --batch_size=256 --epochs=100
    
    # Multi-node (2 nodes with 8 GPUs each = 16 GPUs total)
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 train_phase1.py ...
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 train_phase1.py ...
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bifrost.pipeline import BifrostPipeline
from bifrost.training import ContrastiveCoherenceLoss
from bifrost.distributed_training import DistributedTrainerConfig, DistributedTrainer
from bifrost.checkpoint_manager import CheckpointManager
from bifrost.evaluation import Phase1Evaluator
from bifrost.data.loader import create_audio_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase1Trainer:
    """
    Complete Phase 1 training pipeline with distributed support.
    
    Key features:
    - Distributed training (DDP)
    - Mixed precision (FP16/FP32)
    - Gradient accumulation
    - Automatic checkpointing
    - Comprehensive evaluation
    """
    
    def __init__(
        self,
        model: BifrostPipeline,
        distributed_config: DistributedTrainerConfig,
        checkpoint_dir: Path = Path("./checkpoints"),
        log_dir: Path = Path("./logs"),
    ):
        self.model = model
        self.config = distributed_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Initialize distributed trainer
        self.trainer = DistributedTrainer(model, distributed_config)
        
        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=5,
            best_metric="val_loss",
            best_metric_mode="min",
        )
        
        # Evaluation
        self.evaluator = Phase1Evaluator()
        
        # Loss and optimizer
        self.loss_fn = ContrastiveCoherenceLoss()
        self.optimizer = optim.AdamW(
            self.trainer.wrapped_model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=100_000,  # Adjust based on total steps
        )
        
        logger.info(f"Phase 1 Trainer initialized: {distributed_config}")
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Parameters
        ----------
        dataloader : DataLoader
            Training dataloader
        epoch : int
            Current epoch
        global_step : int
            Current global training step
        
        Returns
        -------
        Dict[str, float]
            Epoch metrics
        """
        self.trainer.wrapped_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Forward-backward pass
            metrics = self.trainer.forward_backward(
                batch,
                loss_fn=self._loss_wrapper,
                optimizer=self.optimizer,
                accumulation_step=batch_idx % self.config.accumulation_steps + 1,
            )
            
            total_loss += metrics['loss']
            num_batches += 1
            
            # Optimizer step every accumulation_steps
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.trainer.wrapped_model.parameters(),
                    max_norm=1.0
                )
                self.trainer.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                self.scheduler.step()
                global_step += 1
            
            # Logging (only on main process)
            if self.config.is_main_process and batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {metrics['loss']:.4f} | Step: {global_step}"
                )
        
        # Synchronize all processes
        self.trainer.barrier()
        
        avg_loss = total_loss / num_batches
        
        if self.config.is_main_process:
            logger.info(f"Epoch {epoch} completed: avg_loss={avg_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'global_step': global_step,
        }
    
    def _loss_wrapper(self, model, batch: Dict) -> tuple:
        """
        Wrapper for loss computation compatible with forward_backward.
        
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            (loss, metrics_dict)
        """
        # Unpack batch
        sig_t = batch['audio']  # [batch, time]
        meta = batch['metadata']  # Dict with duration, sr, etc.
        
        # Forward pass
        bound, coherence = model(sig_t, meta)
        
        # Generate noise for contrastive loss
        noise = torch.randn_like(sig_t)
        bound_noise, coherence_noise = model(noise, meta)
        
        # Compute loss
        loss = self.loss_fn(coherence, coherence_noise)
        
        return loss, {'loss': loss.item()}
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Parameters
        ----------
        dataloader : DataLoader
            Validation dataloader
        
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        self.trainer.wrapped_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                loss, _ = self._loss_wrapper(self.trainer.wrapped_model, batch)
                total_loss += loss.item()
                num_batches += 1
        
        self.trainer.barrier()
        
        avg_loss = total_loss / num_batches
        
        if self.config.is_main_process:
            logger.info(f"Validation: loss={avg_loss:.4f}")
        
        return {'val_loss': avg_loss}
    
    def train(
        self,
        train_dataloader,
        val_dataloader,
        epochs: int,
        start_epoch: int = 0,
        start_global_step: int = 0,
    ) -> None:
        """
        Complete training loop.
        
        Parameters
        ----------
        train_dataloader : DataLoader
            Training dataloader
        val_dataloader : DataLoader
            Validation dataloader
        epochs : int
            Number of epochs to train
        start_epoch : int
            Starting epoch (for resuming)
        start_global_step : int
            Starting global step (for resuming)
        """
        global_step = start_global_step
        
        for epoch in range(start_epoch, epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, epoch, global_step)
            global_step = train_metrics['global_step']
            
            # Validate
            val_metrics = self.evaluate(val_dataloader)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Save checkpoint
            if self.config.is_main_process:
                self.checkpoint_mgr.save_checkpoint(
                    model_state=self.trainer.get_model().state_dict(),
                    epoch=epoch,
                    global_step=global_step,
                    metrics=all_metrics,
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict(),
                )
        
        self.trainer.cleanup()
        
        if self.config.is_main_process:
            logger.info("Training complete!")
            
            # Log best checkpoint info
            best_ckpt = self.checkpoint_mgr.get_best_checkpoint_info()
            if best_ckpt:
                logger.info(
                    f"Best checkpoint: v{best_ckpt.version:04d} | "
                    f"val_loss={best_ckpt.metrics.get('val_loss', 'N/A')}"
                )


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Enable mixed precision")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Setup distributed config
    dist_config = DistributedTrainerConfig(
        mixed_precision=args.mixed_precision,
        accumulation_steps=args.accumulation_steps,
    )
    
    # Create model
    model = BifrostPipeline(d_model=args.d_model)
    
    # Create trainer
    trainer = Phase1Trainer(
        model=model,
        distributed_config=dist_config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    # Create dataloaders
    # Note: This is a placeholder; you'll need actual data
    logger.warning("Using placeholder dataloaders; replace with real data")
    
    train_dataloader = create_audio_dataloader(
        dataset_dir="./sample_data",
        batch_size=args.batch_size,
        num_workers=4,
        distributed_sampler=trainer.trainer.is_distributed,
    )
    
    val_dataloader = create_audio_dataloader(
        dataset_dir="./sample_data",
        batch_size=args.batch_size,
        num_workers=4,
        distributed_sampler=trainer.trainer.is_distributed,
        shuffle=False,
    )
    
    # Train
    trainer.train(
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
