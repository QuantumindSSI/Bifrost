#!/usr/bin/env python3
"""
Train PhaseLLM adapter with language modeling objective.

This script trains the SpectralFusion adapter (and optionally SpectralProjector)
on a standard language modeling task using cross-entropy loss.

The base LLM is frozen by default. Only adapter parameters are trained.

Usage:
    python scripts/train_phasellm_lm.py \
        --llm-name gpt2 \
        --dataset wikitext-2-raw-v1 \
        --adapter-layer 6 \
        --spectral-dim 128 \
        --epochs 10 \
        --batch-size 8 \
        --lr 1e-4 \
        --projector-checkpoint checkpoints/uncertainty_calibration_real_audio_continued.pt \
        --save-path checkpoints/phasellm_lm_adapter.pt
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from bifrost.llm_adapter import BifrostEnhancedLLM, SpectralProjector


class LMDataset(Dataset):
    """
    Language modeling dataset with tokenization.
    
    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize with truncation and padding
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Input and target are shifted by 1 for causal LM
        input_ids = encodings["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": encodings["attention_mask"].squeeze(0),
        }


class PhaseLLMTrainer:
    """
    Trainer for PhaseLLM language modeling.
    
    Args:
        model: BifrostEnhancedLLM model
        device: Training device
        lr: Learning rate
        weight_decay: Weight decay for optimizer
    """
    
    def __init__(
        self,
        model: BifrostEnhancedLLM,
        device: str,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.device = device
        
        # Only train adapter parameters (SpectralFusion, optionally SpectralProjector)
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Trainable: {name}")
        
        if not trainable_params:
            raise ValueError("No trainable parameters found. Check freeze_llm setting.")
        
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        self.history = {
            "train_loss": [],
            "train_ppl": [],
            "val_loss": [],
            "val_ppl": [],
            "coherence": [],
        }
        self.current_epoch = 0
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Single training step.
        
        Args:
            input_ids: (B, T) token IDs
            labels: (B, T) target token IDs (shifted)
            attention_mask: (B, T) attention mask
            
        Returns:
            loss: Cross-entropy loss
            perplexity: exp(loss)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Compute cross-entropy loss manually
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss (ignore -100 labels)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        perplexity = math.exp(loss.item())
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), perplexity
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, float, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average loss
            avg_ppl: Average perplexity
            avg_coherence: Average phase coherence
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ppl = 0.0
        total_coherence = 0.0
        n_batches = 0
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Compute cross-entropy loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            perplexity = math.exp(loss.item())
            
            total_loss += loss.item()
            total_ppl += perplexity
            
            # Track coherence if available
            if hasattr(self.model, "coherence_history") and self.model.coherence_history:
                total_coherence += sum(self.model.coherence_history) / len(self.model.coherence_history)
                self.model.coherence_history.clear()
            
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_ppl = total_ppl / n_batches
        avg_coherence = total_coherence / n_batches if n_batches > 0 else 0.0
        
        return avg_loss, avg_ppl, avg_coherence
    
    def save_checkpoint(self, path: Path) -> None:
        """
        Save trained adapter parameters.
        
        Args:
            path: Checkpoint save path
        """
        # Save only adapter parameters
        adapter_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                adapter_state[name] = param.data
        
        checkpoint = {
            "adapter_state_dict": adapter_state,
            "history": self.history,
            "current_epoch": self.current_epoch,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load trained adapter parameters.
        
        Args:
            path: Checkpoint load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load adapter parameters
        adapter_state = checkpoint["adapter_state_dict"]
        model_state = self.model.state_dict()
        
        for name, param in adapter_state.items():
            if name in model_state:
                model_state[name].copy_(param)
        
        self.model.load_state_dict(model_state)
        self.history = checkpoint.get("history", self.history)
        self.current_epoch = checkpoint.get("current_epoch", 0)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train PhaseLLM with Language Modeling Objective"
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to text file for training (one text per line)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2-raw-v1",
        help="HuggingFace dataset name (requires datasets library)",
    )
    parser.add_argument(
        "--adapter-layer",
        type=int,
        default=6,
        help="Which layer to inject adapter",
    )
    parser.add_argument(
        "--spectral-dim",
        type=int,
        default=128,
        help="Spectral dimension",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--projector-checkpoint",
        type=str,
        default=None,
        help="Path to calibrated SpectralProjector checkpoint",
    )
    parser.add_argument(
        "--train-projector",
        action="store_true",
        help="Also train SpectralProjector (not just fusion)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/phasellm_lm_adapter.pt",
        help="Checkpoint save path",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Starting epoch (overrides checkpoint current_epoch)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda)",
    )
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("PhaseLLM Language Modeling Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"LLM: {args.llm_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Adapter layer: {args.adapter_layer}")
    print(f"Spectral dim: {args.spectral_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Train projector: {args.train_projector}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data from file or HuggingFace dataset
    if args.data_path:
        # Load from text file
        print(f"Loading from text file: {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            all_texts = [line.strip() for line in f if line.strip()]
        
        # Split into train/val (90/10)
        split_idx = int(len(all_texts) * 0.9)
        train_texts = all_texts[:split_idx]
        val_texts = all_texts[split_idx:]
    else:
        # Load from HuggingFace dataset
        try:
            from datasets import load_dataset
            print(f"Loading HuggingFace dataset: {args.dataset}")
            dataset = load_dataset(args.dataset, split="train")
            val_dataset = load_dataset(args.dataset, split="validation")
            train_texts = [item["text"] for item in dataset]
            val_texts = [item["text"] for item in val_dataset]
        except ImportError:
            print("Error: datasets library not installed.")
            print("Please install with: pip install datasets")
            print("Or provide --data-path to use a text file instead.")
            return
    
    train_dataset = LMDataset(train_texts, tokenizer, args.max_length)
    val_dataset = LMDataset(val_texts, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Initialize model
    print("Initializing PhaseLLM...")
    model = BifrostEnhancedLLM(
        llm_name=args.llm_name,
        adapter_mode="intermediate",
        adapter_layer=args.adapter_layer,
        spectral_dim=args.spectral_dim,
        freeze_llm=True,
    )
    
    # Load calibrated projector if specified
    if args.projector_checkpoint:
        print(f"Loading calibrated projector from {args.projector_checkpoint}")
        projector_checkpoint = torch.load(args.projector_checkpoint, map_location=device)
        projector_state = projector_checkpoint["projector_state_dict"]
        model.spectral_projector.load_state_dict(projector_state)
        print("Loaded calibrated projector")
        print()
    
    # Optionally train projector
    if args.train_projector:
        for param in model.spectral_projector.parameters():
            param.requires_grad = True
        print("SpectralProjector is trainable")
    else:
        for param in model.spectral_projector.parameters():
            param.requires_grad = False
        print("SpectralProjector is frozen")
    
    # Trainer
    trainer = PhaseLLMTrainer(
        model=model,
        device=device,
        lr=args.lr,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(Path(args.resume_from))
        if args.start_epoch is not None:
            trainer.current_epoch = args.start_epoch
        print(f"Resuming from epoch {trainer.current_epoch}")
        print()
    
    # Train
    print("=" * 60)
    print("Training")
    print("=" * 60)
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(trainer.current_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_ppl = 0.0
        n_batches = 0
        
        # Train
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            loss, ppl = trainer.train_step(input_ids, labels, attention_mask)
            
            epoch_loss += loss
            epoch_ppl += ppl
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_ppl = epoch_ppl / n_batches
        
        # Validate
        val_loss, val_ppl, val_coherence = trainer.validate(val_loader)
        
        # Update history
        trainer.history["train_loss"].append(avg_loss)
        trainer.history["train_ppl"].append(avg_ppl)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_ppl"].append(val_ppl)
        trainer.history["coherence"].append(val_coherence)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train: Loss={avg_loss:.4f}, PPL={avg_ppl:.2f}")
        print(f"  Val:   Loss={val_loss:.4f}, PPL={val_ppl:.2f}, Coherence={val_coherence:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(save_path)
            print(f"  ✓ Saved best checkpoint (Val Loss={best_val_loss:.4f})")
        
        trainer.current_epoch = epoch + 1
        print()
    
    # Final evaluation
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    print()
    
    trainer.load_checkpoint(Path(args.save_path))
    val_loss, val_ppl, val_coherence = trainer.validate(val_loader)
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Final phase coherence: {val_coherence:.4f}")
    print()
    print("Training Complete")


if __name__ == "__main__":
    main()
