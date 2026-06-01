#!/usr/bin/env python3
"""
Train PhaseLLM end-to-end for generation tasks.

This script trains the SpectralFusion adapter jointly with the LLM for
generation tasks, using both language modeling loss and spectral coherence loss.

The base LLM can be frozen or fine-tuned depending on configuration.

Usage:
    python scripts/train_phasellm_end_to_end.py \
        --llm-name gpt2 \
        --dataset wikitext-2-raw-v1 \
        --adapter-layer 6 \
        --spectral-dim 128 \
        --epochs 10 \
        --batch-size 8 \
        --lr 1e-4 \
        --projector-checkpoint checkpoints/uncertainty_calibration_real_audio_continued.pt \
        --coherence-weight 0.1 \
        --save-path checkpoints/phasellm_end_to_end.pt
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
from datasets import load_dataset

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


class PhaseLLMEndToEndTrainer:
    """
    End-to-end trainer for PhaseLLM generation.
    
    Combines language modeling loss with spectral coherence loss.
    
    Args:
        model: BifrostEnhancedLLM model
        device: Training device
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        coherence_weight: Weight for spectral coherence loss
    """
    
    def __init__(
        self,
        model: BifrostEnhancedLLM,
        device: str,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        coherence_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.coherence_weight = coherence_weight
        
        # Collect trainable parameters
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Trainable: {name}")
        
        if not trainable_params:
            raise ValueError("No trainable parameters found. Check freeze_llm setting.")
        
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        self.history = {
            "train_lm_loss": [],
            "train_coherence_loss": [],
            "train_total_loss": [],
            "train_ppl": [],
            "val_lm_loss": [],
            "val_coherence_loss": [],
            "val_total_loss": [],
            "val_ppl": [],
            "coherence": [],
        }
        self.current_epoch = 0
    
    def compute_coherence_loss(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute spectral coherence loss.
        
        Encourages high phase coherence for structured sequences.
        
        Args:
            hidden_states: (B, T, d_model) hidden states
            
        Returns:
            coherence_loss: Scalar loss
            avg_coherence: Average coherence value
        """
        # Project to spectral space
        spectral, _ = self.model.spectral_projector(hidden_states)
        
        # Compute phase coherence
        # Coherence = |mean(e^(i*phase))| across time
        phase_complex = torch.exp(1j * spectral.phase)  # (B, T, spectral_dim)
        coherence = torch.abs(phase_complex.mean(dim=1))  # (B, spectral_dim)
        avg_coherence = coherence.mean().item()
        
        # Loss: maximize coherence (minimize negative coherence)
        coherence_loss = -coherence.mean()
        
        return coherence_loss, avg_coherence
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """
        Single training step with combined losses.
        
        Args:
            input_ids: (B, T) token IDs
            labels: (B, T) target token IDs (shifted)
            attention_mask: (B, T) attention mask
            
        Returns:
            lm_loss: Language modeling loss
            coherence_loss: Spectral coherence loss
            total_loss: Combined loss
            perplexity: exp(lm_loss)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass for LM loss
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        lm_loss = outputs.loss
        perplexity = math.exp(lm_loss.item())
        
        # Get hidden states for coherence loss
        # Extract from the adapter layer
        with torch.no_grad():
            # Get base LLM embeddings
            inputs_embeds = self.model.llm.get_input_embeddings()(input_ids)
            
            # Get hidden states from forward pass
            hidden_states = self.model.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[self.model.adapter_layer]
        
        # Compute coherence loss
        coherence_loss, avg_coherence = self.compute_coherence_loss(hidden_states)
        
        # Combined loss
        total_loss = lm_loss + self.coherence_weight * coherence_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track coherence
        self.model.coherence_history.append(avg_coherence)
        
        return lm_loss.item(), coherence_loss.item(), total_loss.item(), perplexity
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, float, float, float, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_lm_loss: Average LM loss
            avg_coherence_loss: Average coherence loss
            avg_total_loss: Average total loss
            avg_ppl: Average perplexity
            avg_coherence: Average phase coherence
        """
        self.model.eval()
        
        total_lm_loss = 0.0
        total_coherence_loss = 0.0
        total_total_loss = 0.0
        total_ppl = 0.0
        total_coherence = 0.0
        n_batches = 0
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # LM loss
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            
            lm_loss = outputs.loss
            perplexity = math.exp(lm_loss.item())
            
            # Coherence loss
            hidden_states = self.model.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[self.model.adapter_layer]
            
            coherence_loss, avg_coherence = self.compute_coherence_loss(hidden_states)
            
            total_lm_loss += lm_loss.item()
            total_coherence_loss += coherence_loss.item()
            total_total_loss += (lm_loss.item() + self.coherence_weight * coherence_loss.item())
            total_ppl += perplexity
            total_coherence += avg_coherence
            
            n_batches += 1
        
        avg_lm_loss = total_lm_loss / n_batches
        avg_coherence_loss = total_coherence_loss / n_batches
        avg_total_loss = total_total_loss / n_batches
        avg_ppl = total_ppl / n_batches
        avg_coherence = total_coherence / n_batches
        
        return avg_lm_loss, avg_coherence_loss, avg_total_loss, avg_ppl, avg_coherence
    
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
            "coherence_weight": self.coherence_weight,
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
        self.coherence_weight = checkpoint.get("coherence_weight", self.coherence_weight)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train PhaseLLM End-to-End for Generation"
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2-raw-v1",
        help="HuggingFace dataset name",
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
        "--coherence-weight",
        type=float,
        default=0.1,
        help="Weight for spectral coherence loss",
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
        "--freeze-llm",
        action="store_true",
        default=True,
        help="Freeze base LLM weights",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/phasellm_end_to_end.pt",
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
    print("PhaseLLM End-to-End Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"LLM: {args.llm_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Adapter layer: {args.adapter_layer}")
    print(f"Spectral dim: {args.spectral_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Coherence weight: {args.coherence_weight}")
    print(f"Train projector: {args.train_projector}")
    print(f"Freeze LLM: {args.freeze_llm}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, split="train")
    val_dataset = load_dataset(args.dataset, split="validation")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_texts = [item["text"] for item in dataset]
    val_texts = [item["text"] for item in val_dataset]
    
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
        freeze_llm=args.freeze_llm,
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
    trainer = PhaseLLMEndToEndTrainer(
        model=model,
        device=device,
        lr=args.lr,
        coherence_weight=args.coherence_weight,
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
        epoch_lm_loss = 0.0
        epoch_coherence_loss = 0.0
        epoch_total_loss = 0.0
        epoch_ppl = 0.0
        n_batches = 0
        
        # Train
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            lm_loss, coh_loss, total_loss, ppl = trainer.train_step(
                input_ids, labels, attention_mask
            )
            
            epoch_lm_loss += lm_loss
            epoch_coherence_loss += coh_loss
            epoch_total_loss += total_loss
            epoch_ppl += ppl
            n_batches += 1
        
        avg_lm_loss = epoch_lm_loss / n_batches
        avg_coherence_loss = epoch_coherence_loss / n_batches
        avg_total_loss = epoch_total_loss / n_batches
        avg_ppl = epoch_ppl / n_batches
        
        # Validate
        val_lm_loss, val_coh_loss, val_total_loss, val_ppl, val_coherence = trainer.validate(val_loader)
        
        # Update history
        trainer.history["train_lm_loss"].append(avg_lm_loss)
        trainer.history["train_coherence_loss"].append(avg_coherence_loss)
        trainer.history["train_total_loss"].append(avg_total_loss)
        trainer.history["train_ppl"].append(avg_ppl)
        trainer.history["val_lm_loss"].append(val_lm_loss)
        trainer.history["val_coherence_loss"].append(val_coh_loss)
        trainer.history["val_total_loss"].append(val_total_loss)
        trainer.history["val_ppl"].append(val_ppl)
        trainer.history["coherence"].append(val_coherence)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train: LM={avg_lm_loss:.4f}, Coh={avg_coherence_loss:.4f}, Total={avg_total_loss:.4f}, PPL={avg_ppl:.2f}")
        print(f"  Val:   LM={val_lm_loss:.4f}, Coh={val_coh_loss:.4f}, Total={val_total_loss:.4f}, PPL={val_ppl:.2f}, Coherence={val_coherence:.4f}")
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
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
    val_lm_loss, val_coh_loss, val_total_loss, val_ppl, val_coherence = trainer.validate(val_loader)
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Final phase coherence: {val_coherence:.4f}")
    print()
    print("Training Complete")


if __name__ == "__main__":
    main()
