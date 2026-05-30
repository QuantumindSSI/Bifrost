#!/usr/bin/env python
"""
Train SpectralAdapter on text classification tasks.

Usage:
    python scripts/train_adapter.py \
        --llm gpt2 \
        --task sentiment \
        --epochs 3 \
        --batch-size 8 \
        --adapter-layer 6
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, "src")

from bifrost.llm_adapter import BifrostEnhancedLLM


def create_synthetic_dataset(task: str, num_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """Create synthetic dataset for testing."""
    if task == "sentiment":
        positive = [
            "This is amazing", "I love this", "Fantastic work", "Great job",
            "Wonderful experience", "Excellent quality", "Very happy",
            "Outstanding performance", "Brilliant idea", "Perfect solution",
        ]
        negative = [
            "This is terrible", "I hate this", "Awful work", "Bad job",
            "Horrible experience", "Poor quality", "Very disappointed",
            "Underwhelming performance", "Terrible idea", "Worst solution",
        ]
        
        texts = []
        labels = []
        for i in range(num_samples):
            if i % 2 == 0:
                texts.append(positive[i % len(positive)])
                labels.append(1)
            else:
                texts.append(negative[i % len(negative)])
                labels.append(0)
        
        return texts, labels
    
    elif task == "topic":
        sports = ["Football game was intense", "Basketball finals tonight", "Soccer championship"]
        tech = ["New AI model released", "Quantum computing breakthrough", "Neural network advances"]
        
        texts = []
        labels = []
        for i in range(num_samples):
            if i % 2 == 0:
                texts.append(sports[i % len(sports)])
                labels.append(0)
            else:
                texts.append(tech[i % len(tech)])
                labels.append(1)
        
        return texts, labels
    
    else:
        raise ValueError(f"Unknown task: {task}")


class AdapterTrainer:
    """Trainer for SpectralAdapter."""
    
    def __init__(
        self,
        model: BifrostEnhancedLLM,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        
        # Only optimize adapter parameters
        adapter_params = [
            {"params": model.spectral_projector.parameters()},
            {"params": model.bifrost.parameters()},
        ]
        
        if hasattr(model, 'spectral_fusion'):
            adapter_params.append({"params": model.spectral_fusion.parameters()})
        
        self.optimizer = torch.optim.AdamW(adapter_params, lr=learning_rate)
        
        # Classification head
        self.classifier = torch.nn.Linear(model.d_model, 2).to(device)
        
        print(f"Model parameters: {model.get_trainable_params()}")
    
    def train_epoch(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        # Create batches
        num_batches = len(texts) // batch_size
        total_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(num_batches):
            batch_texts = texts[i * batch_size:(i + 1) * batch_size]
            batch_labels = labels[i * batch_size:(i + 1) * batch_size]
            
            # Tokenize
            inputs = self.model.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            
            # Forward
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_spectral=True,
            )
            
            # Classification (use first token / CLS)
            logits = self.classifier(outputs.logits[:, 0, :])
            
            # Task loss
            labels_tensor = torch.tensor(batch_labels, device=self.device)
            task_loss = F.cross_entropy(logits, labels_tensor)
            
            # Spectral coherence regularization (encourage high coherence)
            coherence_bonus = -0.01 * (outputs.coherence_score or 0.0)
            
            total_batch_loss = task_loss + coherence_bonus
            
            # Backward
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Stats
            total_loss += task_loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == labels_tensor).sum().item()
            total += len(batch_labels)
            
            if i % 10 == 0:
                print(f"  Batch {i}/{num_batches}: loss={task_loss.item():.4f}, "
                      f"coherence={outputs.coherence_score:.4f}")
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": correct / total,
        }
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        correct = 0
        total = 0
        coherence_scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                inputs = self.model.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)
                
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_spectral=True,
                )
                
                logits = self.classifier(outputs.logits[:, 0, :])
                labels_tensor = torch.tensor(batch_labels, device=self.device)
                
                pred = logits.argmax(dim=-1)
                correct += (pred == labels_tensor).sum().item()
                total += len(batch_labels)
                
                if outputs.coherence_score:
                    coherence_scores.append(outputs.coherence_score)
        
        return {
            "accuracy": correct / total,
            "avg_coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description="Train SpectralAdapter")
    parser.add_argument("--llm", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--task", type=str, default="sentiment", help="Task name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--adapter-layer", type=int, default=6, help="Adapter injection layer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/adapter", help="Save directory")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("SpectralAdapter Training")
    print("=" * 60)
    print(f"LLM: {args.llm}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print(f"Adapter layer: {args.adapter_layer}")
    
    # Create model
    print("\n[Loading Model]")
    model = BifrostEnhancedLLM(
        llm_name=args.llm,
        adapter_mode="intermediate",
        adapter_layer=args.adapter_layer,
        freeze_llm=True,
    )
    
    # Create trainer
    trainer = AdapterTrainer(model, learning_rate=args.lr, device=device)
    
    # Create dataset
    print("\n[Loading Dataset]")
    train_texts, train_labels = create_synthetic_dataset(args.task, num_samples=500)
    val_texts, val_labels = create_synthetic_dataset(args.task, num_samples=100)
    print(f"Train: {len(train_texts)} samples")
    print(f"Val: {len(val_texts)} samples")
    
    # Train
    print("\n[Training]")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_texts, train_labels, args.batch_size)
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
        
        # Eval
        val_metrics = trainer.evaluate(val_texts, val_labels, args.batch_size)
        print(f"  Val: acc={val_metrics['accuracy']:.4f}, coherence={val_metrics['avg_coherence']:.4f}")
        
        # Save best
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_adapter(save_dir / "best_adapter.pt")
    
    # Final generation test
    print("\n[Generation Test]")
    model.eval()
    test_prompts = [
        "This product is",
        "I feel very",
        "The experience was",
    ]
    
    for prompt in test_prompts:
        result = model.generate_with_spectral(prompt, max_length=20, track_coherence=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result['text']}")
        print(f"Coherence: {result.get('avg_coherence', 0.0):.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Adapter saved to: {args.save_dir}/best_adapter.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
