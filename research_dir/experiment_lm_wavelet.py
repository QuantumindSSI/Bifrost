"""
Experiment LM-WAVELET: Wavelet Augmentation for LLM Reasoning

Tests whether adding Haar wavelet multi-scale structure to a transformer
improves reasoning accuracy after fine-tuning.

Compares:
1. Baseline Qwen2.5-0.5B fine-tuned on reasoning tasks
2. Wavelet-augmented Qwen2.5-0.5B fine-tuned on reasoning tasks

Based on Wavelet GPT (arXiv:2411.16720) which shows 2x faster pre-training
and the Spectral Geometry finding that multi-scale structure distinguishes
reasoning from factual recall.

Usage:
    python3 research_dir/experiment_lm_wavelet.py
    python3 research_dir/experiment_lm_wavelet.py --n_epochs 3 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from bifrost.wavelet_augmentation import (
    create_wavelet_augmented_model, count_parameters,
)


# Training data: reasoning tasks with chain-of-thought style answers
TRAIN_TASKS = [
    # Math with steps
    {"prompt": "What is 7 + 3? ", "answer": " 7 + 3 = 10. The answer is 10."},
    {"prompt": "What is 15 - 8? ", "answer": " 15 - 8 = 7. The answer is 7."},
    {"prompt": "What is 4 * 6? ", "answer": " 4 * 6 = 24. The answer is 24."},
    {"prompt": "What is 20 / 4? ", "answer": " 20 / 4 = 5. The answer is 5."},
    {"prompt": "What is 9 + 6? ", "answer": " 9 + 6 = 15. The answer is 15."},
    {"prompt": "What is 12 - 5? ", "answer": " 12 - 5 = 7. The answer is 7."},
    {"prompt": "What is 3 * 8? ", "answer": " 3 * 8 = 24. The answer is 24."},
    {"prompt": "What is 18 / 3? ", "answer": " 18 / 3 = 6. The answer is 6."},
    {"prompt": "What is 11 + 9? ", "answer": " 11 + 9 = 20. The answer is 20."},
    {"prompt": "What is 14 - 6? ", "answer": " 14 - 6 = 8. The answer is 8."},
    {"prompt": "What is 15 * 4? ", "answer": " 15 * 4 = 60. The answer is 60."},
    {"prompt": "What is 25 + 17? ", "answer": " 25 + 17 = 42. The answer is 42."},
    {"prompt": "What is 100 - 37? ", "answer": " 100 - 37 = 63. The answer is 63."},
    {"prompt": "What is 7 * 8? ", "answer": " 7 * 8 = 56. The answer is 56."},
    {"prompt": "What is 45 / 9? ", "answer": " 45 / 9 = 5. The answer is 5."},
    {"prompt": "What is 6 + 7? ", "answer": " 6 + 7 = 13. The answer is 13."},
    {"prompt": "What is 13 - 8? ", "answer": " 13 - 8 = 5. The answer is 5."},
    {"prompt": "What is 9 * 4? ", "answer": " 9 * 4 = 36. The answer is 36."},
    {"prompt": "What is 32 / 8? ", "answer": " 32 / 8 = 4. The answer is 4."},
    {"prompt": "What is 17 + 6? ", "answer": " 17 + 6 = 23. The answer is 23."},
    # Logic with steps
    {"prompt": "If all cats are animals, and Whiskers is a cat, then Whiskers is a: ",
     "answer": " Since all cats are animals, and Whiskers is a cat, Whiskers is an animal. The answer is animal."},
    {"prompt": "If A > B and B > C, then A > C. True or false? ",
     "answer": " If A > B and B > C, then by transitivity A > C. The answer is True."},
    {"prompt": "If today is Monday, what day is it in 3 days? ",
     "answer": " Monday + 1 = Tuesday, + 2 = Wednesday, + 3 = Thursday. The answer is Thursday."},
    {"prompt": "If John is taller than Mary, and Mary is taller than Sue, who is shortest? ",
     "answer": " John > Mary > Sue. Sue is the shortest. The answer is Sue."},
    {"prompt": "How many sides does a triangle have? ",
     "answer": " A triangle has 3 sides by definition. The answer is 3."},
    {"prompt": "A farmer has 17 sheep. All but 9 die. How many are left? ",
     "answer": " All but 9 die means 9 survive. The answer is 9."},
    {"prompt": "If you have 3 apples and eat 1, how many do you have? ",
     "answer": " 3 - 1 = 2. The answer is 2."},
    {"prompt": "What comes next in the sequence: 2, 4, 6, 8, ? ",
     "answer": " The sequence increases by 2. 8 + 2 = 10. The answer is 10."},
    {"prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets? ",
     "answer": " Each machine makes 1 widget in 5 minutes. 100 machines make 100 widgets in 5 minutes. The answer is 5."},
    {"prompt": "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball? ",
     "answer": " Let the ball cost x. Then the bat costs x + 1.00. Total: x + x + 1.00 = 1.10. So 2x = 0.10, x = 0.05. The answer is 0.05."},
]

# Held-out test tasks (different from training)
TEST_TASKS = [
    {"prompt": "What is 8 + 5? ", "answer": "13", "category": "math"},
    {"prompt": "What is 16 - 9? ", "answer": "7", "category": "math"},
    {"prompt": "What is 5 * 7? ", "answer": "35", "category": "math"},
    {"prompt": "What is 24 / 6? ", "answer": "4", "category": "math"},
    {"prompt": "What is 13 + 8? ", "answer": "21", "category": "math"},
    {"prompt": "What is 20 - 12? ", "answer": "8", "category": "math"},
    {"prompt": "What is 6 * 9? ", "answer": "54", "category": "math"},
    {"prompt": "What is 36 / 4? ", "answer": "9", "category": "math"},
    {"prompt": "What is 19 + 14? ", "answer": "33", "category": "math"},
    {"prompt": "What is 50 - 23? ", "answer": "27", "category": "math"},
    {"prompt": "What is 8 * 7? ", "answer": "56", "category": "math"},
    {"prompt": "What is 42 / 7? ", "answer": "6", "category": "math"},
    {"prompt": "What is 22 + 19? ", "answer": "41", "category": "math"},
    {"prompt": "What is 75 - 38? ", "answer": "37", "category": "math"},
    {"prompt": "What is 9 * 6? ", "answer": "54", "category": "math"},
    # Logic
    {"prompt": "If all birds can fly, and a penguin is a bird, can a penguin fly? ",
     "answer": "yes", "category": "logic"},
    {"prompt": "If X > Y and Y > Z, is X > Z? ",
     "answer": "yes", "category": "logic"},
    {"prompt": "If today is Wednesday, what day is it in 4 days? ",
     "answer": "sunday", "category": "logic"},
    {"prompt": "If Alice is older than Bob, and Bob is older than Carol, who is youngest? ",
     "answer": "carol", "category": "logic"},
    {"prompt": "How many corners does a square have? ",
     "answer": "4", "category": "logic"},
    {"prompt": "A baker has 12 loaves. All but 5 are sold. How many remain? ",
     "answer": "5", "category": "logic"},
    {"prompt": "If you have 5 oranges and give away 2, how many do you have? ",
     "answer": "3", "category": "logic"},
    {"prompt": "What comes next: 3, 6, 9, 12, ? ",
     "answer": "15", "category": "logic"},
    {"prompt": "If 3 workers take 3 hours to build 3 walls, how long for 6 workers to build 6 walls? ",
     "answer": "3", "category": "logic"},
    {"prompt": "A book and pen cost $2.50. The book costs $2.00 more than the pen. How much is the pen? ",
     "answer": "0.25", "category": "logic"},
]


class ReasoningDataset(Dataset):
    """Dataset for fine-tuning on reasoning tasks."""

    def __init__(self, tasks: List[dict], tokenizer, max_length: int = 128):
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        prompt = task["prompt"]
        answer = task["answer"]

        # Full text: prompt + answer
        full_text = prompt + answer
        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            return_tensors="pt", padding="max_length"
        )["input_ids"].squeeze(0)

        # Prompt-only ids (for masking)
        prompt_ids = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        prompt_len = len(prompt_ids)

        # Labels: -100 for prompt tokens (don't compute loss), actual ids for answer
        labels = full_ids.clone()
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": (full_ids != self.tokenizer.pad_token_id).long(),
        }


def check_answer(generated_text: str, correct_answer: str) -> bool:
    """Check if generated text contains the correct answer."""
    gen_lower = generated_text.lower().strip()
    ans_lower = correct_answer.lower().strip()
    first_part = gen_lower[:150]

    if ans_lower in first_part:
        return True

    # Numeric check
    if ans_lower.replace(".", "").replace("-", "").isdigit():
        numbers = re.findall(r'\d+\.?\d*', gen_lower[:300])
        for num in numbers:
            try:
                if float(num) == float(ans_lower):
                    return True
            except ValueError:
                continue

    return False


def fine_tune_model(
    model: nn.Module,
    tokenizer,
    train_tasks: List[dict],
    n_epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
    device: str = "cpu",
    max_length: int = 128,
) -> Dict:
    """Fine-tune a model on reasoning tasks."""

    dataset = ReasoningDataset(train_tasks, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Only train wavelet parameters + layernorm + lm_head
    # Freeze base model weights for efficiency
    for name, param in model.named_parameters():
        if "wavelet" in name or "gate" in name or "output_scale" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Also unfreeze the LM head and embedding for better adaptation
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = True

    params = count_parameters(model)
    print(f"  Trainable parameters: {params['trainable_M']:.2f}M / {params['total_M']:.2f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    model.train()
    losses = []

    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels,
                            attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    model.eval()
    return {"final_loss": losses[-1], "losses": losses, "params": params}


def evaluate_model(
    model: nn.Module,
    tokenizer,
    test_tasks: List[dict],
    device: str = "cpu",
    max_new_tokens: int = 60,
) -> Dict:
    """Evaluate model on test tasks."""

    results = []
    correct_count = 0

    for i, task in enumerate(test_tasks):
        prompt = task["prompt"]
        correct_answer = task["answer"]
        category = task["category"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output[0, input_ids.shape[1]:], skip_special_tokens=True)
        is_correct = check_answer(generated_text, correct_answer)

        if is_correct:
            correct_count += 1

        results.append({
            "task_id": i,
            "prompt": prompt,
            "category": category,
            "correct_answer": correct_answer,
            "generated_text": generated_text[:200],
            "is_correct": is_correct,
        })

        status = "✓" if is_correct else "✗"
        print(f"    Task {i+1}/{len(test_tasks)}: [{category}] {status} "
              f"{generated_text[:40]}...")

    accuracy = correct_count / len(test_tasks)
    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(test_tasks),
        "results": results,
    }


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    n_epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
) -> Dict:
    """Run the wavelet augmentation experiment."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Step 1: Evaluate pre-fine-tuning baseline
    print("\n--- Step 1: Pre-fine-tuning evaluation ---")
    print("Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    baseline_model.eval()

    print("Evaluating baseline (no fine-tuning)...")
    pre_ft_results = evaluate_model(
        baseline_model, tokenizer, TEST_TASKS, device=device)

    print(f"\n  Pre-FT accuracy: {pre_ft_results['accuracy']:.4f}")

    # Free memory
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 2: Fine-tune baseline (no wavelet)
    print("\n--- Step 2: Fine-tuning baseline model ---")
    baseline_ft_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    t0 = time.time()
    baseline_ft_info = fine_tune_model(
        baseline_ft_model, tokenizer, TRAIN_TASKS,
        n_epochs=n_epochs, lr=lr, batch_size=batch_size, device=device)
    baseline_ft_time = time.time() - t0
    print(f"  Fine-tuning time: {baseline_ft_time:.1f}s")

    print("  Evaluating fine-tuned baseline...")
    baseline_ft_results = evaluate_model(
        baseline_ft_model, tokenizer, TEST_TASKS, device=device)
    print(f"\n  Baseline FT accuracy: {baseline_ft_results['accuracy']:.4f}")

    # Free memory
    del baseline_ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 3: Create and fine-tune wavelet-augmented model
    print("\n--- Step 3: Fine-tuning wavelet-augmented model ---")
    wavelet_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    wavelet_model = create_wavelet_augmented_model(
        wavelet_model, n_scales=3,
        use_wavelet_after_attn=False,  # Only after MLP for efficiency
        use_wavelet_after_mlp=True,
    )

    t0 = time.time()
    wavelet_ft_info = fine_tune_model(
        wavelet_model, tokenizer, TRAIN_TASKS,
        n_epochs=n_epochs, lr=lr, batch_size=batch_size, device=device)
    wavelet_ft_time = time.time() - t0
    print(f"  Fine-tuning time: {wavelet_ft_time:.1f}s")

    print("  Evaluating wavelet-augmented model...")
    wavelet_ft_results = evaluate_model(
        wavelet_model, tokenizer, TEST_TASKS, device=device)
    print(f"\n  Wavelet FT accuracy: {wavelet_ft_results['accuracy']:.4f}")

    # Step 4: Analysis
    print("\n--- Step 4: Analysis ---")
    analysis = {
        "pre_ft": {
            "accuracy": pre_ft_results["accuracy"],
            "correct": pre_ft_results["correct"],
            "total": pre_ft_results["total"],
        },
        "baseline_ft": {
            "accuracy": baseline_ft_results["accuracy"],
            "correct": baseline_ft_results["correct"],
            "total": baseline_ft_results["total"],
            "ft_time": baseline_ft_time,
            "final_loss": baseline_ft_info["final_loss"],
            "trainable_params": baseline_ft_info["params"]["trainable_M"],
        },
        "wavelet_ft": {
            "accuracy": wavelet_ft_results["accuracy"],
            "correct": wavelet_ft_results["correct"],
            "total": wavelet_ft_results["total"],
            "ft_time": wavelet_ft_time,
            "final_loss": wavelet_ft_info["final_loss"],
            "trainable_params": wavelet_ft_info["params"]["trainable_M"],
        },
    }

    # Accuracy by category
    for cond_name, cond_results in [("baseline_ft", baseline_ft_results),
                                     ("wavelet_ft", wavelet_ft_results)]:
        analysis[cond_name]["by_category"] = {}
        for category in ["math", "logic"]:
            cat = [r for r in cond_results["results"] if r["category"] == category]
            correct = sum(1 for r in cat if r["is_correct"])
            total = len(cat)
            analysis[cond_name]["by_category"][category] = {
                "correct": correct, "total": total,
                "accuracy": correct / total if total > 0 else 0,
            }

    # Delta
    delta = wavelet_ft_results["accuracy"] - baseline_ft_results["accuracy"]
    analysis["delta"] = delta

    # McNemar's test
    from scipy import stats
    b_correct = [r["is_correct"] for r in baseline_ft_results["results"]]
    w_correct = [r["is_correct"] for r in wavelet_ft_results["results"]]
    b_c_w = sum(1 for b, w in zip(b_correct, w_correct) if b and not w)
    b_w_c = sum(1 for b, w in zip(b_correct, w_correct) if not b and w)
    if b_c_w + b_w_c > 0:
        mcnemar = (abs(b_c_w - b_w_c) - 1) ** 2 / (b_c_w + b_w_c)
        p_value = 1 - stats.chi2.cdf(mcnemar, 1)
    else:
        mcnemar = 0
        p_value = 1.0
    analysis["mcnemar"] = {
        "statistic": float(mcnemar),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "baseline_correct_wavelet_wrong": b_c_w,
        "baseline_wrong_wavelet_correct": b_w_c,
    }

    return {
        "model": model_name,
        "n_train": len(TRAIN_TASKS),
        "n_test": len(TEST_TASKS),
        "analysis": analysis,
        "baseline_results": baseline_ft_results,
        "wavelet_results": wavelet_ft_results,
        "pre_ft_results": pre_ft_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Wavelet Augmentation for LLM Reasoning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp_lm_wavelet.json")
    args = parser.parse_args()

    print("=" * 70)
    print("LM WAVELET: Wavelet Augmentation for LLM Reasoning")
    print("Tests if wavelet multi-scale structure improves reasoning")
    print("=" * 70)

    results = run_experiment(
        model_name=args.model,
        device=args.device,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)

    a = results["analysis"]
    print(f"\nAccuracy:")
    print(f"  Pre-fine-tuning:        {a['pre_ft']['accuracy']:.4f}")
    print(f"  Baseline (FT):          {a['baseline_ft']['accuracy']:.4f}")
    print(f"  Wavelet-augmented (FT): {a['wavelet_ft']['accuracy']:.4f}")
    print(f"  Delta (wavelet - base): {a['delta']:+.4f}")

    print(f"\nBy category:")
    for cat in ["math", "logic"]:
        b = a["baseline_ft"]["by_category"].get(cat, {})
        w = a["wavelet_ft"]["by_category"].get(cat, {})
        print(f"  {cat}: baseline={b.get('accuracy', 0):.4f} "
              f"wavelet={w.get('accuracy', 0):.4f}")

    print(f"\nFine-tuning time:")
    print(f"  Baseline: {a['baseline_ft']['ft_time']:.1f}s")
    print(f"  Wavelet:  {a['wavelet_ft']['ft_time']:.1f}s")

    print(f"\nStatistical test (McNemar):")
    print(f"  p-value: {a['mcnemar']['p_value']:.4f}")
    print(f"  baseline✓→wavelet✗: {a['mcnemar']['baseline_correct_wavelet_wrong']}")
    print(f"  baseline✗→wavelet✓: {a['mcnemar']['baseline_wrong_wavelet_correct']}")

    print(f"\nVerdict:")
    if a["mcnemar"]["significant"] and a["delta"] > 0:
        print("  POSITIVE: Wavelet augmentation significantly improves reasoning.")
    elif a["delta"] > 0.05:
        print("  MIXED: Wavelet shows improvement but not statistically significant.")
    elif a["delta"] > 0:
        print(f"  MARGINAL: Wavelet shows small improvement (+{a['delta']:.4f}).")
    elif a["delta"] < -0.05:
        print("  NEGATIVE: Wavelet augmentation HURTS reasoning accuracy.")
    else:
        print("  NEUTRAL: Wavelet augmentation has no effect on reasoning accuracy.")


if __name__ == "__main__":
    main()
