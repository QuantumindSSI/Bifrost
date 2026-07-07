"""
Experiment LM-SELECT: Spectral Alpha for Best-of-N Response Selection

Instead of intervening during generation (which proved ineffective on 0.5B model),
use spectral alpha as a SELECTION criterion: generate N responses, compute alpha
for each, select the one with the highest alpha (most "reasoning-like").

This is a more robust use of the spectral alpha signal because:
1. It doesn't disrupt generation (no token injection)
2. It compares alpha ACROSS responses to the same prompt (relative, not absolute)
3. It leverages the finding that reasoning has higher alpha than factual recall

Usage:
    python3 research_dir/experiment_lm_select.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

from bifrost.utils.spectral_monitor import SpectralAlphaMonitor
from research_dir.experiment_lm_monitor import REASONING_TASKS, check_answer


def generate_multiple_responses(
    monitor: SpectralAlphaMonitor,
    prompt: str,
    n_samples: int = 4,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
) -> List[Dict]:
    """Generate multiple responses with alpha monitoring for each."""
    responses = []
    for i in range(n_samples):
        result = monitor.generate_with_monitoring(
            prompt,
            max_new_tokens=max_new_tokens,
            intervention_strategy="none",
            do_sample=True,
            temperature=temperature,
        )
        responses.append({
            "sample_id": i,
            "text": result["text"],
            "mean_alpha": result["mean_alpha"],
            "n_tokens": result["n_tokens_generated"],
            "alpha_trajectory": [
                s["alpha"] for s in result["alpha_trajectory"]
            ],
        })
    return responses


def select_best_response(
    responses: List[Dict],
    strategy: str = "highest_alpha",
) -> Dict:
    """Select the best response using the given strategy."""
    if strategy == "highest_alpha":
        # Select response with highest mean alpha (most reasoning-like)
        return max(responses, key=lambda r: r["mean_alpha"])
    elif strategy == "lowest_alpha":
        # Select response with lowest mean alpha (most factual/compressed)
        return min(responses, key=lambda r: r["mean_alpha"])
    elif strategy == "first":
        # Baseline: just take the first sample
        return responses[0]
    elif strategy == "longest":
        # Baseline: take the longest response (more tokens = more reasoning?)
        return max(responses, key=lambda r: r["n_tokens"])
    elif strategy == "random":
        # Random selection
        return responses[np.random.randint(len(responses))]
    else:
        return responses[0]


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    n_samples: int = 4,
    max_new_tokens: int = 60,
) -> Dict:
    """Run the best-of-N selection experiment."""

    print(f"\nLoading model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    monitor = SpectralAlphaMonitor(
        model, tokenizer, threshold=-999, device=device)

    strategies = ["first", "random", "longest", "highest_alpha", "lowest_alpha"]
    results = {s: [] for s in strategies}

    print(f"\nRunning {len(REASONING_TASKS)} tasks x {n_samples} samples each")

    for i, task in enumerate(REASONING_TASKS):
        prompt = task["prompt"]
        correct_answer = task["answer"]
        category = task["category"]

        print(f"\n  Task {i+1}/{len(REASONING_TASKS)}: [{category}] {prompt[:50]}...")

        # Generate multiple responses
        responses = generate_multiple_responses(
            monitor, prompt, n_samples=n_samples,
            max_new_tokens=max_new_tokens, temperature=0.8)

        # Check correctness of each response
        for r in responses:
            r["is_correct"] = check_answer(r["text"], correct_answer)

        # Print individual samples
        for r in responses:
            status = "✓" if r["is_correct"] else "✗"
            print(f"    sample {r['sample_id']}: {status} alpha={r['mean_alpha']:.4f} "
                  f"text={r['text'][:40]}...")

        # Apply each selection strategy
        for strategy in strategies:
            selected = select_best_response(responses, strategy)
            results[strategy].append({
                "task_id": i,
                "prompt": prompt,
                "category": category,
                "correct_answer": correct_answer,
                "selected_text": selected["text"][:200],
                "is_correct": selected["is_correct"],
                "selected_alpha": selected["mean_alpha"],
                "selected_sample_id": selected["sample_id"],
                "n_correct_samples": sum(1 for r in responses if r["is_correct"]),
                "n_total_samples": len(responses),
                "all_alphas": [r["mean_alpha"] for r in responses],
                "all_correct": [r["is_correct"] for r in responses],
            })

        # Print selection results
        for strategy in strategies:
            r = results[strategy][-1]
            status = "✓" if r["is_correct"] else "✗"
            print(f"    → {strategy:<15} {status} (sample {r['selected_sample_id']})")

    # Analysis
    analysis = analyze_results(results, strategies)
    return {
        "model": model_name,
        "n_samples": n_samples,
        "results": results,
        "analysis": analysis,
    }


def analyze_results(results: Dict, strategies: List[str]) -> Dict:
    """Analyze selection strategy results."""

    analysis = {}

    # Accuracy per strategy
    print("\n--- Accuracy per selection strategy ---")
    analysis["accuracy"] = {}
    for strategy in strategies:
        correct = sum(1 for r in results[strategy] if r["is_correct"])
        total = len(results[strategy])
        acc = correct / total if total > 0 else 0
        analysis["accuracy"][strategy] = {
            "correct": correct, "total": total, "accuracy": acc,
        }
        print(f"  {strategy:<20} {correct}/{total} = {acc:.4f}")

    # Oracle accuracy (if any sample is correct)
    oracle_correct = sum(1 for i, r in enumerate(results["first"])
                         if r["n_correct_samples"] > 0)
    oracle_acc = oracle_correct / len(results["first"])
    print(f"  {'oracle (any correct)':<20} {oracle_correct}/{len(results['first'])} = {oracle_acc:.4f}")
    analysis["oracle_accuracy"] = oracle_acc

    # Accuracy by category
    print("\n--- Accuracy by category ---")
    analysis["accuracy_by_category"] = {}
    for strategy in strategies:
        analysis["accuracy_by_category"][strategy] = {}
        for category in ["math", "logic"]:
            cat_results = [r for r in results[strategy] if r["category"] == category]
            correct = sum(1 for r in cat_results if r["is_correct"])
            total = len(cat_results)
            acc = correct / total if total > 0 else 0
            analysis["accuracy_by_category"][strategy][category] = {
                "correct": correct, "total": total, "accuracy": acc,
            }
            print(f"  {strategy:<20} {category:<6} {correct}/{total} = {acc:.4f}")

    # Does alpha correlate with correctness across all samples?
    print("\n--- Alpha vs correctness (across all samples) ---")
    all_alphas = []
    all_correct = []
    for r in results["first"]:
        all_alphas.extend(r["all_alphas"])
        all_correct.extend(r["all_correct"])

    correct_alphas = [a for a, c in zip(all_alphas, all_correct) if c]
    wrong_alphas = [a for a, c in zip(all_alphas, all_correct) if not c]

    if len(correct_alphas) > 1 and len(wrong_alphas) > 1:
        t_stat, p_value = stats.ttest_ind(correct_alphas, wrong_alphas)
        delta = np.mean(correct_alphas) - np.mean(wrong_alphas)
        analysis["alpha_vs_correctness"] = {
            "correct_mean_alpha": float(np.mean(correct_alphas)),
            "wrong_mean_alpha": float(np.mean(wrong_alphas)),
            "delta": float(delta),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "n_correct": len(correct_alphas),
            "n_wrong": len(wrong_alphas),
        }
        sig = "*" if p_value < 0.05 else "ns"
        print(f"  Correct: {np.mean(correct_alphas):.4f} (n={len(correct_alphas)})")
        print(f"  Wrong:   {np.mean(wrong_alphas):.4f} (n={len(wrong_alphas)})")
        print(f"  Delta:   {delta:+.4f}, p={p_value:.4f} {sig}")

    # Statistical tests: highest_alpha vs first
    print("\n--- Statistical tests ---")
    analysis["statistical_tests"] = {}

    first_correct = [r["is_correct"] for r in results["first"]]

    for strategy in ["highest_alpha", "lowest_alpha", "longest", "random"]:
        strat_correct = [r["is_correct"] for r in results[strategy]]

        # McNemar's test
        b_correct_s_wrong = sum(1 for b, s in zip(first_correct, strat_correct)
                                if b and not s)
        b_wrong_s_correct = sum(1 for b, s in zip(first_correct, strat_correct)
                                if not b and s)

        if b_correct_s_wrong + b_wrong_s_correct > 0:
            mcnemar_stat = (abs(b_correct_s_wrong - b_wrong_s_correct) - 1) ** 2 / \
                           (b_correct_s_wrong + b_wrong_s_correct)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0

        delta = sum(strat_correct) / len(strat_correct) - sum(first_correct) / len(first_correct)

        analysis["statistical_tests"][f"first_vs_{strategy}"] = {
            "first_accuracy": sum(first_correct) / len(first_correct),
            "strategy_accuracy": sum(strat_correct) / len(strat_correct),
            "delta": delta,
            "mcnemar_statistic": float(mcnemar_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "first_correct_strategy_wrong": b_correct_s_wrong,
            "first_wrong_strategy_correct": b_wrong_s_correct,
        }

        sig = "*" if p_value < 0.05 else "ns"
        print(f"  first vs {strategy:<15} delta={delta:+.4f} "
              f"McNemar p={p_value:.4f} {sig}")
        print(f"    first✓→{strategy}✗: {b_correct_s_wrong}, "
              f"first✗→{strategy}✓: {b_wrong_s_correct}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Spectral Alpha for Best-of-N Response Selection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp_lm_select.json")
    args = parser.parse_args()

    print("=" * 70)
    print("LM SELECT: Spectral Alpha for Best-of-N Response Selection")
    print("Generate N responses, select using alpha as quality signal")
    print("=" * 70)

    results = run_experiment(
        model_name=args.model,
        device=args.device,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
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

    analysis = results["analysis"]
    first_acc = analysis["accuracy"]["first"]["accuracy"]
    high_acc = analysis["accuracy"]["highest_alpha"]["accuracy"]
    low_acc = analysis["accuracy"]["lowest_alpha"]["accuracy"]
    oracle_acc = analysis.get("oracle_accuracy", 0)

    print(f"\nAccuracy:")
    print(f"  First sample (baseline):     {first_acc:.4f}")
    print(f"  Highest alpha selection:     {high_acc:.4f}")
    print(f"  Lowest alpha selection:      {low_acc:.4f}")
    print(f"  Oracle (any correct):        {oracle_acc:.4f}")

    alpha_test = analysis["statistical_tests"].get("first_vs_highest_alpha", {})
    alpha_corr = analysis.get("alpha_vs_correctness", {})

    print(f"\nAlpha vs correctness:")
    print(f"  Correct mean alpha: {alpha_corr.get('correct_mean_alpha', 0):.4f}")
    print(f"  Wrong mean alpha:   {alpha_corr.get('wrong_mean_alpha', 0):.4f}")
    print(f"  Delta:              {alpha_corr.get('delta', 0):+.4f}")
    print(f"  p-value:            {alpha_corr.get('p_value', 1):.4f}")

    print(f"\nHighest alpha vs first:")
    print(f"  Delta:   {alpha_test.get('delta', 0):+.4f}")
    print(f"  p-value: {alpha_test.get('p_value', 1):.4f}")

    print(f"\nVerdict:")
    if alpha_test.get("significant", False) and alpha_test.get("delta", 0) > 0:
        print("  POSITIVE: Alpha-based selection significantly improves accuracy.")
    elif high_acc > first_acc + 0.05:
        print("  MIXED: Alpha selection shows improvement but not significant.")
    elif alpha_corr.get("significant", False):
        print("  MIXED: Alpha correlates with correctness but selection doesn't help.")
        print("  Signal may be too weak for reliable selection in this model.")
    else:
        print("  NEGATIVE: Alpha does not predict response quality in this model.")
        print("  The 0.5B model may be too small for spectral alpha to be useful.")
        print("  The Spectral Geometry paper found AUC=1.0 on 7B models —")
        print("  larger models may show stronger alpha-correctness correlation.")


if __name__ == "__main__":
    main()
