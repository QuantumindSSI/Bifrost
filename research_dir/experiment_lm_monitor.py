"""
Experiment LM-MONITOR: Spectral Alpha Monitoring for LLM Reasoning

Tests whether real-time spectral alpha monitoring can:
1. Detect reasoning breakdown during generation
2. Improve reasoning accuracy via intervention (CoT continuation)

Compares three conditions:
- Baseline: No monitoring, no intervention
- Monitored: Monitoring only (no intervention) — measures detection accuracy
- Intervened: Monitoring + CoT continuation when breakdown detected

Usage:
    python3 research_dir/experiment_lm_monitor.py
    python3 research_dir/experiment_lm_monitor.py --model Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

from bifrost.utils.spectral_monitor import (
    SpectralAlphaMonitor, calibrate_threshold,
)


# Reasoning tasks with known correct answers
# Tasks designed to be challenging for a 0.5B model
REASONING_TASKS = [
    # Math
    {"prompt": "What is 7 + 3? ", "answer": "10", "category": "math"},
    {"prompt": "What is 15 - 8? ", "answer": "7", "category": "math"},
    {"prompt": "What is 4 * 6? ", "answer": "24", "category": "math"},
    {"prompt": "What is 20 / 4? ", "answer": "5", "category": "math"},
    {"prompt": "What is 9 + 6? ", "answer": "15", "category": "math"},
    {"prompt": "What is 12 - 5? ", "answer": "7", "category": "math"},
    {"prompt": "What is 3 * 8? ", "answer": "24", "category": "math"},
    {"prompt": "What is 18 / 3? ", "answer": "6", "category": "math"},
    {"prompt": "What is 11 + 9? ", "answer": "20", "category": "math"},
    {"prompt": "What is 14 - 6? ", "answer": "8", "category": "math"},
    {"prompt": "What is 15 * 4? ", "answer": "60", "category": "math"},
    {"prompt": "What is 25 + 17? ", "answer": "42", "category": "math"},
    {"prompt": "What is 100 - 37? ", "answer": "63", "category": "math"},
    {"prompt": "What is 7 * 8? ", "answer": "56", "category": "math"},
    {"prompt": "What is 45 / 9? ", "answer": "5", "category": "math"},
    # Logic
    {"prompt": "If all cats are animals, and Whiskers is a cat, then Whiskers is a: ",
     "answer": "animal", "category": "logic"},
    {"prompt": "If A > B and B > C, then A > C. True or false? ",
     "answer": "true", "category": "logic"},
    {"prompt": "If today is Monday, what day is it in 3 days? ",
     "answer": "thursday", "category": "logic"},
    {"prompt": "If John is taller than Mary, and Mary is taller than Sue, who is shortest? ",
     "answer": "sue", "category": "logic"},
    {"prompt": "How many sides does a triangle have? ",
     "answer": "3", "category": "logic"},
    {"prompt": "A farmer has 17 sheep. All but 9 die. How many are left? ",
     "answer": "9", "category": "logic"},
    {"prompt": "If you have 3 apples and eat 1, how many do you have? ",
     "answer": "2", "category": "logic"},
    {"prompt": "What comes next in the sequence: 2, 4, 6, 8, ? ",
     "answer": "10", "category": "logic"},
    {"prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets? ",
     "answer": "5", "category": "logic"},
    {"prompt": "A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball? ",
     "answer": "0.05", "category": "logic"},
]

# Factual recall tasks for calibration
FACTUAL_PROMPTS = [
    "The capital of France is ",
    "The largest planet in our solar system is ",
    "Water boils at what temperature in Celsius? ",
    "The chemical symbol for gold is ",
    "How many continents are there? ",
    "The capital of Japan is ",
    "The largest ocean is ",
    "How many planets are in our solar system? ",
]


def check_answer(generated_text: str, correct_answer: str) -> bool:
    """Check if the generated text contains the correct answer.

    Uses simple string matching — the answer should appear in the first
    few words of the response.
    """
    # Normalize: lowercase, strip punctuation
    gen_lower = generated_text.lower().strip()
    ans_lower = correct_answer.lower().strip()

    # Check if answer appears in the first 50 characters
    first_part = gen_lower[:100]

    # Direct match
    if ans_lower in first_part:
        return True

    # For numeric answers, check if the number appears
    if ans_lower.replace(".", "").replace("-", "").isdigit():
        # Look for the number in the response
        numbers = re.findall(r'\d+\.?\d*', gen_lower[:200])
        for num in numbers:
            if float(num) == float(ans_lower):
                return True

    return False


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    max_new_tokens: int = 80,
) -> Dict:
    """Run the monitoring experiment."""

    print(f"\nLoading model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Step 1: Calibrate threshold
    print("\n--- Step 1: Calibrating threshold ---")
    reasoning_cal_prompts = [t["prompt"] for t in REASONING_TASKS[:10]]
    calibration = calibrate_threshold(
        model, tokenizer, reasoning_cal_prompts, FACTUAL_PROMPTS, device=device)
    threshold = calibration["suggested_threshold"]

    print(f"  Reasoning mean alpha: {calibration['reasoning_mean_alpha']:.4f}")
    print(f"  Factual mean alpha:   {calibration['factual_mean_alpha']:.4f}")
    print(f"  Suggested threshold:  {threshold:.4f}")
    print(f"  Separation:           {calibration['separation']:.4f}")

    if abs(calibration["separation"]) < 0.01:
        print("  WARNING: Very small separation between reasoning and factual alpha.")
        print("  Monitoring may not be effective for this model.")

    # Step 2: Run three conditions
    monitor = SpectralAlphaMonitor(
        model, tokenizer, threshold=threshold, device=device)

    conditions = ["baseline", "monitored", "cot_intervention", "step_by_step"]
    results = {cond: [] for cond in conditions}

    print(f"\n--- Step 2: Running {len(REASONING_TASKS)} tasks x {len(conditions)} conditions ---")

    for i, task in enumerate(REASONING_TASKS):
        prompt = task["prompt"]
        correct_answer = task["answer"]
        category = task["category"]

        print(f"\n  Task {i+1}/{len(REASONING_TASKS)}: [{category}] {prompt[:50]}...")

        for condition in conditions:
            strategy = {
                "baseline": "none",
                "monitored": "none",
                "cot_intervention": "cot_continuation",
                "step_by_step": "step_by_step",
            }[condition]

            result = monitor.generate_with_monitoring(
                prompt,
                max_new_tokens=max_new_tokens,
                intervention_strategy=strategy,
                max_interventions=3,
                do_sample=False,
            )

            is_correct = check_answer(result["text"], correct_answer)

            results[condition].append({
                "task_id": i,
                "prompt": prompt,
                "category": category,
                "correct_answer": correct_answer,
                "generated_text": result["text"][:200],
                "is_correct": is_correct,
                "mean_alpha": result["mean_alpha"],
                "n_interventions": result["n_interventions"],
                "breakdown_detected": result["breakdown_detected"],
                "n_tokens": result["n_tokens_generated"],
                "alpha_trajectory": [
                    {"step": s["step"], "alpha": s["alpha"]}
                    for s in result["alpha_trajectory"]
                ],
            })

            status = "✓" if is_correct else "✗"
            intv = f" [{result['n_interventions']} interventions]" if result["n_interventions"] > 0 else ""
            print(f"    {condition:<20} {status} alpha={result['mean_alpha']:.4f}{intv}")

    # Step 3: Analyze results
    print("\n--- Step 3: Analysis ---")
    analysis = analyze_results(results, conditions)

    return {
        "model": model_name,
        "threshold": threshold,
        "calibration": calibration,
        "results": results,
        "analysis": analysis,
    }


def analyze_results(results: Dict, conditions: List[str]) -> Dict:
    """Analyze the experiment results."""

    analysis = {}

    # Accuracy per condition
    print("\n  Accuracy per condition:")
    analysis["accuracy"] = {}
    for cond in conditions:
        correct = sum(1 for r in results[cond] if r["is_correct"])
        total = len(results[cond])
        acc = correct / total if total > 0 else 0
        analysis["accuracy"][cond] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
        }
        print(f"    {cond:<20} {correct}/{total} = {acc:.4f}")

    # Accuracy by category
    print("\n  Accuracy by category:")
    analysis["accuracy_by_category"] = {}
    for cond in conditions:
        analysis["accuracy_by_category"][cond] = {}
        for category in ["math", "logic"]:
            cat_results = [r for r in results[cond] if r["category"] == category]
            correct = sum(1 for r in cat_results if r["is_correct"])
            total = len(cat_results)
            acc = correct / total if total > 0 else 0
            analysis["accuracy_by_category"][cond][category] = {
                "correct": correct, "total": total, "accuracy": acc,
            }
            print(f"    {cond:<20} {category:<6} {correct}/{total} = {acc:.4f}")

    # Mean alpha per condition
    print("\n  Mean alpha per condition:")
    analysis["mean_alpha"] = {}
    for cond in conditions:
        alphas = [r["mean_alpha"] for r in results[cond]]
        mean_a = float(np.mean(alphas))
        analysis["mean_alpha"][cond] = mean_a
        print(f"    {cond:<20} {mean_a:.4f}")

    # Intervention statistics
    print("\n  Intervention statistics:")
    analysis["interventions"] = {}
    for cond in conditions:
        n_intv = [r["n_interventions"] for r in results[cond]]
        n_detected = sum(1 for r in results[cond] if r["breakdown_detected"])
        total_intv = sum(n_intv)
        analysis["interventions"][cond] = {
            "total_interventions": total_intv,
            "tasks_with_breakdown": n_detected,
            "mean_interventions_per_task": float(np.mean(n_intv)),
        }
        print(f"    {cond:<20} breakdowns={n_detected} total_intv={total_intv}")

    # Statistical tests: baseline vs intervention
    print("\n  Statistical tests (baseline vs interventions):")
    analysis["statistical_tests"] = {}

    baseline_correct = [r["is_correct"] for r in results["baseline"]]

    for cond in ["cot_intervention", "step_by_step"]:
        cond_correct = [r["is_correct"] for r in results[cond]]

        # McNemar's test for paired binary outcomes
        # Count discordant pairs
        b_correct_c_wrong = sum(1 for b, c in zip(baseline_correct, cond_correct)
                                if b and not c)
        b_wrong_c_correct = sum(1 for b, c in zip(baseline_correct, cond_correct)
                                if not b and c)

        # McNemar's test
        if b_correct_c_wrong + b_wrong_c_correct > 0:
            mcnemar_stat = (abs(b_correct_c_wrong - b_wrong_c_correct) - 1) ** 2 / \
                           (b_correct_c_wrong + b_wrong_c_correct)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0

        delta = sum(cond_correct) / len(cond_correct) - sum(baseline_correct) / len(baseline_correct)

        analysis["statistical_tests"][f"baseline_vs_{cond}"] = {
            "baseline_accuracy": sum(baseline_correct) / len(baseline_correct),
            "cond_accuracy": sum(cond_correct) / len(cond_correct),
            "delta": delta,
            "mcnemar_statistic": float(mcnemar_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "b_correct_c_wrong": b_correct_c_wrong,
            "b_wrong_c_correct": b_wrong_c_correct,
        }

        sig = "*" if p_value < 0.05 else "ns"
        print(f"    baseline vs {cond:<15} delta={delta:+.4f} "
              f"McNemar p={p_value:.4f} {sig}")
        print(f"      baseline correct → {cond} wrong: {b_correct_c_wrong}")
        print(f"      baseline wrong → {cond} correct: {b_wrong_c_correct}")

    # Alpha and correctness correlation
    print("\n  Alpha vs correctness correlation:")
    analysis["alpha_correctness"] = {}
    all_results = results["monitored"]  # Use monitored (no intervention) for this
    correct_alphas = [r["mean_alpha"] for r in all_results if r["is_correct"]]
    wrong_alphas = [r["mean_alpha"] for r in all_results if not r["is_correct"]]

    if len(correct_alphas) > 1 and len(wrong_alphas) > 1:
        t_stat, p_value = stats.ttest_ind(correct_alphas, wrong_alphas)
        delta = np.mean(correct_alphas) - np.mean(wrong_alphas)
        analysis["alpha_correctness"] = {
            "correct_mean_alpha": float(np.mean(correct_alphas)),
            "wrong_mean_alpha": float(np.mean(wrong_alphas)),
            "delta": float(delta),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }
        sig = "*" if p_value < 0.05 else "ns"
        print(f"    Correct: {np.mean(correct_alphas):.4f}, Wrong: {np.mean(wrong_alphas):.4f} "
              f"delta={delta:+.4f} p={p_value:.4f} {sig}")
    else:
        print("    Not enough data for correlation test")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Spectral Alpha Monitoring for LLM Reasoning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp_lm_monitor.json")
    args = parser.parse_args()

    print("=" * 70)
    print("LM MONITOR: Spectral Alpha Monitoring for Reasoning")
    print("Tests whether alpha monitoring + intervention improves accuracy")
    print("=" * 70)

    results = run_experiment(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Honest conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (HONEST)")
    print("=" * 70)

    analysis = results["analysis"]
    baseline_acc = analysis["accuracy"]["baseline"]["accuracy"]
    cot_acc = analysis["accuracy"]["cot_intervention"]["accuracy"]
    sbs_acc = analysis["accuracy"]["step_by_step"]["accuracy"]

    print(f"\nAccuracy:")
    print(f"  Baseline (no intervention):     {baseline_acc:.4f}")
    print(f"  CoT continuation intervention:  {cot_acc:.4f}")
    print(f"  Step-by-step intervention:      {sbs_acc:.4f}")

    cot_test = analysis["statistical_tests"].get("baseline_vs_cot_intervention", {})
    sbs_test = analysis["statistical_tests"].get("baseline_vs_step_by_step", {})

    print(f"\nStatistical significance:")
    print(f"  CoT vs baseline:  delta={cot_test.get('delta', 0):+.4f} "
          f"p={cot_test.get('p_value', 1):.4f}")
    print(f"  SBS vs baseline:  delta={sbs_test.get('delta', 0):+.4f} "
          f"p={sbs_test.get('p_value', 1):.4f}")

    alpha_corr = analysis.get("alpha_correctness", {})
    print(f"\nAlpha vs correctness:")
    print(f"  Correct mean alpha: {alpha_corr.get('correct_mean_alpha', 0):.4f}")
    print(f"  Wrong mean alpha:   {alpha_corr.get('wrong_mean_alpha', 0):.4f}")
    print(f"  p-value:            {alpha_corr.get('p_value', 1):.4f}")

    print(f"\nVerdict:")
    if cot_test.get("significant", False) and cot_test.get("delta", 0) > 0:
        print("  POSITIVE: CoT intervention significantly improves reasoning accuracy.")
    elif sbs_test.get("significant", False) and sbs_test.get("delta", 0) > 0:
        print("  POSITIVE: Step-by-step intervention significantly improves accuracy.")
    elif cot_acc > baseline_acc + 0.05 or sbs_acc > baseline_acc + 0.05:
        print("  MIXED: Intervention shows improvement but not statistically significant.")
        print("  May need more tasks or larger model for significance.")
    elif alpha_corr.get("significant", False):
        print("  MIXED: Alpha predicts correctness but intervention doesn't help.")
        print("  Detection works but intervention strategy needs improvement.")
    else:
        print("  NEGATIVE: Alpha monitoring does not improve reasoning accuracy.")
        print("  The model may be too small, or the intervention strategy ineffective.")


if __name__ == "__main__":
    main()
