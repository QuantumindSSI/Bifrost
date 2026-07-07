"""
Experiment LM-REASONING: Phase Coherence in LLM Hidden States Predicts Reasoning

Tests whether phase coherence metrics applied to LLM hidden states can:
1. Distinguish reasoning from factual recall
2. Predict reasoning correctness BEFORE the answer is generated

This is based on the modality-dependence finding: phase coherence captures
structure in images and sensors. If reasoning has structure (logical steps),
then phase coherence in hidden states should be higher during reasoning.

Uses a small LLM (Qwen2.5-0.5B or Pythia-160M) to make this feasible on
limited hardware.

Usage:
    python3 research_dir/experiment_lm_reasoning_phase.py
    python3 research_dir/experiment_lm_reasoning_phase.py --model Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


# Reasoning and factual recall task sets
# Each task has a prompt and a known correct answer

REASONING_TASKS = [
    # Math reasoning
    {"prompt": "What is 7 + 3? Answer:", "answer": " 10", "type": "math", "correct": True},
    {"prompt": "What is 15 - 8? Answer:", "answer": " 7", "type": "math", "correct": True},
    {"prompt": "What is 4 * 6? Answer:", "answer": " 24", "type": "math", "correct": True},
    {"prompt": "What is 20 / 4? Answer:", "answer": " 5", "type": "math", "correct": True},
    {"prompt": "What is 9 + 6? Answer:", "answer": " 15", "type": "math", "correct": True},
    {"prompt": "What is 12 - 5? Answer:", "answer": " 7", "type": "math", "correct": True},
    {"prompt": "What is 3 * 8? Answer:", "answer": " 24", "type": "math", "correct": True},
    {"prompt": "What is 18 / 3? Answer:", "answer": " 6", "type": "math", "correct": True},
    {"prompt": "What is 11 + 9? Answer:", "answer": " 20", "type": "math", "correct": True},
    {"prompt": "What is 14 - 6? Answer:", "answer": " 8", "type": "math", "correct": True},
    # Logic reasoning
    {"prompt": "If all cats are animals, and Whiskers is a cat, then Whiskers is a: Answer:", "answer": " animal", "type": "logic", "correct": True},
    {"prompt": "If A > B and B > C, then A > C. True or false? Answer:", "answer": " True", "type": "logic", "correct": True},
    {"prompt": "If today is Monday, what day is it in 3 days? Answer:", "answer": " Thursday", "type": "logic", "correct": True},
    {"prompt": "If John is taller than Mary, and Mary is taller than Sue, who is shortest? Answer:", "answer": " Sue", "type": "logic", "correct": True},
    {"prompt": "How many sides does a triangle have? Answer:", "answer": " 3", "type": "logic", "correct": True},
    # Incorrect reasoning (wrong answers)
    {"prompt": "What is 7 + 3? Answer:", "answer": " 11", "type": "math", "correct": False},
    {"prompt": "What is 15 - 8? Answer:", "answer": " 6", "type": "math", "correct": False},
    {"prompt": "What is 4 * 6? Answer:", "answer": " 22", "type": "math", "correct": False},
    {"prompt": "What is 20 / 4? Answer:", "answer": " 6", "type": "math", "correct": False},
    {"prompt": "What is 9 + 6? Answer:", "answer": " 14", "type": "math", "correct": False},
    {"prompt": "If all cats are animals, and Whiskers is a cat, then Whiskers is a: Answer:", "answer": " plant", "type": "logic", "correct": False},
    {"prompt": "If A > B and B > C, then A > C. True or false? Answer:", "answer": " False", "type": "logic", "correct": False},
    {"prompt": "If today is Monday, what day is it in 3 days? Answer:", "answer": " Friday", "type": "logic", "correct": False},
]

FACTUAL_TASKS = [
    {"prompt": "The capital of France is:", "answer": " Paris", "type": "factual", "correct": True},
    {"prompt": "The largest planet in our solar system is:", "answer": " Jupiter", "type": "factual", "correct": True},
    {"prompt": "Water boils at what temperature in Celsius? Answer:", "answer": " 100", "type": "factual", "correct": True},
    {"prompt": "The chemical symbol for gold is:", "answer": " Au", "type": "factual", "correct": True},
    {"prompt": "How many continents are there? Answer:", "answer": " 7", "type": "factual", "correct": True},
    {"prompt": "The capital of Japan is:", "answer": " Tokyo", "type": "factual", "correct": True},
    {"prompt": "The speed of light is approximately how many km/s? Answer:", "answer": " 300000", "type": "factual", "correct": True},
    {"prompt": "The largest ocean is:", "answer": " Pacific", "type": "factual", "correct": True},
    {"prompt": "How many planets are in our solar system? Answer:", "answer": " 8", "type": "factual", "correct": True},
    {"prompt": "The chemical symbol for water is:", "answer": " H2O", "type": "factual", "correct": True},
    # Incorrect factual
    {"prompt": "The capital of France is:", "answer": " London", "type": "factual", "correct": False},
    {"prompt": "The largest planet in our solar system is:", "answer": " Mars", "type": "factual", "correct": False},
    {"prompt": "Water boils at what temperature in Celsius? Answer:", "answer": " 50", "type": "factual", "correct": False},
    {"prompt": "The chemical symbol for gold is:", "answer": " Ag", "type": "factual", "correct": False},
    {"prompt": "How many continents are there? Answer:", "answer": " 5", "type": "factual", "correct": False},
]


def compute_phase_coherence_metrics(hidden_states: torch.Tensor) -> Dict[str, float]:
    """Compute phase coherence metrics on LLM hidden states.

    Treats the hidden dimension as a "signal" and computes spectral properties.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape (n_layers, n_tokens, hidden_dim) — hidden states from LLM

    Returns
    -------
    dict of metrics averaged across layers and tokens
    """
    n_layers, n_tokens, hidden_dim = hidden_states.shape

    all_plv = []
    all_phase_entropy = []
    all_phase_stability = []
    all_spectral_alpha = []

    for layer in range(n_layers):
        layer_states = hidden_states[layer]  # (n_tokens, hidden_dim)

        # Treat each token's hidden state as a signal
        # FFT across the hidden dimension
        fft = torch.fft.rfft(layer_states, dim=-1)  # (n_tokens, hidden_dim//2+1)
        amplitude = fft.abs()
        phase = fft.angle()

        # 1. Phase Locking Value (PLV) across tokens at each frequency
        # How coherent are the phases across tokens?
        exp_phase = torch.exp(1j * phase)  # (n_tokens, n_freq)
        plv_per_freq = torch.abs(exp_phase.mean(dim=0)).real  # (n_freq,)
        mean_plv = float(plv_per_freq.mean())
        all_plv.append(mean_plv)

        # 2. Phase entropy (how spread out are the phases?)
        # Average across tokens, then compute entropy across frequencies
        # Phase entropy via numpy histogram (torch.histc API differs across versions)
        phase_np = phase.flatten().numpy()
        phase_hist, _ = np.histogram(phase_np, bins=32, range=(-np.pi, np.pi))
        phase_hist = phase_hist / (phase_hist.sum() + 1e-10)
        phase_entropy = float(-np.sum(phase_hist * np.log(phase_hist + 1e-10)))
        all_phase_entropy.append(phase_entropy)

        # 3. Phase stability (how stable is phase across tokens?)
        # Standard deviation of phase across tokens, averaged over frequencies
        phase_std = phase.std(dim=0).mean()  # scalar
        phase_stability = float(1.0 / (1.0 + phase_std))
        all_phase_stability.append(phase_stability)

        # 4. Spectral alpha (power law decay of amplitude spectrum)
        # This is the metric from "Spectral Geometry of Thought"
        amp_spectrum = amplitude.mean(dim=0)  # (n_freq,)
        freqs = torch.arange(1, len(amp_spectrum) + 1, dtype=torch.float)
        log_amp = torch.log(amp_spectrum + 1e-10)
        log_freq = torch.log(freqs)

        # Linear regression: log(amp) = -alpha * log(freq) + c
        if len(log_amp) > 2:
            X = log_freq.unsqueeze(1)
            y = log_amp.unsqueeze(1)
            try:
                alpha = torch.linalg.lstsq(X, y).solution[0, 0]
                all_spectral_alpha.append(float(-alpha))
            except Exception:
                all_spectral_alpha.append(0.0)
        else:
            all_spectral_alpha.append(0.0)

    return {
        "mean_plv": float(np.mean(all_plv)),
        "mean_phase_entropy": float(np.mean(all_phase_entropy)),
        "mean_phase_stability": float(np.mean(all_phase_stability)),
        "mean_spectral_alpha": float(np.mean(all_spectral_alpha)),
        "plv_per_layer": all_plv,
        "spectral_alpha_per_layer": all_spectral_alpha,
    }


def compute_layerwise_metrics(hidden_states: torch.Tensor) -> List[Dict[str, float]]:
    """Compute phase coherence metrics per layer."""
    n_layers, n_tokens, hidden_dim = hidden_states.shape
    metrics_per_layer = []

    for layer in range(n_layers):
        layer_states = hidden_states[layer]

        fft = torch.fft.rfft(layer_states, dim=-1)
        amplitude = fft.abs()
        phase = fft.angle()

        exp_phase = torch.exp(1j * phase)
        plv = float(torch.abs(exp_phase.mean(dim=0)).real.mean())

        phase_hist_np, _ = np.histogram(phase.flatten().numpy(), bins=32, range=(-np.pi, np.pi))
        phase_hist_np = phase_hist_np / (phase_hist_np.sum() + 1e-10)
        phase_entropy = float(-np.sum(phase_hist_np * np.log(phase_hist_np + 1e-10)))

        phase_std = phase.std(dim=0).mean()
        phase_stability = float(1.0 / (1.0 + phase_std))

        amp_spectrum = amplitude.mean(dim=0)
        freqs = torch.arange(1, len(amp_spectrum) + 1, dtype=torch.float)
        log_amp = torch.log(amp_spectrum + 1e-10)
        log_freq = torch.log(freqs)

        alpha = 0.0
        if len(log_amp) > 2:
            X = log_freq.unsqueeze(1)
            y = log_amp.unsqueeze(1)
            try:
                alpha = float(-torch.linalg.lstsq(X, y).solution[0, 0])
            except Exception:
                pass

        metrics_per_layer.append({
            "layer": layer,
            "plv": plv,
            "phase_entropy": phase_entropy,
            "phase_stability": phase_stability,
            "spectral_alpha": alpha,
        })

    return metrics_per_layer


def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    max_length: int = 128,
) -> torch.Tensor:
    """Extract hidden states from all layers for a given prompt.

    Returns
    -------
    hidden_states : torch.Tensor (n_layers, n_tokens, hidden_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (n_layers+1) tensors
    # Each is (1, n_tokens, hidden_dim)
    hidden_states = torch.stack([hs.squeeze(0) for hs in outputs.hidden_states])
    # Shape: (n_layers+1, n_tokens, hidden_dim)

    return hidden_states


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
) -> Dict:
    """Run the LM reasoning phase coherence experiment."""

    print(f"\nLoading model: {model_name}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    print(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")

    all_tasks = REASONING_TASKS + FACTUAL_TASKS

    results = []
    print(f"\nProcessing {len(all_tasks)} tasks...")

    for i, task in enumerate(all_tasks):
        prompt = task["prompt"]
        task_type = task["type"]
        correct = task["correct"]

        try:
            hidden_states = extract_hidden_states(
                model, tokenizer, prompt, device=device)
            metrics = compute_phase_coherence_metrics(hidden_states)
            layer_metrics = compute_layerwise_metrics(hidden_states)

            result = {
                "task_id": i,
                "prompt": prompt,
                "type": task_type,
                "correct": correct,
                "n_tokens": hidden_states.shape[1],
                "metrics": metrics,
                "layer_metrics": layer_metrics,
            }
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(all_tasks)} tasks")

        except Exception as e:
            print(f"  Task {i}: ERROR - {e}")
            results.append({
                "task_id": i,
                "prompt": prompt,
                "type": task_type,
                "correct": correct,
                "error": str(e),
            })

    # Analysis
    analysis = analyze_results(results)
    return {"model": model_name, "results": results, "analysis": analysis}


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze whether phase metrics predict correctness and task type."""

    # Separate by task type and correctness
    reasoning_correct = [r for r in results if r.get("type") == "math" or r.get("type") == "logic"]
    reasoning_correct = [r for r in reasoning_correct if "metrics" in r]
    factual = [r for r in results if r.get("type") == "factual" and "metrics" in r]

    reasoning_true = [r for r in reasoning_correct if r["correct"]]
    reasoning_false = [r for r in reasoning_correct if not r["correct"]]
    factual_true = [r for r in factual if r["correct"]]
    factual_false = [r for r in factual if not r["correct"]]

    def get_metric_values(items, metric_name):
        return [r["metrics"][metric_name] for r in items]

    metrics_to_test = ["mean_plv", "mean_phase_entropy", "mean_phase_stability", "mean_spectral_alpha"]

    analysis = {}

    # Test 1: Reasoning correct vs incorrect
    print("\n--- Test 1: Reasoning correct vs incorrect ---")
    analysis["reasoning_correct_vs_incorrect"] = {}
    for metric in metrics_to_test:
        correct_vals = get_metric_values(reasoning_true, metric)
        incorrect_vals = get_metric_values(reasoning_false, metric)
        if len(correct_vals) > 1 and len(incorrect_vals) > 1:
            t_stat, p_value = stats.ttest_ind(correct_vals, incorrect_vals)
            delta = np.mean(correct_vals) - np.mean(incorrect_vals)
            analysis["reasoning_correct_vs_incorrect"][metric] = {
                "correct_mean": float(np.mean(correct_vals)),
                "incorrect_mean": float(np.mean(incorrect_vals)),
                "delta": float(delta),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {metric:<25} correct={np.mean(correct_vals):.4f} incorrect={np.mean(incorrect_vals):.4f} "
                  f"delta={delta:+.4f} p={p_value:.4f} {sig}")

    # Test 2: Reasoning vs factual recall
    print("\n--- Test 2: Reasoning vs factual recall ---")
    analysis["reasoning_vs_factual"] = {}
    for metric in metrics_to_test:
        reasoning_vals = get_metric_values(reasoning_true, metric)
        factual_vals = get_metric_values(factual_true, metric)
        if len(reasoning_vals) > 1 and len(factual_vals) > 1:
            t_stat, p_value = stats.ttest_ind(reasoning_vals, factual_vals)
            delta = np.mean(reasoning_vals) - np.mean(factual_vals)
            analysis["reasoning_vs_factual"][metric] = {
                "reasoning_mean": float(np.mean(reasoning_vals)),
                "factual_mean": float(np.mean(factual_vals)),
                "delta": float(delta),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {metric:<25} reasoning={np.mean(reasoning_vals):.4f} factual={np.mean(factual_vals):.4f} "
                  f"delta={delta:+.4f} p={p_value:.4f} {sig}")

    # Test 3: Factual correct vs incorrect
    print("\n--- Test 3: Factual correct vs incorrect ---")
    analysis["factual_correct_vs_incorrect"] = {}
    for metric in metrics_to_test:
        correct_vals = get_metric_values(factual_true, metric)
        incorrect_vals = get_metric_values(factual_false, metric)
        if len(correct_vals) > 1 and len(incorrect_vals) > 1:
            t_stat, p_value = stats.ttest_ind(correct_vals, incorrect_vals)
            delta = np.mean(correct_vals) - np.mean(incorrect_vals)
            analysis["factual_correct_vs_incorrect"][metric] = {
                "correct_mean": float(np.mean(correct_vals)),
                "incorrect_mean": float(np.mean(incorrect_vals)),
                "delta": float(delta),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {metric:<25} correct={np.mean(correct_vals):.4f} incorrect={np.mean(incorrect_vals):.4f} "
                  f"delta={delta:+.4f} p={p_value:.4f} {sig}")

    # Test 4: Layer-wise analysis for spectral_alpha (following Spectral Geometry of Thought)
    print("\n--- Test 4: Layer-wise spectral alpha (reasoning vs factual) ---")
    if reasoning_true and factual_true:
        n_layers = len(reasoning_true[0]["layer_metrics"])
        analysis["layerwise_spectral_alpha"] = {}
        for layer in range(n_layers):
            r_alphas = [r["layer_metrics"][layer]["spectral_alpha"] for r in reasoning_true]
            f_alphas = [r["layer_metrics"][layer]["spectral_alpha"] for r in factual_true]
            if len(r_alphas) > 1 and len(f_alphas) > 1:
                t_stat, p_value = stats.ttest_ind(r_alphas, f_alphas)
                delta = np.mean(r_alphas) - np.mean(f_alphas)
                analysis["layerwise_spectral_alpha"][f"layer_{layer}"] = {
                    "reasoning_mean": float(np.mean(r_alphas)),
                    "factual_mean": float(np.mean(f_alphas)),
                    "delta": float(delta),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                }
                sig = "*" if p_value < 0.05 else " "
                print(f"  Layer {layer:2d}: reasoning={np.mean(r_alphas):.4f} "
                      f"factual={np.mean(f_alphas):.4f} delta={delta:+.4f} p={p_value:.4f} {sig}")

    # Test 5: Can phase metrics predict correctness? (AUC)
    print("\n--- Test 5: Correctness prediction (reasoning tasks) ---")
    from sklearn.metrics import roc_auc_score
    analysis["correctness_prediction"] = {}
    all_reasoning = reasoning_true + reasoning_false
    labels = [1 if r["correct"] else 0 for r in all_reasoning]
    for metric in metrics_to_test:
        values = [r["metrics"][metric] for r in all_reasoning]
        try:
            auc = roc_auc_score(labels, values)
            # If AUC < 0.5, the metric is inversely predictive
            analysis["correctness_prediction"][metric] = {
                "auc": float(auc),
                "inverted": auc < 0.5,
                "effective_auc": float(max(auc, 1 - auc)),
            }
            eff_auc = max(auc, 1 - auc)
            print(f"  {metric:<25} AUC={auc:.4f} effective_AUC={eff_auc:.4f}")
        except Exception:
            analysis["correctness_prediction"][metric] = {"auc": 0.5, "error": True}

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="LM Reasoning Phase Coherence Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str,
                        default="research_dir/results/exp_lm_reasoning_phase.json")
    args = parser.parse_args()

    print("=" * 70)
    print("LM REASONING: Phase Coherence in LLM Hidden States")
    print("Tests whether phase metrics predict reasoning correctness")
    print("=" * 70)

    results = run_experiment(model_name=args.model, device=args.device)

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

    # Check reasoning correct vs incorrect
    r_vs_i = analysis.get("reasoning_correct_vs_incorrect", {})
    sig_metrics = [m for m, t in r_vs_i.items() if t.get("significant", False)]

    # Check correctness prediction
    pred = analysis.get("correctness_prediction", {})
    best_auc = 0
    best_metric = ""
    for metric, scores in pred.items():
        eff = scores.get("effective_auc", 0.5)
        if eff > best_auc:
            best_auc = eff
            best_metric = metric

    # Check reasoning vs factual
    r_vs_f = analysis.get("reasoning_vs_factual", {})
    sig_type = [m for m, t in r_vs_f.items() if t.get("significant", False)]

    print(f"\n1. Reasoning correct vs incorrect:")
    print(f"   Significant metrics: {sig_metrics if sig_metrics else 'NONE'}")
    if sig_metrics:
        for m in sig_metrics:
            t = r_vs_i[m]
            print(f"   {m}: delta={t['delta']:+.4f}, p={t['p_value']:.4f}")

    print(f"\n2. Reasoning vs factual recall:")
    print(f"   Significant metrics: {sig_type if sig_type else 'NONE'}")
    if sig_type:
        for m in sig_type:
            t = r_vs_f[m]
            print(f"   {m}: delta={t['delta']:+.4f}, p={t['p_value']:.4f}")

    print(f"\n3. Correctness prediction (AUC):")
    print(f"   Best metric: {best_metric} (AUC={best_auc:.4f})")
    if best_auc > 0.7:
        print(f"   → Phase metrics PREDICT reasoning correctness (AUC > 0.7)")
    elif best_auc > 0.6:
        print(f"   → Phase metrics WEAKLY predict correctness (AUC > 0.6)")
    else:
        print(f"   → Phase metrics do NOT predict correctness (AUC ≤ 0.6)")

    # Overall verdict
    print(f"\nOverall verdict:")
    if best_auc > 0.7 and (sig_metrics or sig_type):
        print("  POSITIVE: Phase coherence in LLM hidden states captures reasoning structure.")
        print("  This supports extending the coherence approach to language models.")
    elif best_auc > 0.6 or sig_metrics or sig_type:
        print("  MIXED: Some evidence that phase metrics relate to reasoning, but weak.")
    else:
        print("  NEGATIVE: Phase metrics do not capture reasoning structure in this model.")
        print("  May need larger model, different metrics, or different task design.")


if __name__ == "__main__":
    main()
