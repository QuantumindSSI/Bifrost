"""
End-to-end test of Spectral Guardrail on Qwen2.5-0.5B.

Tests the guardrail on:
1. A context-supported statement (should be safe)
2. A context-contradicted statement (should be flagged)
3. A tool-use scenario (hallucinated vs real tool output)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, "/Users/playferanmi/quantumind/Bifrost")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bifrost.agent.spectral_guardrail import SpectralGuardrail, AgentMonitor

def main():
    print("=" * 70)
    print("Spectral Guardrail E2E Test on Qwen2.5-0.5B")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, output_hidden_states=True
    )

    print("Creating spectral guardrail...")
    guardrail = SpectralGuardrail(
        model, tokenizer,
        monitor_layers=[2, 3, 4, 5],  # Early layers
        hfer_threshold=0.25,
        device="cpu",
    )

    # Calibrate with a known-good example
    print("Calibrating baseline with known-good example...")
    guardrail.calibrate_baseline(
        "Paris is the capital of France.",
        "The capital of France is Paris."
    )
    print(f"  Baseline HFER: {guardrail.baseline_features.hfer:.4f}")
    print(f"  Adjusted HFER threshold: {guardrail.hfer_threshold:.4f}")
    print(f"  Adjusted smoothness threshold: {guardrail.smoothness_threshold:.4f}")

    # Test 1: Context-supported statement
    print("\n--- Test 1: Context-supported statement ---")
    context = "Paris is the capital of France. It is located in the north-central part of the country."
    statement = "The capital of France is Paris."
    result = guardrail.check_context(context, statement)
    print(f"  HFER: {result.hfer:.4f}")
    print(f"  Spectral Entropy: {result.spectral_entropy:.4f}")
    print(f"  Smoothness: {result.smoothness:.4f}")
    print(f"  Fiedler Value: {result.fiedler_value:.4f}")
    print(f"  Is Safe: {result.is_safe}")
    print(f"  Confidence: {result.confidence:.4f}")

    # Test 2: Context-contradicted statement
    print("\n--- Test 2: Context-contradicted statement ---")
    context = "Paris is the capital of France. It is located in the north-central part of the country."
    statement = "The capital of France is Tokyo, which is in Japan."
    result = guardrail.check_context(context, statement)
    print(f"  HFER: {result.hfer:.4f}")
    print(f"  Spectral Entropy: {result.spectral_entropy:.4f}")
    print(f"  Smoothness: {result.smoothness:.4f}")
    print(f"  Fiedler Value: {result.fiedler_value:.4f}")
    print(f"  Is Safe: {result.is_safe}")
    print(f"  Confidence: {result.confidence:.4f}")

    # Test 3: Unrelated statement
    print("\n--- Test 3: Unrelated statement ---")
    context = "The Python programming language was created by Guido van Rossum in 1991."
    statement = "The weather today is sunny and warm with a gentle breeze."
    result = guardrail.check_context(context, statement)
    print(f"  HFER: {result.hfer:.4f}")
    print(f"  Spectal Entropy: {result.spectral_entropy:.4f}")
    print(f"  Smoothness: {result.smoothness:.4f}")
    print(f"  Is Safe: {result.is_safe}")

    # Test 4: Agent monitor with multiple steps
    print("\n--- Test 4: Agent monitor with multiple steps ---")
    monitor = AgentMonitor(model, tokenizer, device="cpu")
    monitor.calibrate("Paris is the capital of France.", "The capital of France is Paris.")

    steps = [
        ("Search for: capital of France", "The capital of France is Paris."),
        ("Search for: capital of Japan", "The capital of Japan is Tokyo."),
        ("Search for: capital of France", "The capital of France is London."),  # Wrong
        ("Search for: 2+2", "2+2=4"),
        ("Search for: 2+2", "2+2=5"),  # Wrong
    ]

    for i, (ctx, out) in enumerate(steps):
        result = monitor.check_step(ctx, out, step_id=i)
        status = "SAFE" if result.is_safe else "UNSAFE"
        print(f"  Step {i+1}: [{status}] HFER={result.hfer:.4f} | {out[:40]}")

    summary = monitor.get_health_summary()
    print(f"\n  Health Summary:")
    print(f"    Status: {summary['status']}")
    print(f"    Safe steps: {summary['safe_steps']}/{summary['steps']}")
    print(f"    Avg HFER: {summary['avg_hfer']:.4f}")
    print(f"    HFER trend: {[f'{h:.3f}' for h in summary['hfer_trend']]}")

    # Test 5: Tool call verification
    print("\n--- Test 5: Tool call verification ---")
    tool_tests = [
        ("Calculator: adds two numbers", "2 + 3", "5"),  # Correct
        ("Calculator: adds two numbers", "2 + 3", "7"),  # Hallucinated
    ]

    for desc, inp, out in tool_tests:
        result = monitor.check_tool_call(desc, inp, out)
        status = "SAFE" if result.is_safe else "UNSAFE"
        print(f"  [{status}] {desc} | {inp} = {out} | HFER={result.hfer:.4f}")

    print("\n" + "=" * 70)
    print("E2E Test Complete")
    print("=" * 70)
    print("\nNote: On 0.5B model, the spectral signal may be weaker than")
    print("the 97.7% recall reported on Llama 3.1 8B. The guardrail is")
    print("designed for 7B+ models where the bimodal HFER pattern is stronger.")


if __name__ == "__main__":
    main()
