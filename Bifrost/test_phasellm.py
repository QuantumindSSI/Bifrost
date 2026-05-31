#!/usr/bin/env python3
"""Simple test script for PhaseLLM (BifrostEnhancedLLM)"""

from bifrost.llm_adapter import BifrostEnhancedLLM

print("Loading model...")
# Use GPT-2 for faster testing
model = BifrostEnhancedLLM(
    llm_name="gpt2",
    adapter_mode="intermediate",
    adapter_layer=6,
)

print(f"Model loaded successfully")
print(f"Trainable params: {model.get_trainable_params()}")

print("\nGenerating text with spectral tracking...")
result = model.generate_with_spectral("Hello world", max_length=20)

print(f"\nGenerated text: {result['text']}")
print(f"Phase coherence: {result['avg_coherence']:.4f}")
print(f"Tokens generated: {result['tokens_generated']}")
