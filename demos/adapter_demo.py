#!/usr/bin/env python
"""
Interactive demo of SpectralAdapter.

Usage:
    python demos/adapter_demo.py --llm gpt2
"""

import sys
sys.path.insert(0, "src")

import torch
import argparse
from bifrost.llm_adapter import BifrostEnhancedLLM


def demo_basic_generation(model):
    """Demo: Basic text generation with coherence tracking."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Generation with Spectral Coherence")
    print("=" * 60)
    
    prompts = [
        "The future of artificial intelligence is",
        "In the realm of quantum physics,",
        "The most important scientific discovery",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        # Generate with tracking
        result = model.generate_with_spectral(
            prompt,
            max_length=30,
            temperature=0.8,
            track_coherence=True,
        )
        
        print(f"Generated: {result['text']}")
        print(f"  Tokens: {result['tokens_generated']}")
        print(f"  Avg coherence: {result.get('avg_coherence', 0.0):.4f}")
        print(f"  Coherence range: {result.get('min_coherence', 0.0):.4f} - {result.get('max_coherence', 0.0):.4f}")


def demo_comparison(model, baseline_model):
    """Demo: Compare spectral vs baseline generation."""
    print("\n" + "=" * 60)
    print("Demo 2: Spectral vs Baseline Comparison")
    print("=" * 60)
    
    prompt = "The key to understanding consciousness lies in"
    
    print(f"\nPrompt: {prompt}")
    
    # Baseline
    print("\n[Baseline GPT2]")
    baseline_inputs = baseline_model.tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        baseline_output = baseline_model.llm.generate(
            baseline_inputs.input_ids,
            max_length=50,
            do_sample=True,
            temperature=0.8,
        )
    baseline_text = baseline_model.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    print(f"Generated: {baseline_text}")
    
    # Spectral
    print("\n[Spectral-Enhanced]")
    spectral_result = model.generate_with_spectral(prompt, max_length=30, temperature=0.8)
    print(f"Generated: {spectral_result['text']}")
    print(f"Coherence: {spectral_result.get('avg_coherence', 0.0):.4f}")


def demo_uncertainty_analysis(model):
    """Demo: Show uncertainty quantification."""
    print("\n" + "=" * 60)
    print("Demo 3: Uncertainty Quantification")
    print("=" * 60)
    
    prompts = [
        "The capital of France is",  # Factual - should have low uncertainty
        "In the year 3045, humans will",  # Speculative - may have higher uncertainty
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        inputs = model.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                return_spectral=True,
            )
            
            if outputs.uncertainty is not None:
                avg_unc = outputs.uncertainty.mean().item()
                max_unc = outputs.uncertainty.max().item()
                print(f"  Average uncertainty: {avg_unc:.4f}")
                print(f"  Max uncertainty: {max_unc:.4f}")
                print(f"  Interpretation: {'Low (factual)' if avg_unc < 0.5 else 'High (uncertain)'}")


def demo_parameter_efficiency(model):
    """Demo: Show how many parameters are trainable."""
    print("\n" + "=" * 60)
    print("Demo 4: Parameter Efficiency")
    print("=" * 60)
    
    params = model.get_trainable_params()
    
    print(f"\nTotal parameters: {params['total']:,}")
    print(f"Trainable (adapter): {params['trainable']:,}")
    print(f"Frozen (LLM): {params['frozen']:,}")
    print(f"Trainable percentage: {params['trainable_pct']:.4f}%")
    print(f"\nThis means we're only training {params['trainable_pct']:.2f}% of parameters!")


def main():
    parser = argparse.ArgumentParser(description="SpectralAdapter Demo")
    parser.add_argument("--llm", type=str, default="gpt2", help="HuggingFace model")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to trained adapter")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip baseline comparison")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpectralAdapter Interactive Demo")
    print("=" * 60)
    print(f"Loading model: {args.llm}")
    
    # Load spectral-enhanced model
    model = BifrostEnhancedLLM(
        llm_name=args.llm,
        adapter_mode="intermediate",
        adapter_layer=6,
        freeze_llm=True,
    )
    
    if args.adapter_path:
        print(f"Loading trained adapter: {args.adapter_path}")
        model.load_adapter(args.adapter_path)
    
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Run demos
    try:
        demo_parameter_efficiency(model)
        demo_basic_generation(model)
        
        if not args.skip_comparison:
            # Load baseline for comparison
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                baseline_model = type('obj', (object,), {
                    'llm': AutoModelForCausalLM.from_pretrained(args.llm),
                    'tokenizer': AutoTokenizer.from_pretrained(args.llm)
                })()
                demo_comparison(model, baseline_model)
            except Exception as e:
                print(f"\nSkipping comparison demo: {e}")
        
        demo_uncertainty_analysis(model)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
