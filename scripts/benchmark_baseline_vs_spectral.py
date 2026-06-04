#!/usr/bin/env python3
"""
Benchmark: Baseline GPT-2 vs Bifrost Spectral-Enhanced GPT-2

Usage:
    python scripts/benchmark_baseline_vs_spectral.py \
        --llm-name gpt2 \
        --data-path train_data/text_corpus.txt \
        --adapter-checkpoint checkpoints/phasellm_lm_adapter_corrected.pt \
        --output benchmark_results.json
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from bifrost.llm_adapter import BifrostEnhancedLLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-name", type=str, default="gpt2")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--adapter-checkpoint", type=str, required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--output", type=str, default="benchmark_results.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_texts(path: str, n: int) -> List[str]:
    texts, current = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                current.append(line)
            else:
                if current:
                    texts.append("\n".join(current))
                    current = []
                if len(texts) >= n:
                    break
    return texts[:n]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        enc = tokenizer(texts, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}


def eval_ppl(model, loader, device):
    model.eval()
    total_loss, total_tokens, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits if hasattr(out, "logits") else out
            if logits.dim() != 3:
                continue
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction="none")
            loss = (loss * shift_mask.view(-1)).sum()
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()
            n += 1
    avg = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg), avg


def measure_latency(model, loader, device, warmup=10, runs=50):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= warmup:
                break
            _ = model(input_ids=batch["input_ids"].to(device))
    times = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= runs:
                break
            ids = batch["input_ids"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids=ids)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), min(times), max(times)


def generate_samples(model, tokenizer, device, n=5):
    model.eval()
    prompts = ["The scientific method is", "In the beginning",
               "The theory of", "Once upon a time", "The fundamental principle"]
    out = []
    with torch.no_grad():
        for p in prompts[:n]:
            ids = tokenizer.encode(p, return_tensors="pt").to(device)
            gen = model.generate(ids, max_length=50, do_sample=True,
                                 temperature=0.8, pad_token_id=tokenizer.eos_token_id)
            out.append({"prompt": p, "text": tokenizer.decode(gen[0], skip_special_tokens=True)})
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    tokenizer.pad_token = tokenizer.eos_token

    texts = load_texts(args.data_path, args.num_samples)
    print(f"Loaded {len(texts)} validation samples")
    ds = TextDataset(texts, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    results = {}

    # Baseline
    print("\n=== BASELINE GPT-2 ===")
    base = AutoModelForCausalLM.from_pretrained(args.llm_name).to(device)
    base.eval()
    ppl, loss = eval_ppl(base, loader, device)
    print(f"  PPL: {ppl:.4f} | Loss: {loss:.4f}")
    lat_avg, lat_min, lat_max = measure_latency(base, loader, device)
    print(f"  Latency: {lat_avg:.2f}ms (min={lat_min:.2f}, max={lat_max:.2f})")
    gens = generate_samples(base, tokenizer, device)
    results["baseline"] = {"perplexity": ppl, "loss": loss,
                          "latency_ms": {"avg": lat_avg, "min": lat_min, "max": lat_max},
                          "generations": gens}
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Spectral
    print("\n=== BIFROST SPECTRAL ===")
    spectral = BifrostEnhancedLLM(
        llm_name=args.llm_name,
        adapter_mode="intermediate",
        adapter_layer=6,
        spectral_dim=128,
    ).to(device)

    ckpt = torch.load(args.adapter_checkpoint, map_location=device)
    spectral.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    spectral.eval()

    ppl_s, loss_s = eval_ppl(spectral, loader, device)
    print(f"  PPL: {ppl_s:.4f} | Loss: {loss_s:.4f}")
    lat_s_avg, lat_s_min, lat_s_max = measure_latency(spectral, loader, device)
    print(f"  Latency: {lat_s_avg:.2f}ms (min={lat_s_min:.2f}, max={lat_s_max:.2f})")
    gens_s = generate_samples(spectral, tokenizer, device)
    results["spectral"] = {"perplexity": ppl_s, "loss": loss_s,
                           "latency_ms": {"avg": lat_s_avg, "min": lat_s_min, "max": lat_s_max},
                           "generations": gens_s}

    # Summary
    print("\n=== SUMMARY ===")
    ppl_delta = ((ppl_s - ppl) / ppl) * 100
    lat_delta = ((lat_s_avg - lat_avg) / lat_avg) * 100
    print(f"Perplexity: Baseline={ppl:.4f} | Spectral={ppl_s:.4f} | Delta={ppl_delta:+.2f}%")
    print(f"Latency:    Baseline={lat_avg:.2f}ms | Spectral={lat_s_avg:.2f}ms | Delta={lat_delta:+.2f}%")

    results["summary"] = {
        "perplexity_baseline": ppl,
        "perplexity_spectral": ppl_s,
        "perplexity_delta_percent": ppl_delta,
        "latency_baseline_ms": lat_avg,
        "latency_spectral_ms": lat_s_avg,
        "latency_delta_percent": lat_delta,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
