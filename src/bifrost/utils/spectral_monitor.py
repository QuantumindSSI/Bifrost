"""
SpectralAlphaMonitor: Real-time reasoning breakdown detection for LLMs.

Computes spectral alpha (power law decay of amplitude spectrum) of LLM hidden
states at each generation step. When alpha drops below a threshold (indicating
the model has stopped reasoning and switched to factual/recall mode), triggers
an intervention to restore reasoning.

Based on the finding (doc 19) that:
- Reasoning has higher spectral alpha (more distributed representations)
- Factual recall has lower spectral alpha (more compressed representations)
- The difference is significant (p=0.002) and consistent across layers

Usage:
    from bifrost.utils.spectral_monitor import SpectralAlphaMonitor

    monitor = SpectralAlphaMonitor(model, tokenizer, threshold=-0.88)
    result = monitor.generate_with_monitoring(prompt, max_new_tokens=100)
    print(f"Answer: {result['text']}")
    print(f"Alpha trajectory: {result['alpha_trajectory']}")
    print(f"Interventions: {result['interventions']}")
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_spectral_alpha(hidden_states: torch.Tensor) -> float:
    """Compute spectral alpha for a single hidden state vector.

    Spectral alpha is the power law exponent of the amplitude spectrum:
    amp(f) ~ f^(-alpha)

    Higher alpha = more low-frequency concentration = more compressed
    Lower alpha = more distributed across frequencies

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape (hidden_dim,) — a single token's hidden state

    Returns
    -------
    alpha : float
        The spectral alpha value (higher = more compressed)
    """
    # FFT across hidden dimension
    fft = torch.fft.rfft(hidden_states)
    amplitude = fft.abs()

    # Power law fit: log(amp) = -alpha * log(freq) + c
    freqs = torch.arange(1, len(amplitude) + 1, dtype=torch.float)
    log_amp = torch.log(amplitude + 1e-10)
    log_freq = torch.log(freqs)

    # Linear regression
    if len(log_amp) > 2:
        X = log_freq.unsqueeze(1)  # (N, 1)
        y = log_amp.unsqueeze(1)  # (N, 1)
        try:
            alpha = torch.linalg.lstsq(X, y).solution[0, 0]
            return float(-alpha)
        except Exception:
            return 0.0
    return 0.0


def compute_layer_alphas(hidden_states: torch.Tensor) -> List[float]:
    """Compute spectral alpha for each layer's hidden states.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape (n_layers, n_tokens, hidden_dim) or (n_tokens, hidden_dim)

    Returns
    -------
    alphas : List[float]
        One alpha value per layer (averaged across tokens)
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)

    n_layers = hidden_states.shape[0]
    alphas = []
    for layer in range(n_layers):
        layer_states = hidden_states[layer]  # (n_tokens, hidden_dim)
        # Average alpha across tokens
        layer_alphas = [compute_spectral_alpha(layer_states[t])
                        for t in range(layer_states.shape[0])]
        alphas.append(float(np.mean(layer_alphas)))
    return alphas


class SpectralAlphaMonitor:
    """Monitors spectral alpha during LLM generation and detects reasoning breakdown.

    The monitor hooks into the model's forward pass to capture hidden states
    at each generation step. It computes spectral alpha for the last generated
    token across all layers, maintaining a trajectory over time.

    When alpha drops below a threshold for a sustained period, it triggers
    an intervention (e.g., injecting chain-of-thought continuation tokens).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        threshold: float = -0.88,
        window_size: int = 5,
        intervention_layers: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        """Initialize the monitor.

        Parameters
        ----------
        model : nn.Module
            The LLM (HuggingFace AutoModelForCausalLM)
        tokenizer
            The tokenizer
        threshold : float
            Alpha below this triggers reasoning breakdown detection.
            Based on experiment: reasoning ~-0.84, factual ~-0.90.
            Default -0.88 is between these.
        window_size : int
            Number of consecutive tokens below threshold to trigger intervention.
            Default 5 (sustained drop, not single-token noise).
        intervention_layers : List[int], optional
            Which layers to monitor. Default: middle layers (skip first/last 2).
        device : str
            Device for computation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.window_size = window_size
        self.device = device

        # Determine which layers to monitor
        n_layers = model.config.num_hidden_layers + 1
        if intervention_layers is None:
            # Monitor middle layers (where reasoning/factual difference is strongest)
            self.monitored_layers = list(range(2, n_layers - 2))
        else:
            self.monitored_layers = intervention_layers

        # State
        self.alpha_trajectory: List[Dict] = []
        self.hidden_states_buffer: List[torch.Tensor] = []
        self.interventions: List[Dict] = []
        self.intervention_count = 0

        # Hook registration
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on transformer layers to capture hidden states."""
        # For Qwen/Llama-style models, hook the output of each decoder layer
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        else:
            # Fallback: no hooks, use output_hidden_states instead
            self._use_hooks = False
            return

        self._use_hooks = True
        self._layer_outputs = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is typically a tuple; first element is hidden states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Store the last token's hidden state
                self._layer_outputs[layer_idx] = hidden[:, -1, :].detach()
            return hook_fn

        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def _compute_current_alpha(self) -> Optional[float]:
        """Compute spectral alpha for the current generation step.

        Returns the average alpha across monitored layers for the last token.
        """
        if not self._use_hooks:
            return None

        alphas = []
        for layer_idx in self.monitored_layers:
            if layer_idx in self._layer_outputs:
                hidden = self._layer_outputs[layer_idx].squeeze(0)  # (hidden_dim,)
                alpha = compute_spectral_alpha(hidden)
                alphas.append(alpha)

        if not alphas:
            return None

        return float(np.mean(alphas))

    def _detect_breakdown(self) -> bool:
        """Check if reasoning breakdown is detected.

        Returns True if alpha has been below threshold for window_size consecutive tokens.
        """
        if len(self.alpha_trajectory) < self.window_size:
            return False

        recent = self.alpha_trajectory[-self.window_size:]
        below_threshold = all(step["alpha"] < self.threshold for step in recent)

        # Also check that alpha was previously above threshold (actual drop)
        if len(self.alpha_trajectory) > self.window_size:
            prior = self.alpha_trajectory[-self.window_size - 1]
            was_reasoning = prior["alpha"] > self.threshold
        else:
            was_reasoning = True

        return below_threshold and was_reasoning

    def generate_with_monitoring(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        intervention_strategy: str = "cot_continuation",
        max_interventions: int = 3,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Dict:
        """Generate text with spectral alpha monitoring and optional intervention.

        Parameters
        ----------
        prompt : str
            The input prompt.
        max_new_tokens : int
            Maximum tokens to generate.
        intervention_strategy : str
            "none" — no intervention (baseline)
            "cot_continuation" — inject " Therefore," when breakdown detected
            "step_by_step" — inject " Let me think step by step." when breakdown detected
            "reask" — inject " Let me reconsider." when breakdown detected
        max_interventions : int
            Maximum number of interventions per generation.
        do_sample : bool
            Whether to sample (True) or greedy decode (False).
        temperature : float
            Sampling temperature.

        Returns
        -------
        dict with:
            text: generated text
            alpha_trajectory: list of {token, alpha, layer_alphas} per step
            interventions: list of {position, strategy, alpha_before, alpha_after}
            mean_alpha: average alpha during generation
            breakdown_detected: whether breakdown was detected
        """
        self.alpha_trajectory = []
        self._layer_outputs = {}
        self.interventions = []
        self.intervention_count = 0

        # Tokenize prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        generated_ids = input_ids.clone()

        # Intervention tokens
        intervention_tokens = {
            "cot_continuation": " Therefore,",
            "step_by_step": " Let me think step by step.",
            "reask": " Let me reconsider.",
        }

        for step in range(max_new_tokens):
            # Forward pass to get next token
            with torch.no_grad():
                outputs = self.model(
                    generated_ids,
                    output_hidden_states=not self._use_hooks,
                )

            # Get logits for next token
            logits = outputs.logits[:, -1, :]

            # Get next token
            if do_sample:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Compute spectral alpha for this step
            if self._use_hooks:
                alpha = self._compute_current_alpha()
                layer_alphas = {}
                for layer_idx in self.monitored_layers:
                    if layer_idx in self._layer_outputs:
                        hidden = self._layer_outputs[layer_idx].squeeze(0)
                        layer_alphas[layer_idx] = compute_spectral_alpha(hidden)
            else:
                # Fallback: use output_hidden_states
                if outputs.hidden_states is not None:
                    hidden_states = torch.stack(
                        [hs[:, -1, :].squeeze(0) for hs in outputs.hidden_states]
                    )
                    alphas = compute_layer_alphas(hidden_states)
                    monitored = [alphas[i] for i in self.monitored_layers
                                 if i < len(alphas)]
                    alpha = float(np.mean(monitored)) if monitored else None
                    layer_alphas = {i: alphas[i] for i in self.monitored_layers
                                    if i < len(alphas)}
                else:
                    alpha = None
                    layer_alphas = {}

            # Record trajectory
            token_id = next_token.item()
            token_text = self.tokenizer.decode([token_id])
            self.alpha_trajectory.append({
                "step": step,
                "token": token_text,
                "token_id": token_id,
                "alpha": alpha if alpha is not None else 0.0,
                "layer_alphas": layer_alphas,
            })

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for reasoning breakdown and intervene
            if (intervention_strategy != "none" and
                    self.intervention_count < max_interventions and
                    self._detect_breakdown()):

                intervention_text = intervention_tokens.get(
                    intervention_strategy, " Therefore,")
                intervention_ids = self.tokenizer(
                    intervention_text, return_tensors="pt"
                ).input_ids.to(self.device)

                alpha_before = self.alpha_trajectory[-1]["alpha"]

                # Inject intervention tokens
                generated_ids = torch.cat([generated_ids, intervention_ids], dim=-1)

                # Record intervention
                self.interventions.append({
                    "position": step,
                    "strategy": intervention_strategy,
                    "text": intervention_text,
                    "alpha_before": alpha_before,
                })
                self.intervention_count += 1

                # Clear the trajectory window to avoid re-triggering
                # (the intervention changes the context)
                self.alpha_trajectory = []  # Reset for post-intervention tracking

        # Extract generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0, prompt_len:], skip_special_tokens=True
        )

        # Compute summary statistics
        all_alphas = [s["alpha"] for s in self.alpha_trajectory if s["alpha"] != 0.0]
        mean_alpha = float(np.mean(all_alphas)) if all_alphas else 0.0

        return {
            "text": generated_text,
            "alpha_trajectory": self.alpha_trajectory,
            "interventions": self.interventions,
            "mean_alpha": mean_alpha,
            "breakdown_detected": len(self.interventions) > 0,
            "n_interventions": len(self.interventions),
            "n_tokens_generated": len(self.alpha_trajectory),
        }


def calibrate_threshold(
    model: nn.Module,
    tokenizer,
    reasoning_prompts: List[str],
    factual_prompts: List[str],
    device: str = "cpu",
) -> Dict:
    """Calibrate the spectral alpha threshold for a given model.

    Computes alpha trajectories for reasoning and factual prompts,
    then suggests a threshold between the two distributions.

    Returns
    -------
    dict with reasoning_mean, factual_mean, suggested_threshold, separation
    """
    monitor = SpectralAlphaMonitor(model, tokenizer, threshold=-999, device=device)

    reasoning_alphas = []
    factual_alphas = []

    for prompt in reasoning_prompts:
        result = monitor.generate_with_monitoring(
            prompt, max_new_tokens=30, intervention_strategy="none")
        reasoning_alphas.append(result["mean_alpha"])

    for prompt in factual_prompts:
        result = monitor.generate_with_monitoring(
            prompt, max_new_tokens=30, intervention_strategy="none")
        factual_alphas.append(result["mean_alpha"])

    r_mean = float(np.mean(reasoning_alphas))
    f_mean = float(np.mean(factual_alphas))
    suggested = (r_mean + f_mean) / 2  # midpoint

    return {
        "reasoning_mean_alpha": r_mean,
        "factual_mean_alpha": f_mean,
        "suggested_threshold": suggested,
        "separation": r_mean - f_mean,
        "reasoning_alphas": reasoning_alphas,
        "factual_alphas": factual_alphas,
    }
