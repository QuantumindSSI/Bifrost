"""
SpectralAdapter: Integrate Bifrost phase coherence with any HuggingFace LLM.

This module provides a wrapper that injects spectral processing into existing
language models at intermediate layers, enabling phase-aware representations
without retraining the entire model from scratch.

Usage:
    from bifrost.llm_adapter import BifrostEnhancedLLM
    
    model = BifrostEnhancedLLM(
        llm_name="meta-llama/Llama-2-7b-hf",
        adapter_mode="intermediate",
        adapter_layer=16,
    )
    
    result = model.generate_with_spectral("Hello world", max_length=50)
    print(result["text"])
    print(f"Phase coherence: {result['avg_coherence']:.4f}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass

from bifrost import BifrostPipeline, SpectralTensor


@dataclass
class SpectralAdapterOutput:
    """Output container for spectral-enhanced generation."""
    logits: torch.Tensor
    hidden_states: torch.Tensor
    spectral: Optional[SpectralTensor] = None
    coherence_score: Optional[float] = None
    uncertainty: Optional[torch.Tensor] = None


class SpectralProjector(nn.Module):
    """
    Bidirectional projection between LLM hidden space and spectral space.
    
    Converts (B, T, d_model) ↔ (B, T, spectral_dim * 4) for SpectralTensor
    components: amplitude, phase, scale, uncertainty.
    """
    
    def __init__(self, d_model: int, spectral_dim: int):
        super().__init__()
        self.d_model = d_model
        self.spectral_dim = spectral_dim
        
        # Project to 4 * spectral_dim (amp, phase, scale, unc)
        self.to_spectral = nn.Linear(d_model, spectral_dim * 4)
        
        # Project back to d_model
        self.to_hidden = nn.Linear(spectral_dim * 4, d_model)
        
        # === UNCERTAINTY CALIBRATION ===
        # Learnable temperature for scaling (initially 1.0, learned during training)
        self.uncertainty_temperature = nn.Parameter(torch.tensor(1.0))
        # Learnable bias for shifting uncertainty baseline
        self.uncertainty_bias = nn.Parameter(torch.tensor(0.0))
        
        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.to_spectral.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_hidden.weight, gain=0.01)
        nn.init.zeros_(self.to_spectral.bias)
        nn.init.zeros_(self.to_hidden.bias)
    
    def forward(
        self, 
        hidden: torch.Tensor
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Project hidden states to spectral and back.
        
        Args:
            hidden: (B, T, d_model) LLM hidden states
            
        Returns:
            spectral: SpectralTensor with 4 components
            reconstructed: (B, T, d_model) projected back
        """
        B, T, _ = hidden.shape
        
        # Project to spectral space
        spectral_flat = self.to_spectral(hidden)  # (B, T, 4 * spectral_dim)
        spectral_flat = spectral_flat.view(B, T, 4, self.spectral_dim)
        
        # Split into SpectralTensor components
        amplitude = spectral_flat[:, :, 0, :].abs() + 1e-8  # Ensure positive
        phase = torch.tanh(spectral_flat[:, :, 1, :]) * 3.14159  # Bound to [-π, π]
        scale = spectral_flat[:, :, 2, :].abs() + 1e-8  # Ensure positive
        
        # Uncertainty with learned calibration (temperature scaling + bias)
        raw_uncertainty = spectral_flat[:, :, 3, :]
        uncertainty = (raw_uncertainty.abs() * F.softplus(self.uncertainty_temperature) 
                        + self.uncertainty_bias)
        uncertainty = torch.sigmoid(uncertainty)  # Scale to [0, 1] for probabilistic interpretation
        
        spectral = SpectralTensor(
            amplitude=amplitude,
            phase=phase,
            scale=scale,
            uncertainty=uncertainty,
        )
        
        # Project back to hidden space
        reconstructed_flat = torch.cat([
            spectral.amplitude,
            spectral.phase,
            spectral.scale,
            spectral.uncertainty,
        ], dim=-1)
        reconstructed = self.to_hidden(reconstructed_flat)
        
        return spectral, reconstructed
    
    def compute_uncertainty_calibration_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        n_bins: int = 10,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty calibration loss using Expected Calibration Error (ECE).
        
        Encourages predicted uncertainty to match actual prediction error.
        Higher uncertainty should correlate with higher error.
        
        Args:
            predictions: (B, T, d_model) model predictions
            targets: (B, T, d_model) ground truth targets
            uncertainties: (B, T, d_model) predicted uncertainties in [0, 1]
            n_bins: Number of bins for ECE computation
            
        Returns:
            calibration_loss: Scalar loss for backprop
            metrics: Dict with ECE, reliability diagram stats
        """
        # Compute absolute error
        errors = (predictions - targets).abs()  # (B, T, d_model)
        
        # Normalize errors to [0, 1] for comparison with uncertainty
        error_max = errors.max() + 1e-8
        errors_normalized = errors / error_max
        
        # Compute Expected Calibration Error (ECE)
        # Bin predictions by uncertainty, compute accuracy vs confidence
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=predictions.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_metrics = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Mask for samples in this bin
            in_bin = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
            bin_count = in_bin.sum().item()
            
            if bin_count > 0:
                # Average uncertainty (confidence) in bin
                avg_uncertainty = uncertainties[in_bin].mean().item()
                # Average error in bin
                avg_error = errors_normalized[in_bin].mean().item()
                # Weight by bin size
                bin_weight = bin_count / uncertainties.numel()
                # ECE contribution: |confidence - accuracy| * weight
                ece += abs(avg_uncertainty - avg_error) * bin_weight
                
                bin_metrics.append({
                    "bin_lower": bin_lower.item(),
                    "bin_upper": bin_upper.item(),
                    "count": bin_count,
                    "avg_uncertainty": avg_uncertainty,
                    "avg_error": avg_error,
                })
        
        # Calibration loss: minimize ECE + encourage proper uncertainty ordering
        # Loss 1: ECE (direct calibration)
        ece_loss = torch.tensor(ece, device=predictions.device)
        
        # Loss 2: Ranking loss - higher uncertainty should correlate with higher error
        # Use Spearman correlation approximation
        error_flat = errors_normalized.flatten()
        uncertainty_flat = uncertainties.flatten()
        
        # Compute correlation
        error_mean = error_flat.mean()
        uncertainty_mean = uncertainty_flat.mean()
        error_std = error_flat.std() + 1e-8
        uncertainty_std = uncertainty_flat.std() + 1e-8
        
        correlation = ((error_flat - error_mean) * (uncertainty_flat - uncertainty_mean)).mean()
        correlation = correlation / (error_std * uncertainty_std)
        
        # We want positive correlation (higher uncertainty = higher error)
        correlation_loss = -correlation  # Negative because we want to maximize correlation
        
        # Total loss
        total_loss = ece_loss + 0.1 * correlation_loss
        
        metrics = {
            "ece": ece,
            "correlation": correlation.item(),
            "avg_uncertainty": uncertainties.mean().item(),
            "avg_error": errors_normalized.mean().item(),
        }
        
        return total_loss, metrics
    
    def spectral_to_hidden(self, spectral: SpectralTensor) -> torch.Tensor:
        """Convert SpectralTensor back to hidden representation."""
        spectral_flat = torch.cat([
            spectral.amplitude,
            spectral.phase,
            spectral.scale,
            spectral.uncertainty,
        ], dim=-1)
        return self.to_hidden(spectral_flat)


class SpectralFusion(nn.Module):
    """
    Fuse spectral-enhanced representations with original hidden states.
    
    Uses cross-attention to allow model to selectively attend to spectral features.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism: how much to trust spectral vs original
        self.spectral_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        original: torch.Tensor,
        spectral: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse original and spectral representations.
        
        Args:
            original: (B, T_orig, d_model) from LLM
            spectral: (B, T_spectral, d_model) from Bifrost processing
            
        Returns:
            fused: (B, T_orig, d_model) combined representation
        """
        # Cross-attention: spectral queries, original keys/values
        # This allows spectral (potentially different length) to attend to original
        attn_out, _ = self.cross_attn(
            query=spectral,
            key=original,
            value=original,
        )
        
        # Project attn_out back to original length if needed
        if attn_out.shape[1] != original.shape[1]:
            # Interpolate to match original sequence length
            attn_out = torch.nn.functional.interpolate(
                attn_out.transpose(1, 2),  # (B, d_model, T_spectral)
                size=original.shape[1],
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)  # (B, T_orig, d_model)
        
        # Gating: use original for gate computation
        gate = self.spectral_gate(original)
        
        # Gated fusion
        fused = gate * attn_out + (1 - gate) * original
        
        # Residual + LayerNorm
        fused = self.layer_norm(original + self.dropout(fused))
        
        return fused


class BifrostEnhancedLLM(nn.Module):
    """
    Enhanced LLM with Bifrost spectral coherence integration.
    
    Injects phase-aware processing at intermediate layers while keeping
    base LLM frozen. Only adapter parameters are trainable.
    
    Args:
        llm_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
        adapter_mode: "intermediate" (recommended), "input", or "output"
        adapter_layer: Which layer to inject adapter (for intermediate mode)
        d_model: LLM hidden dimension (auto-detected if None)
        spectral_dim: Bifrost spectral dimension (default 128)
        freeze_llm: Whether to freeze base LLM weights (default True)
    """
    
    def __init__(
        self,
        llm_name: str = "gpt2",  # Default to small model for testing
        adapter_mode: str = "intermediate",
        adapter_layer: int = 6,
        d_model: Optional[int] = None,
        spectral_dim: int = 128,
        freeze_llm: bool = True,
    ):
        super().__init__()
        
        self.adapter_mode = adapter_mode
        self.adapter_layer = adapter_layer
        self.spectral_dim = spectral_dim
        
        # Lazy import to avoid hard dependency
        try:
            from transformers import AutoModel, AutoConfig, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required. Install: pip install transformers"
            )
        
        # Load base LLM
        self.config = AutoConfig.from_pretrained(llm_name)
        self.d_model = d_model or self.config.hidden_size
        
        self.llm = AutoModel.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        
        # Handle missing pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze LLM if requested
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
        
        # Initialize Bifrost pipeline
        self.bifrost = BifrostPipeline(
            d_model=spectral_dim,
            n_fft_canonical=min(1024, spectral_dim * 4),
            use_complex_ssm=True,
        )
        
        # Spectral projector (bidirectional)
        self.spectral_projector = SpectralProjector(self.d_model, spectral_dim)
        
        # Fusion layer for intermediate mode
        if adapter_mode == "intermediate":
            self.spectral_fusion = SpectralFusion(self.d_model, num_heads=8)
        
        # Track coherence for monitoring
        self.coherence_history: List[float] = []
    
    def apply_spectral_processing(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, SpectralTensor, float]:
        """
        Apply Bifrost spectral processing to hidden states.
        
        Args:
            hidden: (B, T, d_model) from LLM
            
        Returns:
            enhanced: (B, T, d_model) spectral-enhanced hidden states
            spectral: SpectralTensor representation
            coherence: Phase coherence score
        """
        B, T, _ = hidden.shape
        
        # Project to spectral space
        spectral, spectral_hidden = self.spectral_projector(hidden)
        
        # Apply Bifrost decomposer for phase coherence
        with torch.set_grad_enabled(self.training):
            if self.bifrost.use_complex_ssm:
                decomposed, _ = self.bifrost.decomposer(spectral, None)
            else:
                decomposed = self.bifrost.decomposer(spectral)
        
        # Project back to hidden space
        enhanced = self.spectral_projector.spectral_to_hidden(decomposed)
        
        # Compute phase coherence score
        phase_variance = decomposed.phase.var(dim=-1).mean().item()
        coherence_score = 1.0 / (1.0 + phase_variance)  # Higher = more coherent
        
        return enhanced, decomposed, coherence_score
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_spectral: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, SpectralAdapterOutput]:
        """
        Forward pass with spectral enhancement.
        
        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) attention mask
            return_spectral: Whether to return spectral tensors
            
        Returns:
            If return_spectral=False: logits (B, T, vocab_size)
            If return_spectral=True: SpectralAdapterOutput with spectral info
        """
        if self.adapter_mode == "intermediate":
            return self._forward_intermediate(
                input_ids, attention_mask, return_spectral, **kwargs
            )
        elif self.adapter_mode == "input":
            return self._forward_input(
                input_ids, attention_mask, return_spectral, **kwargs
            )
        else:  # output
            return self._forward_output(
                input_ids, attention_mask, return_spectral, **kwargs
            )
    
    def _forward_intermediate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        return_spectral: bool,
        **kwargs
    ) -> Union[torch.Tensor, SpectralAdapterOutput]:
        """Intermediate adapter: inject spectral at middle layer."""
        B, T = input_ids.shape
        
        # Get embeddings
        if hasattr(self.llm, 'embed_tokens'):
            # Llama-style
            embeddings = self.llm.embed_tokens(input_ids)
        elif hasattr(self.llm, 'wte'):
            # GPT2-style
            embeddings = self.llm.wte(input_ids)
        else:
            raise ValueError("Unsupported LLM architecture")
        
        hidden = embeddings
        
        # First half: standard LLM layers
        num_layers = len(self.llm.layers) if hasattr(self.llm, 'layers') else len(self.llm.h)
        layers = self.llm.layers if hasattr(self.llm, 'layers') else self.llm.h
        
        # Note: We don't pass attention_mask to individual layers because GPT2's
        # SDPA attention expects it in a specific format. The model handles
        # padding via pad_token_id set during tokenizer initialization.
        for i in range(min(self.adapter_layer, num_layers)):
            layer_outputs = layers[i](hidden)
            hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        # ↓↓↓ SPECTRAL ADAPTER INJECTION ↓↓↓
        enhanced, spectral, coherence = self.apply_spectral_processing(hidden)
        
        # Fuse with original
        hidden = self.spectral_fusion(hidden, enhanced)
        
        if self.training:
            self.coherence_history.append(coherence)
        # ↑↑↑ END SPECTRAL ADAPTER ↑↑↑
        
        # Second half: standard LLM layers
        for i in range(self.adapter_layer, num_layers):
            layer_outputs = layers[i](hidden)
            hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        # Final layer norm
        if hasattr(self.llm, 'norm'):
            hidden = self.llm.norm(hidden)
        elif hasattr(self.llm, 'ln_f'):
            hidden = self.llm.ln_f(hidden)
        
        # LM head
        if hasattr(self.llm, 'lm_head'):
            logits = self.llm.lm_head(hidden)
        elif hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden)
        else:
            # Create LM head if not exists
            vocab_size = self.config.vocab_size
            lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
            lm_head.weight = self.llm.wte.weight if hasattr(self.llm, 'wte') else None
            logits = lm_head(hidden)
        
        if return_spectral:
            return SpectralAdapterOutput(
                logits=logits,
                hidden_states=hidden,
                spectral=spectral,
                coherence_score=coherence,
                uncertainty=spectral.uncertainty.mean(dim=-1) if spectral else None,
            )

        return logits
    
    def _forward_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        return_spectral: bool,
        **kwargs
    ) -> Union[torch.Tensor, SpectralAdapterOutput]:
        """Input adapter: enhance embeddings before LLM."""
        # Get embeddings
        if hasattr(self.llm, 'embed_tokens'):
            embeddings = self.llm.embed_tokens(input_ids)
        elif hasattr(self.llm, 'wte'):
            embeddings = self.llm.wte(input_ids)
        else:
            raise ValueError("Unsupported LLM architecture")
        
        # Apply spectral processing to embeddings
        enhanced, spectral, coherence = self.apply_spectral_processing(embeddings)
        
        # Residual connection
        hidden = embeddings + enhanced
        
        # Pass through all LLM layers
        num_layers = len(self.llm.layers) if hasattr(self.llm, 'layers') else len(self.llm.h)
        layers = self.llm.layers if hasattr(self.llm, 'layers') else self.llm.h
        
        for i in range(num_layers):
            layer_outputs = layers[i](hidden, attention_mask=attention_mask)
            hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        # Final norm and LM head
        if hasattr(self.llm, 'norm'):
            hidden = self.llm.norm(hidden)
        elif hasattr(self.llm, 'ln_f'):
            hidden = self.llm.ln_f(hidden)
        
        if hasattr(self.llm, 'lm_head'):
            logits = self.llm.lm_head(hidden)
        else:
            vocab_size = self.config.vocab_size
            logits = nn.functional.linear(hidden, self.llm.wte.weight if hasattr(self.llm, 'wte') else torch.randn(vocab_size, self.d_model))
        
        if return_spectral:
            return SpectralAdapterOutput(
                logits=logits,
                spectral=spectral,
                coherence_score=coherence,
            )
        
        return logits
    
    def _forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        return_spectral: bool,
        **kwargs
    ) -> Union[torch.Tensor, SpectralAdapterOutput]:
        """Output adapter: enhance final representations."""
        # Standard LLM forward
        llm_output = self.llm(input_ids, attention_mask=attention_mask, **kwargs)
        hidden = llm_output.last_hidden_state if hasattr(llm_output, 'last_hidden_state') else llm_output[0]
        
        # Apply spectral processing
        enhanced, spectral, coherence = self.apply_spectral_processing(hidden)
        
        # Uncertainty-weighted fusion
        uncertainty = spectral.uncertainty.mean(dim=-1, keepdim=True)  # (B, T, 1)
        confidence = torch.sigmoid(-uncertainty)  # Low uncertainty = high confidence
        
        fused = confidence * enhanced + (1 - confidence) * hidden
        
        # LM head
        if hasattr(self.llm, 'lm_head'):
            logits = self.llm.lm_head(fused)
        else:
            vocab_size = self.config.vocab_size
            logits = nn.functional.linear(fused, self.llm.wte.weight if hasattr(self.llm, 'wte') else torch.randn(vocab_size, self.d_model))
        
        if return_spectral:
            return SpectralAdapterOutput(
                logits=logits,
                spectral=spectral,
                coherence_score=coherence,
                uncertainty=uncertainty,
            )
        
        return logits
    
    def generate_with_spectral(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        track_coherence: bool = True,
    ) -> Dict[str, Union[str, float, List[float]]]:
        """
        Generate text with spectral coherence tracking.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            track_coherence: Whether to track phase coherence scores
            
        Returns:
            Dict with generated text, coherence scores, and statistics
        """
        self.eval()
        
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        generated_ids = input_ids[0].tolist()
        coherence_scores = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward with spectral tracking
                outputs = self.forward(
                    input_ids,
                    return_spectral=track_coherence,
                )
                
                if track_coherence:
                    logits = outputs.logits
                    coherence_scores.append(outputs.coherence_score or 0.0)
                    if outputs.uncertainty is not None:
                        # Handle both 2D and 3D uncertainty tensors
                        if outputs.uncertainty.dim() == 3:
                            uncertainties.append(outputs.uncertainty[:, -1, :].mean().item())
                        else:
                            uncertainties.append(outputs.uncertainty.mean().item())
                else:
                    logits = outputs
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated_ids.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        result = {
            "text": generated_text,
            "prompt": prompt,
            "tokens_generated": len(generated_ids) - len(self.tokenizer(prompt).input_ids),
        }
        
        if track_coherence and coherence_scores:
            result["avg_coherence"] = sum(coherence_scores) / len(coherence_scores)
            result["max_coherence"] = max(coherence_scores)
            result["min_coherence"] = min(coherence_scores)
            result["coherence_scores"] = coherence_scores
        
        if uncertainties:
            result["avg_uncertainty"] = sum(uncertainties) / len(uncertainties)
        
        return result
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Return count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": trainable + frozen,
            "trainable_pct": 100 * trainable / (trainable + frozen),
        }
    
    def save_adapter(self, path: str):
        """Save only adapter weights (not base LLM)."""
        adapter_state = {
            "spectral_projector": self.spectral_projector.state_dict(),
            "bifrost": self.bifrost.state_dict(),
        }
        
        if hasattr(self, 'spectral_fusion'):
            adapter_state["spectral_fusion"] = self.spectral_fusion.state_dict()
        
        torch.save(adapter_state, path)
        print(f"Adapter saved to {path}")
    
    def load_adapter(self, path: str):
        """Load adapter weights."""
        adapter_state = torch.load(path, map_location='cpu')
        
        self.spectral_projector.load_state_dict(adapter_state["spectral_projector"])
        self.bifrost.load_state_dict(adapter_state["bifrost"])
        
        if "spectral_fusion" in adapter_state and hasattr(self, 'spectral_fusion'):
            self.spectral_fusion.load_state_dict(adapter_state["spectral_fusion"])
        
        print(f"Adapter loaded from {path}")
