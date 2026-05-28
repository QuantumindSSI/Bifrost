"""
ResonanceAttention — phase-coherence routing mechanism.

Replaces dot-product attention with spectral phase coherence:
    1. Project inputs into Q, K, V in spectral space.
    2. Compute coherence matrix C using phase alignment across bands.
    3. Temperature-scale via learnable tau.
    4. Multi-head attention across spectral subspaces.
    5. Weighted aggregation of values using coherence weights.

Key properties (per Engineering Script S2):
    - Signals with identical amplitude but opposite phase are disambiguated.
    - Phase coherence captures temporal and oscillatory relationships.
    - Supports binding of multimodal / cross-domain structure.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResonanceAttention(nn.Module):
    """
    Multi-head Resonance Attention.

    Parameters
    ----------
    d_model : int
        Input / output feature dimension.
    n_heads : int
        Number of attention heads (must divide d_model).
    n_bands : int
        Number of spectral bands for coherence computation.
    dropout : float
        Dropout on attention weights.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_bands: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_bands = n_bands

        # Q, K, V projections (operate in spectral space)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable temperature per head.
        # Coherence scores are cosines in [-1, 1], NOT dot products in [-sqrt(d),sqrt(d)].
        # tau=sqrt(d_head)~5.66 crushes cosines to ~0 before softmax (dead gradient).
        # tau=1.0 is the identity at init: cosine scores enter softmax unchanged [-1,1].
        # tau is learned and will sharpen during training as phase coherence emerges.
        self.tau = nn.Parameter(torch.ones(n_heads) * 1.0)

        # Band-specific learnable weights for coherence aggregation
        self.band_weights = nn.Parameter(torch.ones(n_bands) / n_bands)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)
            Input features (amplitude-weighted spectral embedding).
        phase : (batch, seq_len, d_model) or None
            Phase information.  If None, phase is extracted from x via
            a Hilbert-style analytic signal approximation.
        mask : (batch, seq_len, seq_len) or None
            Optional boolean mask (True = ignore).

        Returns
        -------
        out : (batch, seq_len, d_model)
            Coherence-weighted output.
        coherence : (batch, n_heads, seq_len, seq_len)
            Coherence attention weights (for diagnostics / visualisation).
        """
        B, S, D = x.shape

        # --- project Q, K, V ------------------------------------------------
        Q = self._reshape_heads(self.W_q(x))  # (B, H, S, d_head)
        K = self._reshape_heads(self.W_k(x))
        V = self._reshape_heads(self.W_v(x))

        # --- obtain phase for Q and K ---------------------------------------
        if phase is not None:
            Q_phase = self._reshape_heads(phase)
            K_phase = self._reshape_heads(phase)
        else:
            Q_phase = self._extract_phase(Q)
            K_phase = self._extract_phase(K)

        # --- compute phase-coherence matrix ----------------------------------
        coherence = self._phase_coherence(Q_phase, K_phase)  # (B, H, S, S)

        # --- temperature scaling per head ------------------------------------
        tau = self.tau.view(1, self.n_heads, 1, 1).clamp(min=0.05, max=2.0)
        coherence = coherence / tau

        # --- optional mask ---------------------------------------------------
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, S, S)
            coherence = coherence.masked_fill(mask, float("-inf"))

        # --- softmax normalisation -------------------------------------------
        weights = F.softmax(coherence, dim=-1)
        weights = self.attn_dropout(weights)

        # --- weighted aggregation of values ----------------------------------
        out = torch.matmul(weights, V)  # (B, H, S, d_head)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.W_o(out)
        out = self.norm(out + x)  # residual + norm

        return out, weights

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, D) -> (B, H, S, d_head)"""
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def _extract_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate analytic-signal phase via per-head FFT.
        x : (B, H, S, d_head) -> phase : same shape
        """
        spec = torch.fft.rfft(x, dim=-1)
        return spec.angle().float()

    def _phase_coherence(
        self,
        phase_q: torch.Tensor,
        phase_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-band phase coherence between Q and K.

        For each spectral band b:
            C_b(i, j) = mean_{k in band_b}  cos(phase_q[i, k] - phase_k[j, k])

        Final coherence:
            C(i, j) = Σ_b  w_b * C_b(i, j)

        Correct temporal coherence formula:
            coherence[i, j] = mean_f(cos(phase_q[i, f] - phase_k[j, f]))

        Returns shape (B, H, S_q, S_k).
        """
        d = phase_q.shape[-1]
        n_bands = min(self.n_bands, d)
        band_size = max(1, d // n_bands)
        usable = n_bands * band_size  # truncate any remainder

        # Reshape last dim → (n_bands, band_size)
        # phase_*: (B, H, S, d) → (B, H, S, n_bands, band_size)
        pq = phase_q[..., :usable].view(*phase_q.shape[:-1], n_bands, band_size)
        pk = phase_k[..., :usable].view(*phase_k.shape[:-1], n_bands, band_size)

        # cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
        # Avoids materialising (B, H, n_bands, band_size, S_q, S_k) intermediate.
        # pq/pk: (B, H, S, n_bands, band_size)
        # Reshape to (B, H, n_bands, S, band_size) for batched matmul
        pq_r = pq.permute(0, 1, 3, 2, 4)  # (B, H, n_bands, S_q, band_size)
        pk_r = pk.permute(0, 1, 3, 2, 4)  # (B, H, n_bands, S_k, band_size)

        # cos(pq) @ cos(pk).T + sin(pq) @ sin(pk).T  →  (B, H, n_bands, S_q, S_k)
        # Each matmul: (B, H, n_bands, S_q, band_size) @ (B, H, n_bands, band_size, S_k)
        cos_per_band = (
            torch.matmul(torch.cos(pq_r), torch.cos(pk_r).transpose(-2, -1))
            + torch.matmul(torch.sin(pq_r), torch.sin(pk_r).transpose(-2, -1))
        ) / band_size  # normalise by band_size to match mean semantics

        # Weighted sum over n_bands (dim 2)
        w = F.softmax(self.band_weights[:n_bands], dim=0)  # (n_bands,)
        coherence = (cos_per_band * w.view(1, 1, n_bands, 1, 1)).sum(dim=2)  # (B, H, S_q, S_k)

        return coherence
