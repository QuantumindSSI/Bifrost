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
        """
        Initialize multi-head resonance attention.

        Parameters
        ----------
        d_model : int
            Input / output feature dimension. Must be > 0 and divisible by n_heads.
        n_heads : int
            Number of attention heads. Must be > 0 and divide d_model.
        n_bands : int
            Number of spectral bands for coherence computation. Must be > 0.
        dropout : float
            Dropout on attention weights. Must be in [0, 1].

        Raises
        ------
        ValueError
            If d_model <= 0, n_heads <= 0, n_bands <= 0, dropout not in [0, 1],
            or d_model not divisible by n_heads.

        Complexity
        ----------
        O(1) - initialization only.

        Side Effects
        ------------
        Registers learnable parameters: W_q, W_k, W_v, W_o, tau, band_weights.
        """
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if n_bands <= 0:
            raise ValueError(f"n_bands must be > 0, got {n_bands}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} % {n_heads} = {d_model % n_heads}")

        super().__init__()
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
        precomputed_coherence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute coherence-weighted attention output.

        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len, d_model) Input features (amplitude-weighted spectral embedding).
            Must be 3D and finite.
        phase : torch.Tensor, optional
            (batch, seq_len, d_model) Phase information. If None, phase is extracted from x
            via Hilbert-style analytic signal approximation. Must match x shape and be finite.
        mask : torch.Tensor, optional
            (batch, seq_len, seq_len) Optional boolean mask (True = ignore).
        precomputed_coherence : torch.Tensor, optional
            (batch, n_heads, seq_len, seq_len) When supplied, skips all internal
            phase/coherence computation and uses this tensor directly as attention weights
            (after softmax). Gradients flow only through W_v and W_o — not through coherence.
            This is the collapse-proof path: coherence is computed from raw canonical STFT
            phase (zero learned parameters) in SpectralBinding, then passed here.
            W_q/W_k cannot collapse it to uniform. Must be 4D and finite.

        Returns
        -------
        torch.Tensor
            (batch, seq_len, d_model) Coherence-weighted output.
        torch.Tensor
            (batch, n_heads, seq_len, seq_len) Coherence attention weights
            (for diagnostics / visualisation).

        Raises
        ------
        ValueError
            If x is not 3D, if phase/mask/precomputed_coherence have invalid shapes,
            or if any tensor contains NaN/Inf.

        Complexity
        ----------
        O(B * H * S^2 * d_head) - attention computation with phase coherence.

        Side Effects
        ------------
        None.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, S, D), got shape {x.shape}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values")
        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(f"x feature dimension {D} does not match d_model {self.d_model}")

        if phase is not None:
            if phase.shape != x.shape:
                raise ValueError(f"phase shape {phase.shape} must match x shape {x.shape}")
            if not torch.isfinite(phase).all():
                raise ValueError("phase contains NaN or Inf values")

        if mask is not None:
            if mask.dim() not in [3, 4]:
                raise ValueError(f"mask must be 3D or 4D, got shape {mask.shape}")
            if mask.shape[0] != B or mask.shape[1] != S or mask.shape[2] != S:
                raise ValueError(
                    f"mask shape {mask.shape} must match batch and seq dimensions ({B}, {S}, {S})"
                )

        if precomputed_coherence is not None:
            if precomputed_coherence.dim() != 4:
                raise ValueError(f"precomputed_coherence must be 4D, got shape {precomputed_coherence.shape}")
            if not torch.isfinite(precomputed_coherence).all():
                raise ValueError("precomputed_coherence contains NaN or Inf values")

        # --- project V only (Q, K not needed when coherence is precomputed) --
        V = self._reshape_heads(self.W_v(x))  # (B, H, S, d_head)

        if precomputed_coherence is not None:
            # Coherence is from raw canonical phase — parameter-free, collapse-proof.
            # Broadcast to n_heads if supplied as (B, 1, S, S).
            coh = precomputed_coherence
            if coh.shape[1] == 1 and self.n_heads > 1:
                coh = coh.expand(-1, self.n_heads, -1, -1)
            if coh.shape != (B, self.n_heads, S, S):
                raise ValueError(
                    f"precomputed_coherence shape {coh.shape} must be ({B}, {self.n_heads}, {S}, {S})"
                )
            weights = F.softmax(coh / self.tau.view(1, self.n_heads, 1, 1).clamp(min=0.05, max=2.0), dim=-1)
            weights = self.attn_dropout(weights)
            out = torch.matmul(weights, V)
            out = out.transpose(1, 2).contiguous().view(B, S, D)
            out = self.W_o(out)
            out = self.norm(out + x)
            # Return pre-softmax coherence (full dynamic range ~[-0.5, 1.0], var~0.05-0.07)
            # NOT post-softmax weights (compressed ~75000:1, var~1e-5, gap~1e-6).
            # The training loss operates on the returned tensor via .var() — it must
            # have the pre-softmax dynamic range to give a meaningful gradient signal.
            return out, coh

        # --- project Q, K for phase coherence computation -----------------
        Q = self._reshape_heads(self.W_q(x))
        K = self._reshape_heads(self.W_k(x))

        # --- obtain phase for coherence computation -------------------------
        if phase is not None:
            ph = self._reshape_heads(phase)  # (B, H, S, d_head)
            Q_phase = ph
            K_phase = ph
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
        """
        Reshape tensor for multi-head attention.

        Args
        ----
        x : torch.Tensor
            (B, S, D) Input tensor. Must be 3D with D = d_model.

        Returns
        -------
        torch.Tensor
            (B, H, S, d_head) Reshaped tensor.

        Raises
        ------
        ValueError
            If x is not 3D or if feature dimension does not match d_model.

        Complexity
        ----------
        O(B * S * D) - reshape and transpose operations.

        Side Effects
        ------------
        None.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, S, D), got shape {x.shape}")
        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(f"x feature dimension {D} does not match d_model {self.d_model}")
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def _extract_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate analytic-signal phase via per-head FFT.

        Args
        ----
        x : torch.Tensor
            (B, H, S, d_head) Input tensor. Must be 4D and finite.

        Returns
        -------
        torch.Tensor
            (B, H, S, d_head) Phase tensor with same shape as input.

        Raises
        ------
        ValueError
            If x is not 4D or contains NaN/Inf values.

        Complexity
        ----------
        O(B * H * S * d_head * log(d_head)) - FFT computation.

        Side Effects
        ------------
        None.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be 4D (B, H, S, d_head), got shape {x.shape}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values")
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

        Args
        ----
        phase_q : torch.Tensor
            (B, H, S_q, d) Query phase tensor. Must be 4D and finite.
        phase_k : torch.Tensor
            (B, H, S_k, d) Key phase tensor. Must be 4D and finite.

        Returns
        -------
        torch.Tensor
            (B, H, S_q, S_k) Phase coherence matrix.

        Raises
        ------
        ValueError
            If inputs are not 4D, have mismatched shapes, or contain NaN/Inf.

        Complexity
        ----------
        O(B * H * n_bands * (S_q * S_k * band_size + S_q * band_size + S_k * band_size))
        - matrix multiplications and trigonometric operations.

        Side Effects
        ------------
        None.
        """
        if phase_q.dim() != 4:
            raise ValueError(f"phase_q must be 4D (B, H, S, d), got shape {phase_q.shape}")
        if phase_k.dim() != 4:
            raise ValueError(f"phase_k must be 4D (B, H, S, d), got shape {phase_k.shape}")
        if phase_q.shape[:-1] != phase_k.shape[:-1]:
            raise ValueError(
                f"phase_q shape {phase_q.shape} and phase_k shape {phase_k.shape} "
                f"must match except for sequence dimension"
            )
        if phase_q.shape[-1] != phase_k.shape[-1]:
            raise ValueError(
                f"phase_q feature dim {phase_q.shape[-1]} must match phase_k feature dim {phase_k.shape[-1]}"
            )
        if not torch.isfinite(phase_q).all():
            raise ValueError("phase_q contains NaN or Inf values")
        if not torch.isfinite(phase_k).all():
            raise ValueError("phase_k contains NaN or Inf values")

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
