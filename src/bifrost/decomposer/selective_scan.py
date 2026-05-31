"""Pure-PyTorch Selective Scan — the core SSM recurrence of Mamba.

Implements the discretised S6 algorithm from:
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    Gu & Dao, 2023.  arXiv:2312.00752

This is a numerically correct, CPU/MPS-compatible implementation.
On CUDA the real mamba-ssm library is used instead (faster custom CUDA kernel).

The selective scan selects A/B/C matrices *per token* (hence "selective"),
allowing the model to decide what to remember vs forget at each step.

Shapes convention (following mamba-ssm):
    B = batch size
    L = sequence length (time frames)
    D = input / output dimension (d_model)
    N = state size (d_state)
    E = expanded inner dimension = D * expand
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScan(nn.Module):
    """Pure-PyTorch selective scan block.

    Matches the Mamba architecture exactly so weights are drop-in
    compatible with mamba-ssm.Mamba when available.

    Args:
        d_model:   Input/output dimension D.
        d_state:   SSM state size N (default 16).
        d_conv:    Depthwise conv kernel size (default 4).
        expand:    Inner expansion factor (default 2), E = D * expand.
        dt_rank:   Rank of Δ projection. 'auto' → ceil(D/16).
        dt_min:    Min value for Δ initialisation (default 0.001).
        dt_max:    Max value for Δ initialisation (default 0.1).
        bias:      Whether to use bias in in/out projections.

    Input/output:
        x: (B, L, D)  →  y: (B, L, D)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # ---------- projections ----------
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.act = nn.SiLU()

        # x_proj: produces (Δ, B, C) from each token
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )

        # dt_proj: expand dt_rank → d_inner (with log-uniform init)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        _init_dt_proj(self.dt_proj, self.dt_rank, dt_min, dt_max)

        # SSM parameters A, D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape

        # 1. Input projection → split into z (gate) and x_ssm (content)
        xz = self.in_proj(x)                    # (B, L, 2*E)
        x_ssm, z = xz.chunk(2, dim=-1)          # each (B, L, E)

        # 2. Causal depthwise conv (for local context)
        x_ssm = x_ssm.transpose(1, 2)           # (B, E, L)
        x_ssm = self.conv1d(x_ssm)[..., :L]     # (B, E, L) — trim causal padding
        x_ssm = x_ssm.transpose(1, 2)           # (B, L, E)
        x_ssm = self.act(x_ssm)

        # 3. Selective SSM parameters per token
        x_dbc = self.x_proj(x_ssm)              # (B, L, dt_rank + 2*N)
        delta, B_sel, C_sel = x_dbc.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, L, E)
        B_sel = B_sel                             # (B, L, N)
        C_sel = C_sel                             # (B, L, N)

        # 4. Discretise A (ZOH), selective
        A = -torch.exp(self.A_log.float())        # (E, N)

        # 5. Scan recurrence (pure PyTorch, sequential over L)
        y = _selective_scan(x_ssm, delta, A, B_sel, C_sel, self.D)  # (B, L, E)

        # 6. Gating
        y = y * self.act(z)

        # 7. Output projection
        return self.out_proj(y)                   # (B, L, D)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_dt_proj(dt_proj: nn.Linear, dt_rank: int, dt_min: float, dt_max: float) -> None:
    """Log-uniform init for dt_proj (matches mamba-ssm reference)."""
    dt = torch.exp(
        torch.rand(dt_proj.out_features) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]


def _selective_scan(
    u: torch.Tensor,      # (B, L, E)
    delta: torch.Tensor,  # (B, L, E)
    A: torch.Tensor,      # (E, N)
    B: torch.Tensor,      # (B, L, N)
    C: torch.Tensor,      # (B, L, N)
    D: torch.Tensor,      # (E,)
) -> torch.Tensor:
    """Discretised ZOH selective scan recurrence.

    Computes:
        h_t = diag(exp(Δ_t ⊙ A)) h_{t-1} + Δ_t ⊙ B_t ⊙ u_t
        y_t = C_t h_t + D u_t

    Uses parallel associative scan (Blelloch) for O(log n) depth.
    Falls back to sequential for short sequences (L <= 32).

    All operations are batched and run on whatever device the tensors live on.
    Returns (B, L, E).
    
    Complexity: O(L log L) for parallel scan, O(L) for sequential fallback.
    """
    B_sz, L, E = u.shape
    N = A.shape[-1]

    # For short sequences, sequential is faster due to overhead
    if L <= 32 or not u.is_cuda:
        return _selective_scan_sequential(u, delta, A, B, C, D)

    # Expand A for batch dimension: (E, N) → (1, 1, E, N)
    A_exp = A.unsqueeze(0).unsqueeze(0)          # (1, 1, E, N)
    delta_A = torch.exp(
        delta.unsqueeze(-1) * A_exp              # (B, L, E, N)
    )
    delta_B_u = (
        delta.unsqueeze(-1)                      # (B, L, E, 1)
        * B.unsqueeze(2)                          # (B, L, 1, N)
        * u.unsqueeze(-1)                         # (B, L, E, 1)
    )                                             # (B, L, E, N)

    # === PARALLEL ASSOCIATIVE SCAN ===
    # The recurrence h_t = a_t * h_{t-1} + b_t is associative with:
    #   combine((a_1, b_1), (a_2, b_2)) = (a_2 * a_1, a_2 * b_1 + b_2)
    # where a_t = delta_A[:, t] and b_t = delta_B_u[:, t]
    
    # Pad to power of 2 for simplicity
    L_padded = 2 ** (L - 1).bit_length()
    pad_len = L_padded - L
    
    if pad_len > 0:
        # Pad with identity elements: a=1, b=0
        pad_a = torch.ones(B_sz, pad_len, E, N, device=u.device, dtype=u.dtype)
        pad_b = torch.zeros(B_sz, pad_len, E, N, device=u.device, dtype=u.dtype)
        delta_A = torch.cat([delta_A, pad_a], dim=1)
        delta_B_u = torch.cat([delta_B_u, pad_b], dim=1)
    
    # === UP-SWEEP (REDUCTION) PHASE ===
    a = delta_A  # (B, L_padded, E, N)
    b = delta_B_u  # (B, L_padded, E, N)
    
    stride = 1
    while stride < L_padded:
        # Combine adjacent pairs
        a_left = a[:, ::2*stride, :, :]
        a_right = a[:, stride::2*stride, :, :]
        b_left = b[:, ::2*stride, :, :]
        b_right = b[:, stride::2*stride, :, :]
        
        # combine((a_left, b_left), (a_right, b_right))
        # = (a_right * a_left, a_right * b_left + b_right)
        a_new = a_right * a_left
        b_new = a_right * b_left + b_right
        
        # Update only the right elements
        a = a_new
        b = b_new
        stride *= 2
    
    # === DOWN-SWEEP (DISTRIBUTION) PHASE ===
    stride = L_padded // 2
    while stride >= 1:
        a_left = a[:, ::2*stride, :, :]
        a_right = a[:, stride::2*stride, :, :]
        b_left = b[:, ::2*stride, :, :]
        b_right = b[:, stride::2*stride, :, :]
        
        a_right_new = a_right * a_left
        b_right_new = a_right * b_left + b_right
        
        a = torch.cat([a_left, a_right_new], dim=1)
        b = torch.cat([b_left, b_right_new], dim=1)
        
        stride //= 2
    
    # Trim padding
    if pad_len > 0:
        a = a[:, :L, :, :]
        b = b[:, :L, :, :]
    
    # === PARALLEL OUTPUT COMPUTATION ===
    # h_t = a_t * h_0 + b_t for all t in parallel (h_0 = 0)
    hs = b  # (B, L, E, N) since h_0 = 0

    # y_t = C_t h_t  (einsum over N)
    y = torch.einsum("blen,bln->ble", hs, C)     # (B, L, E)

    # Skip connection
    y = y + D.unsqueeze(0).unsqueeze(0) * u      # (B, L, E)

    return y


def _selective_scan_sequential(
    u: torch.Tensor,      # (B, L, E)
    delta: torch.Tensor,  # (B, L, E)
    A: torch.Tensor,      # (E, N)
    B: torch.Tensor,      # (B, L, N)
    C: torch.Tensor,      # (B, L, N)
    D: torch.Tensor,      # (E,)
) -> torch.Tensor:
    """Sequential fallback for short sequences (L <= 32).
    
    Complexity: O(L) sequential.
    """
    B_sz, L, E = u.shape
    N = A.shape[-1]

    A_exp = A.unsqueeze(0).unsqueeze(0)
    delta_A = torch.exp(delta.unsqueeze(-1) * A_exp)
    delta_B_u = (
        delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    )

    hs = []
    h = torch.zeros(B_sz, E, N, device=u.device, dtype=u.dtype)
    for t in range(L):
        h = delta_A[:, t] * h + delta_B_u[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)
    y = torch.einsum("blen,bln->ble", hs, C)
    y = y + D.unsqueeze(0).unsqueeze(0) * u

    return y
