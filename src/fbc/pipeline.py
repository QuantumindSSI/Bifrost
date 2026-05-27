"""
FBCPipeline — end-to-end canonicalize → decompose → bind orchestrator.

Connects the data ingest layer to the FBC neural pipeline stages,
producing coherence-bound spectral embeddings ready for attractor
identification.

Usage:
    from fbc.pipeline import FBCPipeline

    pipe = FBCPipeline()
    bound_st, coherence = pipe.process_signal(audio_tensor, metadata)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .spectral_tensor import SpectralTensor
from .canonicalizer import SpectralCanonicalizer
from .decomposer import SpectralDecomposer
from .s1_decomposer.complex_decomposer import ComplexSpectralDecomposer
from .resonance_attention import ResonanceAttention, SpectralBinding


class FBCPipeline(nn.Module):
    """
    Full canonicalize → decompose → bind pipeline.

    Parameters
    ----------
    n_fft_s0 : int
        FFT size for canonicalization.
    n_fft_s1 : int
        FFT size for decomposition sub-band analysis.
    n_scales : int
        Number of wavelet scales in decomposition.
    d_model : int
        Hidden / feature dimension for decomposition and binding.
    n_heads : int
        Number of resonance attention heads.
    n_bands : int
        Number of spectral bands for coherence.
    dropout : float
        Attention dropout rate.
    """

    def __init__(
        self,
        n_fft_s0: int = 1024,
        n_fft_s1: int = 512,
        n_scales: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_bands: int = 8,
        dropout: float = 0.1,
        preserve_frames: bool = True,  # Enable meaningful attention
        use_mamba: bool = True,  # Use real Mamba-3 when CUDA available
        use_2d_fft: bool = False,  # Use 2D FFT for spatial data
        use_complex_ssm: bool = False,  # Use complex-valued SSM for true phase coherence
    ) -> None:
        super().__init__()
        self.use_complex_ssm = use_complex_ssm

        self.canonicalizer = SpectralCanonicalizer(
            n_fft=n_fft_s0,
            preserve_frames=preserve_frames,
            use_2d_fft=use_2d_fft,
        )

        # decomposer output n_freq = n_fft_s1 // 2 + 1
        n_freq_decomp = n_fft_s1 // 2 + 1

        if use_complex_ssm:
            # Complex SSM processes amplitude+phase jointly for true coherence learning
            # Input n_freq, internal d_model, output d_model (matches binding expectation)
            self.decomposer = ComplexSpectralDecomposer(
                n_fft=n_fft_s1,
                d_model=d_model,
                n_frames=32,
            )
        else:
            # Dual-stream SSM (default, backward compatible)
            self.decomposer = SpectralDecomposer(
                n_fft=n_fft_s1,
                n_scales=n_scales,
                d_model=n_freq_decomp,
                use_mamba=use_mamba,
            )

        self.binding = SpectralBinding(
            d_model=d_model,
            n_heads=n_heads,
            n_bands=n_bands,
            dropout=dropout,
        )

        # Bridge projection if decomposer output dim != binding d_model
        # Complex SSM already outputs d_model features, no projection needed
        if use_complex_ssm:
            self._decomp_to_bind_proj = None  # Complex decomposer outputs d_model directly
        elif n_freq_decomp != d_model:
            self._decomp_to_bind_proj = nn.Linear(n_freq_decomp, d_model)
        else:
            self._decomp_to_bind_proj = None

    def forward(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Run the full canonicalize → decompose → bind pipeline.

        Parameters
        ----------
        signal : torch.Tensor
            Raw time-domain signal (1-D, 2-D, or batched 3-D).
        metadata : dict, optional
            Provenance metadata (e.g. sample_rate).

        Returns
        -------
        bound_st : SpectralTensor
            Coherence-bound spectral embedding.
        coherence : torch.Tensor
            Attention coherence weights from binding.
        """
        canonical = self.canonicalizer(signal, metadata)
        decomposed = self.decomposer(canonical)
        bound_st, coherence = self.binding(decomposed, input_proj=self._decomp_to_bind_proj)
        return bound_st, coherence

    @property
    def ssm_type(self) -> str:
        """Return the SSM architecture being used."""
        if self.use_complex_ssm:
            return "ComplexSpectralDecomposer (phase coherence learned)"
        elif hasattr(self.decomposer, 'ssm_type'):
            return self.decomposer.ssm_type
        return "Standard SpectralDecomposer"

    def process_numpy(
        self,
        array: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """Convenience: accept a NumPy array from the ingest layer."""
        tensor = torch.from_numpy(array.astype(np.float32))
        return self.forward(tensor, metadata)
