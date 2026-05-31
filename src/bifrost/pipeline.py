"""
BifrostPipeline — end-to-end canonicalize → decompose → bind orchestrator.

Connects the data ingest layer to the Bifröst spectral pipeline stages,
producing coherence-bound spectral embeddings ready for attractor
identification.

Usage:
    from bifrost.pipeline import FBCPipeline, BifrostPipeline

    pipe = BifrostPipeline()
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
from .resonance_attention.harmonic_binding import HarmonicBinding


class BifrostPipeline(nn.Module):
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
        use_complex_ssm: bool = True,  # Default: complex-valued SSM for true phase coherence
        use_harmonic_binding: bool = False,  # Wire HarmonicBinding (explicit 440Hz↔4880Hz grid)
        sample_rate: float = 16000.0,  # Used by HarmonicBinding frequency grid
        use_s3_attractor: bool = True,  # Enable learned S3 attractor dynamics (CRITICAL_AUDIT fix)
    ) -> None:
        super().__init__()
        self.use_complex_ssm = use_complex_ssm
        self.use_harmonic_binding = use_harmonic_binding
        self.use_s3_attractor = use_s3_attractor
        
        # === CRITICAL AUDIT WARNINGS ===
        # Per Agentic CTO-Persona policy, these limitations are explicitly disclosed
        import warnings
        
        # === S3 ATTRACTOR LEARNING MODULE (OPTIONAL) ===
        # Per CRITICAL_AUDIT.md remediation: Integrate learned attractor dynamics
        # This replaces the placeholder stability=0.5 with learned stability prediction
        if self.use_s3_attractor:
            try:
                from .s3_attractor.attractor_learning import AttractorLearningModule
                self.attractor_learner = AttractorLearningModule(
                    d_model=d_model,
                    n_bands=n_bands,
                    n_attractors=16,
                )
                warnings.warn(
                    "Bifrost Pipeline: S3 (Phase-Lock Bridge) using LEARNED attractor module. "
                    "Neural stability prediction active. See CRITICAL_AUDIT.md for audit trail.",
                    UserWarning,
                    stacklevel=2
                )
            except ImportError:
                self.attractor_learner = None
                warnings.warn(
                    "Bifrost Pipeline: S3 (Phase-Lock Bridge) contains placeholder values (stability=0.5). "
                    "True attractor learning not implemented. See CRITICAL_AUDIT.md",
                    UserWarning,
                    stacklevel=2
                )
        else:
            self.attractor_learner = None
        
        
        if not use_complex_ssm:
            warnings.warn(
                "Bifrost Pipeline: Real SSM mode uses different architecture than Complex SSM. "
                "Switching between modes gives non-equivalent results.",
                UserWarning,
                stacklevel=2
            )

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
            # Use d_model (not n_freq_decomp) so output matches binding expectation
            self.decomposer = SpectralDecomposer(
                n_fft=n_fft_s1,
                n_scales=n_scales,
                d_model=d_model,
                use_mamba=use_mamba,
            )

        # For the dual-stream decomposer the binding receives n_freq_decomp-dimensional
        # phase tensors; passing n_freq_in activates the harmonic-preserving original-phase
        # coherence path in SpectralBinding (use_original_phase=True).
        # For the complex SSM the decomposer already outputs d_model, so no projection needed.
        binding_n_freq_in = None if use_complex_ssm else n_freq_decomp
        if use_harmonic_binding:
            # HarmonicBinding: explicit 440Hz↔4880Hz frequency grid wired into attention.
            # n_freq = n_fft_s1 // 2 + 1 (the frequency dimension of the decomposer output
            # before d_model projection; used by the harmonic grid for bin mapping).
            self.binding = HarmonicBinding(
                d_model=d_model,
                n_heads=n_heads,
                n_freq=n_freq_decomp,
                n_bands=n_bands,
                sample_rate=sample_rate,
                dropout=dropout,
            )
        else:
            self.binding = SpectralBinding(
                d_model=d_model,
                n_heads=n_heads,
                n_bands=n_bands,
                dropout=dropout,
                n_freq_in=binding_n_freq_in,
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
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Run the full canonicalize → decompose → bind pipeline.

        Parameters
        ----------
        signal : torch.Tensor
            Raw time-domain signal (1-D, 2-D, or batched 3-D).
        metadata : dict, optional
            Provenance metadata (e.g. sample_rate).
        h_0 : torch.Tensor, optional
            Initial SSM hidden state (B, d_inner, d_state) complex.
            None → stateless (zeros). Only used when use_complex_ssm=True.
            Pass the h_T returned by a previous call for streaming use.

        Returns
        -------
        bound_st : SpectralTensor
            Coherence-bound spectral embedding.
        coherence : torch.Tensor
            Attention coherence weights from binding.
        """
        if signal.numel() == 0:
            raise ValueError("Input signal is empty (numel=0).")
        
        # === INPUT VALIDATION ASSERTIONS ===
        assert signal.dtype == torch.float32, f"Expected float32 input signal, got {signal.dtype}"
        assert signal.dim() >= 1, f"Expected 1D+ signal, got shape {signal.shape}"
        assert torch.isfinite(signal).all(), "Non-finite values in input signal"
        
        canonical = self.canonicalizer(signal, metadata)
        if self.use_complex_ssm:
            decomposed, _ = self.decomposer(canonical, h_0)
        else:
            decomposed = self.decomposer(canonical)
        if self.use_harmonic_binding:
            # HarmonicBinding takes (amplitude, phase) tensors directly.
            amp = decomposed.amplitude
            phase = decomposed.phase
            if amp.dim() == 2:
                amp = amp.unsqueeze(1)
                phase = phase.unsqueeze(1)
            if amp.shape[1] == 1:
                raise ValueError(
                    "Single-token input (T=1) produces trivially uniform coherence. "
                    "Pass a signal long enough to produce T > 1 spectral frames."
                )
            bound_amp, coherence = self.binding(amp, phase=phase)
            bound_st = SpectralTensor(
                amplitude=bound_amp,
                phase=phase,
                scale=decomposed.scale,
                uncertainty=decomposed.uncertainty,
                metadata=decomposed.metadata,
            )
        else:
            amp = decomposed.amplitude
            if amp.dim() == 3 and amp.shape[1] == 1:
                import warnings
                warnings.warn(
                    "SpectralBinding received T=1 (single token). "
                    "Coherence will be trivially 1.0 — phase carries no information. "
                    "Provide longer signals for meaningful phase-coherence routing.",
                    stacklevel=2,
                )
            # Validate canonical has amplitude attribute
            if not hasattr(canonical, 'amplitude') or canonical.amplitude is None:
                raise ValueError("canonical must have amplitude attribute for harmonic coherence detection")

            bound_st, coherence = self.binding(
                decomposed,
                input_proj=self._decomp_to_bind_proj,
                canonical_phase=canonical.phase,
                canonical_amplitude=canonical.amplitude,  # Raw STFT amplitude for harmonic detection
            )
        
        # === S3 ATTRACTOR LEARNING (Optional) ===
        # Extract learned attractors from spectral binding output
        if hasattr(self, 'attractor_learner') and self.attractor_learner is not None:
            try:
                attractors, assignment_probs = self.attractor_learner(bound_st)
                # Add attractor info to metadata
                bound_st.metadata['s3_attractors'] = len(attractors)
                bound_st.metadata['attractor_stabilities'] = [a.stability for a in attractors]
            except Exception as e:
                # Log error per policy C-02 (no silent failures)
                import warnings
                warnings.warn(
                    f"S3 Attractor Learning failed: {str(e)}. Skipping attractor extraction.",
                    RuntimeWarning,
                    stacklevel=2
                )
                bound_st.metadata['s3_attractors'] = 0
                bound_st.metadata['attractor_error'] = str(e)
        
        return bound_st, coherence

    def forward_stateful(
        self,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor, torch.Tensor]:
        """
        Stateful forward pass — returns h_T for chaining across chunks.

        Parameters
        ----------
        signal : torch.Tensor
            Raw time-domain signal.
        metadata : dict, optional
            Provenance metadata.
        h_0 : torch.Tensor, optional
            Initial SSM hidden state from a previous call. None → zeros.

        Returns
        -------
        bound_st : SpectralTensor
        coherence : torch.Tensor
        h_T : torch.Tensor
            Final SSM hidden state (B, d_inner, d_state) complex, detached.
            Pass as h_0 to the next call to maintain temporal continuity.
        """
        if not self.use_complex_ssm:
            raise RuntimeError(
                "forward_stateful requires use_complex_ssm=True. "
                "The non-complex decomposer does not expose hidden state."
            )
        if signal.numel() == 0:
            raise ValueError("Input signal is empty (numel=0).")
        canonical = self.canonicalizer(signal, metadata)
        decomposed, h_T = self.decomposer(canonical, h_0)
        if self.use_harmonic_binding:
            amp = decomposed.amplitude
            phase = decomposed.phase
            if amp.dim() == 2:
                amp = amp.unsqueeze(1)
                phase = phase.unsqueeze(1)
            if amp.shape[1] == 1:
                raise ValueError(
                    "Single-token input (T=1) produces trivially uniform coherence."
                )
            bound_amp, coherence = self.binding(amp, phase=phase)
            bound_st = SpectralTensor(
                amplitude=bound_amp,
                phase=phase,
                scale=decomposed.scale,
                uncertainty=decomposed.uncertainty,
                metadata=decomposed.metadata,
            )
        else:
            bound_st, coherence = self.binding(decomposed, input_proj=self._decomp_to_bind_proj)
        return bound_st, coherence, h_T

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
