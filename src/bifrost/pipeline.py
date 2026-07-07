"""
BifrostPipeline — end-to-end canonicalize → decompose → bind orchestrator.

Connects the data ingest layer to the Bifröst spectral pipeline stages,
producing coherence-bound spectral embeddings ready for attractor
identification.

Usage:
    from bifrost.pipeline import BifrostPipeline

    pipe = BifrostPipeline()
    bound_st, coherence = pipe.process_signal(audio_tensor, metadata)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import warnings

import numpy as np
import torch
import torch.nn as nn

from .spectral_tensor import SpectralTensor
from .canonicalizer import SpectralCanonicalizer
from .decomposer import SpectralDecomposer
from .decomposer.complex_decomposer import ComplexSpectralDecomposer
from .resonance_attention import ResonanceAttention, SpectralBinding
from .resonance_attention.harmonic_binding import HarmonicBinding


class BifrostPipeline(nn.Module):
    """
    Full canonicalize → decompose → bind pipeline.

    Parameters
    ----------
    n_fft_canonical : int
        FFT size for canonicalization.
    n_fft_decompose : int
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
        n_fft_canonical: int = 1024,
        n_fft_decompose: int = 512,
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
        use_s3_attractor: bool = True,  # Enable learned attractor dynamics
        use_riemannian_semantic: bool = False,  # Enable Riemannian semantic coherence
        riemannian_metric_dim: int = 64,  # Manifold dimension for semantic coherence
        use_cbmpc: bool = False,  # Enable CBMPC pre-SSM feature extraction
        cbmpc_n_mels: int = 64,  # Number of mel bands for CBMPC
        cbmpc_modulation_freqs: Optional[list] = None,  # Modulation frequencies for CBMPC
        cbmpc_duration_seconds: float = 1.0,  # Expected clip duration for CBMPC
    ) -> None:
        super().__init__()
        self.use_complex_ssm = use_complex_ssm
        self.use_harmonic_binding = use_harmonic_binding
        self.use_s3_attractor = use_s3_attractor
        self.use_riemannian_semantic = use_riemannian_semantic
        self.use_cbmpc = use_cbmpc
        self.sample_rate = sample_rate

        self._init_attractor_learner(d_model, n_bands)
        self._init_riemannian_semantic(d_model, riemannian_metric_dim)
        self._init_cbmpc_extractor(
            use_cbmpc, sample_rate, n_fft_canonical,
            cbmpc_n_mels, cbmpc_modulation_freqs, cbmpc_duration_seconds,
        )

        if not use_complex_ssm:
            warnings.warn(
                "Bifrost Pipeline: Real SSM mode uses different architecture "
                "than Complex SSM. Switching between modes gives "
                "non-equivalent results.",
                UserWarning,
                stacklevel=2,
            )

        self.canonicalizer = SpectralCanonicalizer(
            n_fft=n_fft_canonical,
            preserve_frames=preserve_frames,
            use_2d_fft=use_2d_fft,
        )

        n_freq_decomp = n_fft_decompose // 2 + 1
        self.decomposer = self._build_decomposer(
            n_fft_decompose, n_scales, d_model, use_mamba
        )
        self.binding = self._build_binding(
            d_model, n_heads, n_bands, n_freq_decomp, sample_rate,
            dropout, use_complex_ssm, use_harmonic_binding,
        )
        self._decomp_to_bind_proj = self._build_bridge_projection(
            n_freq_decomp, d_model, use_complex_ssm,
        )

    def _init_attractor_learner(self, d_model: int, n_bands: int) -> None:
        """Initialize optional attractor learning module."""
        if not self.use_s3_attractor:
            self.attractor_learner = None
            return

        try:
            from .s3_attractor import AttractorLearningModule
            self.attractor_learner = AttractorLearningModule(
                d_model=d_model,
                n_bands=n_bands,
                n_attractors=16,
            )
            warnings.warn(
                "Bifrost Pipeline: Phase-Lock Bridge using learned attractor "
                "module. Neural stability prediction active.",
                UserWarning,
                stacklevel=2,
            )
        except ImportError:
            self.attractor_learner = None
            warnings.warn(
                "Bifrost Pipeline: Phase-Lock Bridge using placeholder "
                "stability values. Attractor learning module not available.",
                UserWarning,
                stacklevel=2,
            )

    def _init_riemannian_semantic(self, d_model: int, metric_dim: int) -> None:
        """Initialize optional Riemannian semantic coherence module."""
        self.riemannian_semantic_coherence = None
        if not self.use_riemannian_semantic:
            return

        if not self.use_s3_attractor:
            raise ValueError(
                "Riemannian semantic coherence requires use_s3_attractor=True "
                "to provide FrequencyAttractors"
            )
        try:
            from .riemannian_coherence import RiemannianSemanticCoherence
            self.riemannian_semantic_coherence = RiemannianSemanticCoherence(
                d_model=d_model,
                metric_dim=metric_dim,
                k_neighbors=min(5, 16),
                manifold_dim=2,
            )
            warnings.warn(
                "Bifrost Pipeline: Riemannian Semantic Coherence enabled. "
                "Semantic manifold learning active.",
                UserWarning,
                stacklevel=2,
            )
        except ImportError as e:
            warnings.warn(
                f"Bifrost Pipeline: Riemannian coherence import failed "
                f"({str(e)}). Semantic coherence disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.use_riemannian_semantic = False

    def _init_cbmpc_extractor(
        self,
        use_cbmpc: bool,
        sample_rate: float,
        n_fft: int,
        n_mels: int,
        modulation_freqs: Optional[list],
        duration_seconds: float,
    ) -> None:
        """Initialize optional CBMPC pre-SSM feature extractor."""
        self.cbmpc_extractor = None
        if not use_cbmpc:
            return
        try:
            from .cbmpc import CBMPCExtractor
            self.cbmpc_extractor = CBMPCExtractor(
                sample_rate=int(sample_rate),
                n_fft=n_fft,
                hop_length=n_fft // 2,
                n_mels=n_mels,
                modulation_freqs=modulation_freqs,
                duration_seconds=duration_seconds,
                feature_mode="rich",
            )
        except ImportError as e:
            warnings.warn(
                f"Bifrost Pipeline: CBMPC extractor import failed ({str(e)}). "
                f"CBMPC feature extraction disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.use_cbmpc = False

    def _build_decomposer(
        self,
        n_fft: int,
        n_scales: int,
        d_model: int,
        use_mamba: bool,
    ) -> nn.Module:
        """Build spectral decomposer (complex or dual-stream)."""
        if self.use_complex_ssm:
            return ComplexSpectralDecomposer(
                n_fft=n_fft,
                d_model=d_model,
                n_frames=32,
            )
        return SpectralDecomposer(
            n_fft=n_fft,
            n_scales=n_scales,
            d_model=d_model,
            use_mamba=use_mamba,
        )

    def _build_binding(
        self,
        d_model: int,
        n_heads: int,
        n_bands: int,
        n_freq_decomp: int,
        sample_rate: float,
        dropout: float,
        use_complex_ssm: bool,
        use_harmonic_binding: bool,
    ) -> nn.Module:
        """Build spectral binding module."""
        if use_harmonic_binding:
            return HarmonicBinding(
                d_model=d_model,
                n_heads=n_heads,
                n_freq=n_freq_decomp,
                n_bands=n_bands,
                sample_rate=sample_rate,
                dropout=dropout,
            )

        n_freq_in = None if use_complex_ssm else n_freq_decomp
        return SpectralBinding(
            d_model=d_model,
            n_heads=n_heads,
            n_bands=n_bands,
            dropout=dropout,
            n_freq_in=n_freq_in,
        )

    def _build_bridge_projection(
        self,
        n_freq_decomp: int,
        d_model: int,
        use_complex_ssm: bool,
    ) -> Optional[nn.Module]:
        """Build bridge projection if decomposer output dim != binding d_model."""
        if use_complex_ssm:
            return None
        if n_freq_decomp != d_model:
            return nn.Linear(n_freq_decomp, d_model)
        return None

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
        self._validate_input(signal)

        canonical = self.canonicalizer(signal, metadata)

        # Extract CBMPC features from the raw signal (pre-SSM) if enabled.
        # These features capture modulation phase coherence that the SSM
        # would otherwise destroy. They are attached to the output metadata
        # for downstream classifiers to use alongside the SSM embedding.
        cbmpc_features = None
        if self.use_cbmpc and self.cbmpc_extractor is not None:
            # CBMPC operates on the raw audio, not the canonical spectrogram,
            # because it needs to compute its own STFT with specific parameters.
            if signal.dim() == 1:
                cbmpc_input = signal.unsqueeze(0)
            elif signal.dim() == 2:
                cbmpc_input = signal
            else:
                # 3D: (B, C, T) → mixdown to (B, T)
                cbmpc_input = signal.mean(dim=1)
            cbmpc_features = self.cbmpc_extractor(cbmpc_input)

        if self.use_complex_ssm:
            decomposed, _ = self.decomposer(canonical, h_0)
        else:
            decomposed = self.decomposer(canonical)

        bound_st, coherence = self._run_binding(decomposed, canonical)
        bound_st = self._run_attractor_learning(bound_st)
        bound_st = self._run_semantic_coherence(bound_st)

        # Attach CBMPC features to the output metadata for downstream use.
        if cbmpc_features is not None:
            bound_st.metadata['cbmpc_features'] = cbmpc_features

        return bound_st, coherence

    def _validate_input(self, signal: torch.Tensor) -> None:
        """Validate input tensor shape, dtype, and finiteness."""
        if signal.numel() == 0:
            raise ValueError("Input signal is empty (numel=0).")
        if signal.dtype != torch.float32:
            raise TypeError(
                f"Expected float32 input signal, got {signal.dtype}"
            )
        if signal.dim() < 1:
            raise ValueError(
                f"Expected 1D+ signal, got shape {signal.shape}"
            )
        if not torch.isfinite(signal).all():
            raise ValueError("Non-finite values in input signal")

    def _run_binding(
        self,
        decomposed: SpectralTensor,
        canonical: SpectralTensor,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """Run spectral binding (harmonic or standard)."""
        if self.use_harmonic_binding:
            return self._run_harmonic_binding(decomposed)
        return self._run_spectral_binding(decomposed, canonical)

    def _run_harmonic_binding(
        self,
        decomposed: SpectralTensor,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """Run HarmonicBinding on decomposed spectral tensor."""
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
        return SpectralTensor(
            amplitude=bound_amp,
            phase=phase,
            scale=decomposed.scale,
            uncertainty=decomposed.uncertainty,
            metadata=decomposed.metadata,
        ), coherence

    def _run_spectral_binding(
        self,
        decomposed: SpectralTensor,
        canonical: SpectralTensor,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """Run standard SpectralBinding on decomposed spectral tensor."""
        amp = decomposed.amplitude
        if amp.dim() == 3 and amp.shape[1] == 1:
            warnings.warn(
                "SpectralBinding received T=1 (single token). "
                "Coherence will be trivially 1.0 — phase carries no "
                "information. Provide longer signals for meaningful "
                "phase-coherence routing.",
                stacklevel=2,
            )
        if canonical.amplitude is None:
            raise ValueError(
                "canonical must have amplitude attribute for harmonic "
                "coherence detection"
            )
        bound_st, coherence = self.binding(
            decomposed,
            input_proj=self._decomp_to_bind_proj,
            canonical_phase=canonical.phase,
            canonical_amplitude=canonical.amplitude,
        )
        return bound_st, coherence

    def _run_attractor_learning(self, bound_st: SpectralTensor) -> SpectralTensor:
        """Extract learned attractors and attach metadata."""
        if self.attractor_learner is None:
            return bound_st

        try:
            attractors, _ = self.attractor_learner(bound_st)
            bound_st.metadata['attractor_count'] = len(attractors)
            bound_st.metadata['attractor_stabilities'] = [
                a.stability for a in attractors
            ]
            # Cache attractors for S4 processing
            self._last_attractors = attractors
        except Exception as e:
            warnings.warn(
                f"Attractor Learning failed: {str(e)}. "
                f"Skipping attractor extraction.",
                RuntimeWarning,
                stacklevel=2,
            )
            bound_st.metadata['attractor_count'] = 0
            bound_st.metadata['attractor_error'] = str(e)
            self._last_attractors = None
        return bound_st

    def _run_semantic_coherence(self, bound_st: SpectralTensor) -> SpectralTensor:
        """Compute optional Riemannian semantic coherence (S4)."""
        if (
            not self.use_riemannian_semantic
            or self.riemannian_semantic_coherence is None
        ):
            return bound_st

        count = bound_st.metadata.get('attractor_count', 0)
        if count <= 1:
            return bound_st

        # Retrieve cached attractors from S3
        attractors = getattr(self, '_last_attractors', None)
        if attractors is None or len(attractors) < 2:
            bound_st.metadata['semantic_coherence_error'] = "Insufficient attractors for S4"
            return bound_st

        try:
            semantic_output = self.riemannian_semantic_coherence(
                attractors=attractors,
                return_projection=True,
            )
            bound_st.metadata['semantic_coherence'] = (
                semantic_output.coherence_scores.mean().item()
            )
            bound_st.metadata['manifold_coords'] = (
                semantic_output.manifold_coords.cpu().numpy().tolist()
            )
            bound_st.metadata['semantic_coherence_max'] = (
                semantic_output.coherence_scores.max().item()
            )
        except Exception as e:
            bound_st.metadata['semantic_coherence_error'] = str(e)
        return bound_st

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
            bound_st, coherence = self._run_harmonic_binding(decomposed)
        else:
            bound_st, coherence = self.binding(
                decomposed, input_proj=self._decomp_to_bind_proj
            )
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
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"array must be np.ndarray, got {type(array).__name__}"
            )
        tensor = torch.from_numpy(array.astype(np.float32))
        return self.forward(tensor, metadata)
