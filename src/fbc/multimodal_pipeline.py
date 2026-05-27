"""
MultiModalSpectralPipeline — unified FBC pipeline for audio, image, text, and tensor.

This pipeline automatically detects the input modality and routes through the
appropriate processing path while maintaining the core FBC architecture:
    Ingest → Canonicalize → Decompose → Bind

Usage:
    from fbc.multimodal_pipeline import MultiModalSpectralPipeline, Modality

    # Audio (1D temporal)
    pipeline = MultiModalSpectralPipeline(modality=Modality.AUDIO)
    bound, coherence = pipeline(audio_tensor)

    # Image (2D spatial)
    pipeline = MultiModalSpectralPipeline(modality=Modality.IMAGE, use_2d_fft=True)
    bound, coherence = pipeline(image_tensor)

    # Text (sequence embeddings)
    pipeline = MultiModalSpectralPipeline(modality=Modality.TEXT)
    bound, coherence = pipeline(text_tokens)

    # Tensor (direct)
    pipeline = MultiModalSpectralPipeline(modality=Modality.TENSOR)
    bound, coherence = pipeline(tensor)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_tensor import SpectralTensor
from .canonicalizer import SpectralCanonicalizer
from .s1_decomposer.decomposer import SpectralDecomposer
from .resonance_attention.binding import SpectralBinding


class Modality(Enum):
    """Supported data modalities for FBC processing."""
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    TENSOR = "tensor"


class TextSpectralEncoder(nn.Module):
    """
    Encode text sequences into spectral-compatible format.

    Strategy: Treat token embeddings as a 1D signal and apply 1D FFT
    to get "semantic frequencies". This captures patterns in embedding space.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 256,
        n_fft: int = 512,
        max_length: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_fft = n_fft
        self.max_length = max_length

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)

        # Project embeddings to frequency bins
        self.to_freq = nn.Linear(embed_dim, n_fft // 2 + 1)

    def forward(
        self,
        tokens: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpectralTensor:
        """
        Convert text tokens to SpectralTensor.

        Args:
            tokens: (B, seq_len) token indices
            metadata: Optional metadata

        Returns:
            SpectralTensor with shape (B, n_freq)
        """
        B, seq_len = tokens.shape
        seq_len = min(seq_len, self.max_length)

        # Embed tokens
        x = self.embedding(tokens[:, :seq_len])  # (B, seq_len, embed_dim)

        # Add position encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Treat as 1D signal per embedding dimension, then FFT
        # Average across embedding dimension to get 1D signal
        signal = x.mean(dim=-1)  # (B, seq_len)

        # Pad/truncate to n_fft
        if seq_len < self.n_fft:
            signal = F.pad(signal, (0, self.n_fft - seq_len))
        else:
            # Truncate or use STFT - here just truncate for simplicity
            signal = signal[:, :self.n_fft]

        # 1D FFT to get spectral representation
        spectrum = torch.fft.rfft(signal, n=self.n_fft, dim=-1)
        amplitude = spectrum.abs()
        phase = spectrum.angle()

        # Normalize
        amp_max = amplitude.amax(dim=-1, keepdim=True).clamp(min=1e-8)
        amplitude = amplitude / amp_max

        # Create scale (treat as "semantic frequency")
        n_freq = amplitude.shape[-1]
        scale = torch.linspace(0.0, 1.0, n_freq, device=amplitude.device)
        scale = scale.expand_as(amplitude)

        uncertainty = torch.full_like(amplitude, 1.0)

        return SpectralTensor(
            amplitude=amplitude,
            phase=phase,
            scale=scale,
            uncertainty=uncertainty,
            metadata={
                **(metadata or {}),
                "stage": "canonicalize",
                "modality": "text",
                "seq_len": seq_len,
                "n_fft": self.n_fft,
            },
        )


class ImageSpectralDecomposer(nn.Module):
    """
    Decompose 2D spectral data (from 2D FFT) through learned SSM.

    Strategy: Treat spatial frequency bins as "temporal frames".
    Radial frequency bands become the sequence dimension.
    """

    def __init__(
        self,
        n_fft: int = 512,
        d_model: int = 128,
        n_scales: int = 6,
        n_frames: int = 32,
        use_mamba: bool = True,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.d_model = d_model
        self.n_freq = n_fft // 2 + 1
        self.n_frames = n_frames

        # Import here to avoid circular dependency
        from .s1_decomposer.decomposer import MambaBlock
        from .s1_decomposer.selective_scan import SelectiveScan

        # Project radial frequency to d_model
        self.input_proj = nn.Linear(self.n_freq, d_model)
        self.input_proj_phase = nn.Linear(self.n_freq, d_model)

        # SSM for amplitude and phase streams
        if use_mamba:
            self.ssm = MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.ssm_phase = MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = nn.Sequential(SelectiveScan(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.ssm_phase = nn.Sequential(SelectiveScan(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))

        self.norm = nn.LayerNorm(d_model)
        self.norm_phase = nn.LayerNorm(d_model)

        # Output projections
        self.output_proj = nn.Linear(d_model, self.n_freq)
        self.output_proj_phase = nn.Linear(d_model, self.n_freq)

    def forward(self, st: SpectralTensor) -> SpectralTensor:
        """
        Process 2D spectral data.

        Args:
            st: SpectralTensor from 2D FFT (B, n_freq)

        Returns:
            SpectralTensor with learned temporal structure (B, T, n_freq)
        """
        amp = st.amplitude
        phase = st.phase

        if amp.dim() == 1:
            amp = amp.unsqueeze(0)
            phase = phase.unsqueeze(0)
        B = amp.shape[0]

        # Replicate to create "temporal" dimension for SSM processing
        # For images, we treat radial frequency bins as positions
        # and create pseudo-temporal frames through position encoding variations
        amp_expanded = amp.unsqueeze(1).expand(B, self.n_frames, -1)  # (B, T, n_freq)
        phase_expanded = phase.unsqueeze(1).expand(B, self.n_frames, -1)

        # Add positional variation to create meaningful sequence
        position_scale = torch.linspace(0.5, 1.5, self.n_frames, device=amp.device)
        amp_expanded = amp_expanded * position_scale.view(1, -1, 1)

        # Project
        h = self.input_proj(amp_expanded)
        h_phase = self.input_proj_phase(phase_expanded)

        # SSM processing
        h = self.ssm(h)
        h = self.norm(h)

        h_phase = self.ssm_phase(h_phase)
        h_phase = self.norm_phase(h_phase)

        # Output
        out_amp = self.output_proj(h).abs()
        out_phase = self.output_proj_phase(h_phase)

        # Scale and uncertainty
        scale = st.scale[:self.n_freq].view(1, 1, -1).expand(B, self.n_frames, -1)
        uncertainty = torch.full_like(out_amp, 0.5)

        return SpectralTensor(
            amplitude=out_amp,
            phase=out_phase,
            scale=scale,
            uncertainty=uncertainty,
            metadata={
                **st.metadata,
                "stage": "S1",
                "modality": "image",
                "n_frames": self.n_frames,
            },
        )


class TensorSpectralAdapter(nn.Module):
    """
    Adapter for arbitrary tensors to spectral pipeline.

    Flattens tensor and treats as 1D signal, then applies FFT.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        max_elements: int = 10000,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.max_elements = max_elements

    def forward(
        self,
        tensor: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpectralTensor:
        """
        Convert arbitrary tensor to SpectralTensor.

        Args:
            tensor: Any shape tensor
            metadata: Optional metadata

        Returns:
            SpectralTensor
        """
        B = tensor.shape[0] if tensor.ndim > 0 else 1

        # Flatten each sample
        flat = tensor.view(B, -1) if tensor.ndim > 0 else tensor.view(1, -1)

        # Truncate or pad to max_elements
        n_elements = min(flat.shape[-1], self.max_elements)
        flat = flat[:, :n_elements]

        # Pad to n_fft if needed
        if n_elements < self.n_fft:
            flat = F.pad(flat, (0, self.n_fft - n_elements))
        else:
            # Use first n_fft elements or downsample
            flat = flat[:, :self.n_fft]

        # FFT
        spectrum = torch.fft.rfft(flat, n=self.n_fft, dim=-1)
        amplitude = spectrum.abs()
        phase = spectrum.angle()

        # Normalize
        amp_max = amplitude.amax(dim=-1, keepdim=True).clamp(min=1e-8)
        amplitude = amplitude / amp_max

        n_freq = amplitude.shape[-1]
        scale = torch.linspace(0.0, 1.0, n_freq, device=amplitude.device)
        scale = scale.expand_as(amplitude)

        uncertainty = torch.full_like(amplitude, 1.0)

        return SpectralTensor(
            amplitude=amplitude,
            phase=phase,
            scale=scale,
            uncertainty=uncertainty,
            metadata={
                **(metadata or {}),
                "stage": "canonicalize",
                "modality": "tensor",
                "original_shape": list(tensor.shape),
                "n_elements": n_elements,
                "n_fft": self.n_fft,
            },
        )


class MultiModalSpectralPipeline(nn.Module):
    """
    Unified multimodal FBC pipeline supporting audio, image, text, and tensor.

    Automatically routes each modality through the appropriate processing path
    while maintaining consistent output: SpectralTensor → coherence weights.

    Parameters
    ----------
    modality : Modality
        The data modality this pipeline handles
    n_fft : int
        FFT size (audio: 1024, image: 512, text: 512, tensor: 1024)
    d_model : int
        Hidden dimension for decomposer and binding
    n_heads : int
        Number of attention heads
    n_bands : int
        Number of spectral bands for coherence
    use_2d_fft : bool
        For images: use 2D FFT instead of 1D
    use_mamba : bool
        Use MambaBlock SSM when available
    """

    def __init__(
        self,
        modality: Modality = Modality.AUDIO,
        n_fft: int = 1024,
        d_model: int = 128,
        n_heads: int = 4,
        n_bands: int = 8,
        use_2d_fft: bool = False,
        use_mamba: bool = True,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.n_fft = n_fft
        self.d_model = d_model
        self.use_2d_fft = use_2d_fft

        # Modality-specific canonicalizer/decomposer
        if modality == Modality.AUDIO:
            self.canonicalizer = SpectralCanonicalizer(
                n_fft=n_fft,
                preserve_frames=True,
                use_2d_fft=False,
            )
            self.decomposer = SpectralDecomposer(
                n_fft=n_fft // 2,  # S1 uses smaller FFT
                d_model=d_model,
                n_frames=32,
                use_mamba=use_mamba,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
            )

        elif modality == Modality.IMAGE:
            self.canonicalizer = SpectralCanonicalizer(
                n_fft=n_fft,
                use_2d_fft=True,
                preserve_frames=False,
            )
            self.decomposer = ImageSpectralDecomposer(
                n_fft=n_fft,
                d_model=d_model,
                n_frames=32,
                use_mamba=use_mamba,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
            )

        elif modality == Modality.TEXT:
            self.canonicalizer = TextSpectralEncoder(
                vocab_size=50000,
                embed_dim=256,
                n_fft=n_fft,
            )
            self.decomposer = SpectralDecomposer(
                n_fft=n_fft,
                d_model=d_model,
                n_frames=32,
                use_mamba=use_mamba,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
            )

        elif modality == Modality.TENSOR:
            self.canonicalizer = TensorSpectralAdapter(
                n_fft=n_fft,
            )
            self.decomposer = SpectralDecomposer(
                n_fft=n_fft,
                d_model=d_model,
                n_frames=32,
                use_mamba=use_mamba,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
            )

        # Binding stage - needs to know input n_freq from decomposer
        # For audio: decomposer uses n_fft//2, so n_freq = n_fft//2 // 2 + 1 = n_fft//4 + 1
        # For others: decomposer uses n_fft directly, so n_freq = n_fft//2 + 1
        if modality == Modality.AUDIO:
            binding_n_freq = n_fft // 4 + 1  # S1 uses n_fft//2, then n_freq = (n_fft//2)//2 + 1
        else:
            binding_n_freq = n_fft // 2 + 1

        self.binding = SpectralBinding(
            d_model=d_model,
            n_heads=n_heads,
            n_bands=n_bands,
            dropout=0.1,
            n_freq_in=binding_n_freq,
        )

    def forward(
        self,
        data: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SpectralTensor, torch.Tensor]:
        """
        Process data through the full multimodal pipeline.

        Args:
            data: Input tensor (modality-specific shape)
            metadata: Optional metadata dict

        Returns:
            (bound_st, coherence): Final SpectralTensor and attention coherence
        """
        # Canonicalize (modality-specific)
        canonical = self.canonicalizer(data, metadata)

        # Decompose (SSM processing)
        decomposed = self.decomposer(canonical)

        # Bind (resonance attention)
        bound, coherence = self.binding(decomposed)

        return bound, coherence

    @property
    def ssm_type(self) -> str:
        """Return the SSM type being used."""
        if hasattr(self.decomposer, 'ssm_type'):
            return self.decomposer.ssm_type
        return "Unknown"


def create_multimodal_pipeline(
    modality: str = "audio",
    **kwargs,
) -> MultiModalSpectralPipeline:
    """
    Factory function to create a multimodal pipeline.

    Args:
        modality: One of "audio", "image", "text", "tensor"
        **kwargs: Additional arguments passed to pipeline

    Returns:
        Configured MultiModalSpectralPipeline
    """
    mod_map = {
        "audio": Modality.AUDIO,
        "image": Modality.IMAGE,
        "text": Modality.TEXT,
        "tensor": Modality.TENSOR,
    }

    if modality not in mod_map:
        raise ValueError(f"Unknown modality: {modality}. Choose from {list(mod_map.keys())}")

    return MultiModalSpectralPipeline(modality=mod_map[modality], **kwargs)
