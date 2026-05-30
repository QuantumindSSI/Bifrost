"""
Bifröst API Server — FastAPI backend for web dashboard.

Endpoints:
    POST /process   Process audio/image/text
    GET  /health    Health check
    GET  /demo      Interactive demo data
    GET  /metrics   Phase coherence metrics

Usage:
    from bifrost.api import start_server
    start_server(host="0.0.0.0", port=8000)
"""

from __future__ import annotations

# Handle torchaudio import error gracefully
try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None

import io
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import BifrostPipeline, HarmonicBinding, create_multimodal_pipeline
from .complex_training import ComplexBifrostTrainer, PhaseCoherenceMetrics
from .spectral_tensor import SpectralTensor


# Pydantic models for request/response
class ProcessRequest(BaseModel):
    modality: str = "audio"
    n_fft: int = 1024
    d_model: int = 128
    use_complex_ssm: bool = True


class ProcessResponse(BaseModel):
    input_shape: List[int]
    output_shape: Dict[str, List[int]]
    ssm_type: str
    coherence_ratio: float
    metadata: Dict[str, Any]


class DemoResponse(BaseModel):
    demo_type: str
    data: Dict[str, Any]
    visualizations: List[str]


class MetricsResponse(BaseModel):
    diagonal_coherence_ratio: float
    phase_smoothness: float
    complex_correlation: float
    status: str


# Create FastAPI app
app = FastAPI(
    title="Bifröst API",
    description="Bifröst — The Spectral Rainbow Bridge: Frequency-Based Cognition API for audio, image, and text processing",
    version="0.1.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0", "ssm": "ComplexSpectralDecomposer"}


@app.post("/process", response_model=ProcessResponse)
async def process_file(
    file: UploadFile = File(...),
    modality: str = Form("audio"),
    n_fft: int = Form(1024),
    d_model: int = Form(128),
) -> ProcessResponse:
    """
    Process uploaded file through Bifröst pipeline.
    
    Supports:
        - Audio: .wav, .mp3 (requires torchaudio)
        - Images: .png, .jpg
        - Text: raw text (as bytes)
    """
    try:
        contents = await file.read()
        
        if modality == "audio":
            if torchaudio is None:
                raise HTTPException(status_code=503, detail="torchaudio not available")
            # Load audio
            audio_tensor, sr = torchaudio.load(io.BytesIO(contents))
            
            pipeline = BifrostPipeline(
                n_fft_s0=n_fft,
                n_fft_s1=n_fft // 2,
                d_model=d_model,
                use_complex_ssm=True,
            )
            
            bound, coherence = pipeline(audio_tensor, {'sample_rate': sr})
            
            # Compute metrics
            diag_ratio = PhaseCoherenceMetrics.diagonal_coherence_ratio(coherence)
            
            return ProcessResponse(
                input_shape=list(audio_tensor.shape),
                output_shape={
                    'amplitude': list(bound.amplitude.shape),
                    'phase': list(bound.phase.shape),
                },
                ssm_type=pipeline.ssm_type,
                coherence_ratio=diag_ratio,
                metadata=bound.metadata,
            )
        
        elif modality == "image":
            from PIL import Image
            img = Image.open(io.BytesIO(contents)).convert('L')
            tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
            
            pipeline = create_multimodal_pipeline('tensor', n_fft=n_fft, d_model=d_model)
            bound, coherence = pipeline(tensor)
            
            return ProcessResponse(
                input_shape=list(tensor.shape),
                output_shape={'amplitude': list(bound.amplitude.shape), 'phase': list(bound.phase.shape)},
                ssm_type=pipeline.ssm_type,
                coherence_ratio=0.0,
                metadata=bound.metadata,
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo/harmonic", response_model=DemoResponse)
async def demo_harmonic(
    frequencies: str = "440,880,1320",
    duration: float = 1.0,
) -> DemoResponse:
    """
    Generate harmonic binding demo data.
    
    Args:
        frequencies: Comma-separated frequencies (Hz)
        duration: Audio duration in seconds
    """
    freqs = [float(f) for f in frequencies.split(',')]
    
    # Generate audio
    sample_rate = 16000
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    audio = torch.zeros_like(t)
    for f in freqs:
        audio += torch.sin(2 * np.pi * f * t)
        for overtone in [2, 3]:
            audio += torch.sin(2 * np.pi * f * overtone * t) * 0.3
    
    audio = audio.unsqueeze(0)
    
    # Process through harmonic binding
    harmonic = HarmonicBinding(
        d_model=128,
        n_freq=257,
        base_freq=freqs[0],
        sample_rate=sample_rate,
    )
    
    # STFT
    stft = torch.stft(audio.squeeze(0), n_fft=512, return_complex=True)
    amplitude = stft.abs().unsqueeze(0).transpose(-2, -1)
    phase = stft.angle().unsqueeze(0).transpose(-2, -1)
    
    # Interpolate to match
    if amplitude.shape[-1] != 257:
        amplitude = torch.nn.functional.interpolate(
            amplitude.transpose(-2, -1), size=257, mode='linear'
        ).transpose(-2, -1)
        phase = torch.nn.functional.interpolate(
            phase.transpose(-2, -1), size=257, mode='linear'
        ).transpose(-2, -1)
    
    bound, attn = harmonic(amplitude, phase)
    
    # Extract data for visualization
    harmonic_bins = harmonic.harmonic_grid.get_harmonic_bins().tolist()
    
    return DemoResponse(
        demo_type="harmonic_binding",
        data={
            "frequencies": freqs,
            "harmonic_bins": harmonic_bins,
            "amplitude": amplitude[0].mean(dim=0).tolist(),
            "attention_matrix": attn[0, 0].tolist(),  # First head
            "attention_std": attn.std().item(),
            "_note": "This demo uses synthetically generated audio with harmonic structure",
        },
        visualizations=["spectrum", "attention_heatmap", "harmonic_grid"],
    )


@app.get("/demo/coherence", response_model=DemoResponse)
async def demo_coherence(
    n_frames: int = 32,
    n_freq: int = 128,
) -> DemoResponse:
    """
    Generate phase coherence demo data.
    
    Compares coherent vs random phase evolution.
    """
    # Coherent phase
    phase_coherent = torch.cumsum(torch.randn(1, n_frames, n_freq) * 0.1, dim=1)
    smooth_coherent = PhaseCoherenceMetrics.phase_gradient_smoothness(phase_coherent)
    
    # Random phase
    phase_random = torch.randn(1, n_frames, n_freq)
    smooth_random = PhaseCoherenceMetrics.phase_gradient_smoothness(phase_random)
    
    return DemoResponse(
        demo_type="phase_coherence",
        data={
            "coherent": {
                "phase": phase_coherent[0].tolist(),
                "smoothness": smooth_coherent,
            },
            "random": {
                "phase": phase_random[0].tolist(),
                "smoothness": smooth_random,
            },
            "improvement_ratio": smooth_coherent / smooth_random,
            "_note": "This demo uses synthetically generated phase data for demonstration",
        },
        visualizations=["phase_evolution", "smoothness_comparison"],
    )


@app.get("/demo/multimodal", response_model=DemoResponse)
async def demo_multimodal() -> DemoResponse:
    """Show Bifröst working across all modalities."""
    modalities_data = []
    
    test_data = [
        ("audio", torch.randn(1, 8000)),
        ("text", torch.randint(0, 50000, (1, 128))),
        ("tensor", torch.randn(2, 64, 64)),
    ]
    
    for name, data in test_data:
        pipeline = create_multimodal_pipeline(name, n_fft=512, d_model=128)
        bound, coherence = pipeline(data)
        
        modalities_data.append({
            "modality": name,
            "input_shape": list(data.shape),
            "output_shape": list(bound.amplitude.shape),
            "ssm_type": pipeline.ssm_type,
        })
    
    return DemoResponse(
        demo_type="multimodal",
        data={
            "modalities": modalities_data,
            "_note": "This demo uses synthetically generated data for demonstration",
        },
        visualizations=["comparison_table"],
    )




@app.get("/spectrogram")
async def get_spectrogram(
    frequencies: str = "440,880,1320",
    duration: float = 1.0,
) -> StreamingResponse:
    """
    Generate and return spectrogram visualization.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        freqs = [float(f) for f in frequencies.split(',')]
        sample_rate = 16000
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        audio = torch.zeros_like(t)
        for f in freqs:
            audio += torch.sin(2 * np.pi * f * t)
        
        # Generate spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        spec = torch.stft(audio, n_fft=512, return_complex=True).abs()
        ax.imshow(spec.log().numpy(), aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency Bin')
        ax.set_title(f'Spectrogram: {frequencies} Hz')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
    
    except ImportError:
        raise HTTPException(status_code=500, detail="matplotlib not installed")


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
