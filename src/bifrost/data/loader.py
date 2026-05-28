"""Sample data loader for Bifröst quick-start.

Provides easy access to bundled audio/image samples for testing
and experimentation without external dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

# Sample data directory (relative to this file)
SAMPLE_DIR = Path(__file__).parent.parent.parent.parent / "sample_data"

# Registry of available samples
SAMPLES = {
    "audio": {
        "mono_16khz": "mono_16khz.wav",
        "mono_8khz": "mono_8khz.wav",
        "stereo_44khz": "stereo_44khz.wav",
        "speech_synth": "speech_synth.wav",
        "music_chord": "music_chord.wav",
        "noise_pink": "noise_pink.wav",
    },
    "image": {
        "gray": "gray_image.png",
        "rgb": "rgb_image.png",
        "rgb_large": "rgb_large.png",
        "spectrum_vis": "spectrum_vis.png",
        "gradient_rgb": "gradient_rgb.png",
    }
}


def list_samples() -> dict:
    """List all available sample files.
    
    Returns:
        Dictionary with 'audio' and 'image' keys containing sample names.
        
    Example:
        >>> samples = list_samples()
        >>> print("Audio samples:", list(samples['audio'].keys()))
        Audio samples: ['mono_16khz', 'mono_8khz', 'stereo_44khz']
    """
    return {
        "audio": list(SAMPLES["audio"].keys()),
        "image": list(SAMPLES["image"].keys()),
    }


def get_sample_path(name: str, type: Optional[str] = None) -> Path:
    """Get the filesystem path to a sample file.
    
    Args:
        name: Sample identifier (e.g., 'mono_16khz', 'rgb')
        type: Optional type hint ('audio' or 'image') to narrow search
        
    Returns:
        Path to the sample file
        
    Raises:
        FileNotFoundError: If sample doesn't exist
        
    Example:
        >>> path = get_sample_path("mono_16khz")
        >>> print(path.exists())
        True
    """
    # Search in appropriate category if specified
    categories = [type] if type else ["audio", "image"]
    
    for category in categories:
        if name in SAMPLES.get(category, {}):
            filename = SAMPLES[category][name]
            path = SAMPLE_DIR / filename
            if path.exists():
                return path
    
    # Try direct lookup as fallback
    direct_path = SAMPLE_DIR / name
    if direct_path.exists():
        return direct_path
    
    raise FileNotFoundError(
        f"Sample '{name}' not found. Available: {list_samples()}"
    )


def load_sample_audio(name: str = "mono_16khz", return_tensor: bool = True):
    """Load an audio sample.
    
    Args:
        name: Sample name ('mono_16khz', 'mono_8khz', 'stereo_44khz')
        return_tensor: If True, returns torch.Tensor; else numpy array
        
    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: shape (samples,) for mono or (samples, channels) for stereo
        - sample_rate: int (e.g., 16000, 8000, 44100)
        
    Example:
        >>> audio, sr = load_sample_audio("mono_16khz")
        >>> print(f"Loaded {len(audio)} samples at {sr} Hz")
        Loaded 16000 samples at 16000 Hz
        
        >>> audio, sr = load_sample_audio("stereo_44khz")
        >>> print(f"Stereo shape: {audio.shape}")
        Stereo shape: torch.Size([44100, 2])
    """
    import numpy as np
    import torch
    
    try:
        from scipy.io import wavfile
    except ImportError:
        raise ImportError("scipy required for audio loading: pip install scipy")
    
    path = get_sample_path(name, type="audio")
    sr, data = wavfile.read(path)
    
    # Normalize to float32 in [-1, 1] range
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    
    if return_tensor:
        data = torch.from_numpy(data)
    
    return data, sr


def load_sample_image(name: str = "rgb", return_tensor: bool = True):
    """Load an image sample.
    
    Args:
        name: Sample name ('gray', 'rgb', 'rgb_large')
        return_tensor: If True, returns torch.Tensor (C, H, W); else numpy array (H, W, C)
        
    Returns:
        Image data normalized to [0, 1] range
        - torch.Tensor: shape (C, H, W) if return_tensor=True
        - numpy.ndarray: shape (H, W, C) if return_tensor=False
        
    Example:
        >>> img = load_sample_image("rgb")
        >>> print(f"RGB image shape: {img.shape}")  # (3, H, W)
        RGB image shape: torch.Size([3, 16, 16])
        
        >>> gray = load_sample_image("gray")
        >>> print(f"Gray image shape: {gray.shape}")  # (1, H, W) or (H, W)
        Gray image shape: torch.Size([1, 16, 16])
    """
    import numpy as np
    import torch
    
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required for image loading: pip install Pillow")
    
    path = get_sample_path(name, type="image")
    img = Image.open(path)
    
    # Convert to numpy array
    data = np.array(img).astype(np.float32) / 255.0
    
    if return_tensor:
        # Convert (H, W, C) to (C, H, W) for PyTorch
        if len(data.shape) == 3:
            data = np.transpose(data, (2, 0, 1))
        else:
            # Grayscale: add channel dim
            data = data[np.newaxis, ...]
        data = torch.from_numpy(data)
    
    return data


def quick_start_pipeline(audio_name: str = "mono_16khz"):
    """Complete quick-start: load sample + run through Bifröst pipeline.
    
    Returns:
        Tuple of (canonical, decomposed, bound, attention_weights)
    
    This is the fastest way to verify your installation and see Bifröst in action.
    
    Args:
        audio_name: Which audio sample to use
        
    Returns:
        Tuple of (canonical, decomposed, bound, attention_weights)
        
    Example:
        >>> from bifrost.data import quick_start_pipeline
        >>> canonical, decomposed, bound, attn = quick_start_pipeline("mono_16khz")
        >>> print(f"Attention shape: {attn.shape}")
        Pipeline complete. Spectral tensor shape: torch.Size([1, 513, 8])
    """
    from bifrost.canonicalizer import SpectralCanonicalizer
    from bifrost.decomposer import SpectralDecomposer
    from bifrost.resonance_attention import SpectralBinding
    
    # Load sample
    audio, sr = load_sample_audio(audio_name)
    
    # Add batch dimension if needed
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, samples)
    elif audio.dim() == 2 and audio.shape[1] <= 2:  # stereo
        audio = audio.unsqueeze(0)  # (1, samples, channels)
    
    # Build pipeline
    n_fft = 1024
    n_freq = n_fft // 2 + 1
    d_model = 128
    
    canonicalizer = SpectralCanonicalizer(n_fft=n_fft)
    decomposer = SpectralDecomposer(n_fft=n_fft, n_scales=4, d_model=n_freq)
    binding = SpectralBinding(d_model=d_model, n_heads=4, n_bands=8, dropout=0.0)
    
    # Run pipeline
    canonicalizer.eval()
    decomposer.eval()
    binding.eval()
    
    with torch.no_grad():
        canonical = canonicalizer(audio, metadata={"sample_rate": float(sr)})
        decomposed = decomposer(canonical)
        bound, coherence = binding(decomposed)
    
    print(f"✓ Pipeline complete. Spectral tensor shape: {bound.amplitude.shape}")
    return canonical, decomposed, bound, coherence
