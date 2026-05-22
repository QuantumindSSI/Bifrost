"""Generate synthetic sample data for fbc-core/sample_data/.

Creates:
    Audio:
        - mono_16khz.wav    (existing, regenerated)
        - mono_8khz.wav     (existing, regenerated)
        - stereo_44khz.wav  (existing, regenerated)
        - speech_synth.wav  (NEW) — formant-synthesized vowel
        - music_chord.wav   (NEW) — musical chord (A major)
        - noise_pink.wav    (NEW) — pink noise burst

    Images:
        - gray_image.png    (existing, regenerated)
        - rgb_image.png     (existing, regenerated)
        - rgb_large.png     (existing, regenerated)
        - spectrum_vis.png  (NEW) — spectral visualization
        - gradient_rgb.png  (NEW) — smooth color gradient

Run: python scripts/generate_samples.py
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"
SAMPLE_DIR.mkdir(exist_ok=True)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write float32 array to WAV file."""
    audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    if audio_i16.ndim == 1:
        channels, samples = 1, len(audio_i16)
        data = audio_i16
    else:
        channels, samples = audio_i16.shape
        data = audio_i16.T.flatten()

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

    print(f"  Written: {path.name}  ({channels}ch, {sr}Hz, {samples/sr:.2f}s)")


def save_png(path: Path, img: np.ndarray) -> None:
    """Save float32 [0,1] array as PNG."""
    from PIL import Image
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_u8).save(path)
    print(f"  Written: {path.name}  {img_u8.shape}")


def gen_tone(freq: float, sr: int, duration: float, amp: float = 0.7) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def gen_harmonic(f0: float, sr: int, duration: float, n_harmonics: int = 6) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    sig /= np.abs(sig).max() + 1e-8
    return (0.7 * sig).astype(np.float32)


# ── Audio samples ──────────────────────────────────────────────────────────

print("\nGenerating audio samples...")

# mono_16khz — 440 Hz tone
write_wav(SAMPLE_DIR / "mono_16khz.wav", gen_tone(440, 16000, 2.0), 16000)

# mono_8khz — 220 Hz tone
write_wav(SAMPLE_DIR / "mono_8khz.wav", gen_tone(220, 8000, 2.0), 8000)

# stereo_44khz — 440 Hz left, 880 Hz right
sr = 44100
left = gen_tone(440, sr, 2.0)
right = gen_tone(880, sr, 2.0)
stereo = np.stack([left, right], axis=0)
write_wav(SAMPLE_DIR / "stereo_44khz.wav", stereo, sr)

# speech_synth — formant vowel /a/ (F1=800Hz, F2=1200Hz, F3=2500Hz)
sr = 16000
t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
f0 = 120.0
excitation = np.zeros_like(t)
period = int(sr / f0)
excitation[::period] = 1.0
formants = [(800, 80), (1200, 120), (2500, 200)]
speech = np.zeros_like(t)
for center, bw in formants:
    speech += np.sin(2 * np.pi * center * t) * np.exp(-np.pi * bw * t % 1.0)
speech *= excitation
speech /= np.abs(speech).max() + 1e-8
speech = (0.6 * speech).astype(np.float32)
write_wav(SAMPLE_DIR / "speech_synth.wav", speech, sr)

# music_chord — A major chord (A4=440, C#5=554, E5=659)
sr = 44100
chord_freqs = [440.0, 554.37, 659.25]
chord = sum(gen_tone(f, sr, 2.0, amp=0.25) for f in chord_freqs).astype(np.float32)
write_wav(SAMPLE_DIR / "music_chord.wav", chord, sr)

# noise_pink — pink noise (1/f spectrum)
sr = 16000
n = int(sr * 2.0)
white = np.random.RandomState(42).randn(n).astype(np.float32)
freqs = np.fft.rfftfreq(n)
freqs[0] = 1e-10
power = 1.0 / np.sqrt(freqs)
pink_spec = np.fft.rfft(white) * power
pink = np.fft.irfft(pink_spec, n=n).astype(np.float32)
pink /= np.abs(pink).max() + 1e-8
pink = (0.5 * pink).astype(np.float32)
write_wav(SAMPLE_DIR / "noise_pink.wav", pink, sr)


# ── Image samples ──────────────────────────────────────────────────────────

print("\nGenerating image samples...")

rng = np.random.RandomState(0)

# gray_image — 16×16 grayscale gradient
gray = np.linspace(0, 1, 16 * 16).reshape(16, 16).astype(np.float32)
save_png(SAMPLE_DIR / "gray_image.png", gray)

# rgb_image — 16×16 RGB color blocks
rgb = np.zeros((16, 16, 3), dtype=np.float32)
rgb[:8, :8] = [1.0, 0.2, 0.2]   # red quadrant
rgb[:8, 8:] = [0.2, 1.0, 0.2]   # green quadrant
rgb[8:, :8] = [0.2, 0.2, 1.0]   # blue quadrant
rgb[8:, 8:] = [1.0, 1.0, 0.2]   # yellow quadrant
save_png(SAMPLE_DIR / "rgb_image.png", rgb)

# rgb_large — 32×32 color gradient
x = np.linspace(0, 1, 32)
y = np.linspace(0, 1, 32)
xx, yy = np.meshgrid(x, y)
rgb_large = np.stack([xx, yy, 1.0 - xx * yy], axis=-1).astype(np.float32)
save_png(SAMPLE_DIR / "rgb_large.png", rgb_large)

# spectrum_vis — 64×64 visualization of a frequency spectrum
freqs = np.linspace(0, np.pi, 64)
spectrum = np.abs(np.sin(3 * freqs[:, None]) * np.cos(2 * freqs[None, :]))
spec_norm = (spectrum / spectrum.max()).astype(np.float32)
spec_rgb = np.stack([spec_norm, 0.5 * spec_norm, 1.0 - spec_norm], axis=-1)
save_png(SAMPLE_DIR / "spectrum_vis.png", spec_rgb)

# gradient_rgb — 64×64 smooth HSV-like gradient
h = np.linspace(0, 1, 64)
v = np.linspace(0, 1, 64)
hh, vv = np.meshgrid(h, v)
r = 0.5 + 0.5 * np.sin(2 * np.pi * hh)
g = 0.5 + 0.5 * np.sin(2 * np.pi * hh + 2 * np.pi / 3)
b = 0.5 + 0.5 * np.sin(2 * np.pi * hh + 4 * np.pi / 3)
gradient = np.stack([r * vv, g * vv, b * vv], axis=-1).astype(np.float32)
save_png(SAMPLE_DIR / "gradient_rgb.png", gradient)


print(f"\nDone. All samples written to {SAMPLE_DIR}")
print(f"\nAudio: {len(list(SAMPLE_DIR.glob('*.wav')))} files")
print(f"Image: {len(list(SAMPLE_DIR.glob('*.png')))} files")
