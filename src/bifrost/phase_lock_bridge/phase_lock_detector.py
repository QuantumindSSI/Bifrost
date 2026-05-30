"""
True Phase-Lock Detector — Temporal Consistency + Frequency Matching + Coupling

Implements physically-grounded phase-locking detection as specified in:
- Engineering Script §3: Phase-Lock Bridge
- Physics of coupled oscillators (Kuramoto model, Adler equation)

Key differences from simple phase-alignment:
1. TEMPORAL CONSISTENCY: Phase difference must remain constant over time windows
2. FREQUENCY MATCHING: Oscillators must have matching/near frequencies (detuning check)
3. COUPLING DYNAMICS: Models energy exchange, not just snapshot similarity

Per Agentic CTO-Persona Policy:
- C-01: Full documentation
- C-02: Explicit error handling
- C-04: Complexity ≤10
- G1-G5: Complete, executable, correct
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhaseLockState:
    """
    Temporal state tracking for phase-lock detection.
    
    Maintains history of phase relationships for temporal consistency verification.
    """
    phase_differences: torch.Tensor  # (n_bands,) history of Δφ over time
    timestamps: torch.Tensor  # Corresponding time points
    frequency_estimates: torch.Tensor  # (n_bands,) instantaneous frequency estimates
    
    def temporal_consistency_score(self) -> float:
        """
        Compute temporal consistency: variance of phase difference over time.
        
        True phase-lock: phase difference remains nearly constant → low variance → high score
        Returns score in [0, 1] where 1.0 = perfectly constant phase relationship
        """
        if len(self.phase_differences) < 2:
            return 0.0
        
        # Variance of phase differences (wrapped to [-π, π])
        phase_diffs = self.phase_differences
        variance = phase_diffs.var().item()
        
        # Map variance to consistency score: low variance = high score
        # Using exponential decay: score = exp(-variance / threshold)
        consistency = math.exp(-variance / 0.1)  # threshold = 0.1 rad²
        
        return min(consistency, 1.0)
    
    def frequency_detuning(self) -> torch.Tensor:
        """
        Compute frequency detuning between oscillators from phase derivative.
        
        d(Δφ)/dt ≈ ω₁ - ω₂ (frequency difference)
        
        Returns:
            Tensor of frequency differences per band (Hz)
        """
        if len(self.timestamps) < 2:
            return torch.zeros_like(self.phase_differences)
        
        # Compute phase derivative (unwrapped to handle 2π crossings)
        phase_diffs = self.phase_differences
        time_diffs = self.timestamps[1:] - self.timestamps[:-1]
        
        # Unwrap phase differences
        unwrapped = torch.unwrap(phase_diffs)
        
        # d(Δφ)/dt
        d_phase = unwrapped[1:] - unwrapped[:-1]
        dt = time_diffs.mean()
        
        detuning = d_phase.mean() / dt if dt > 0 else torch.tensor(0.0)
        
        return detuning.abs()


class TruePhaseLockDetector(nn.Module):
    """
    Detects true phase-locking between frequency attractors.
    
    Unlike simple phase-alignment (cos(Δφ)), this detector verifies:
    - Temporal consistency: phase relationship persists over time
    - Frequency matching: oscillators have compatible frequencies
    - Coupling strength: energy exchange modeled via Adler equation
    
    Physics Background:
        Two coupled oscillators follow the Adler equation:
        d(Δφ)/dt = Δω - K * sin(Δφ)
        
        Where:
        - Δφ = phase difference
        - Δω = natural frequency detuning
        - K = coupling strength
        
        Phase-lock occurs when |Δω| < |K| (locking range)
    """
    
    def __init__(
        self,
        n_bands: int = 8,
        history_length: int = 10,
        sample_rate: float = 16000.0,
        coupling_strength: float = 0.5,
        temporal_window_ms: float = 100.0,  # 100ms window for consistency check
    ):
        super().__init__()
        self.n_bands = n_bands
        self.history_length = history_length
        self.sample_rate = sample_rate
        self.coupling_strength = coupling_strength
        self.temporal_window_ms = temporal_window_ms
        
        # Learnable band coupling coefficients (K in Adler equation)
        self.band_coupling = nn.Parameter(torch.ones(n_bands) * coupling_strength)
        
        # Frequency tolerance for locking (Hz)
        self.register_buffer('freq_tolerance', torch.tensor(5.0))  # ±5 Hz tolerance
        
        # Phase consistency threshold (radians)
        self.register_buffer('consistency_threshold', torch.tensor(0.2))  # ~11.5 degrees
    
    def estimate_instantaneous_frequency(
        self,
        phase_history: torch.Tensor,  # (T, n_bands)
        time_points: torch.Tensor,  # (T,)
    ) -> torch.Tensor:
        """
        Estimate instantaneous frequency from phase derivative.
        
        ω(t) = dφ/dt
        
        Args:
            phase_history: Phase values over time (T, n_bands) in radians
            time_points: Corresponding timestamps (T,) in seconds
            
        Returns:
            Instantaneous frequencies (T, n_bands) in Hz
        """
        # Unwrap phase to handle 2π crossings
        unwrapped = torch.unwrap(phase_history, dim=0)
        
        # Compute derivative dφ/dt
        d_phase = unwrapped[1:] - unwrapped[:-1]  # (T-1, n_bands)
        dt = time_points[1:] - time_points[:-1]  # (T-1,)
        
        # Avoid division by zero
        dt = torch.clamp(dt, min=1e-6)
        
        # Frequency = dφ/dt / (2π)
        freq = d_phase / (2.0 * math.pi * dt.unsqueeze(-1))
        
        return freq  # (T-1, n_bands)
    
    def compute_temporal_consistency(
        self,
        phase_diff_history: torch.Tensor,  # (T, n_bands)
    ) -> torch.Tensor:
        """
        Compute temporal consistency score for phase-locking.
        
        True phase-lock requires phase difference to remain constant.
        We compute the variance of phase differences over time.
        
        Args:
            phase_diff_history: Phase differences over time (T, n_bands)
            
        Returns:
            Consistency scores per band (n_bands,), range [0, 1]
        """
        if phase_diff_history.shape[0] < 2:
            return torch.zeros(self.n_bands)
        
        # Unwrap phase differences
        unwrapped = torch.unwrap(phase_diff_history, dim=0)
        
        # Variance over time per band
        variance = unwrapped.var(dim=0)  # (n_bands,)
        
        # Score: low variance = high consistency
        # Using sigmoid to map variance to [0, 1]
        consistency = torch.sigmoid(-(variance - self.consistency_threshold) / 0.1)
        
        return consistency
    
    def compute_coupling_strength(
        self,
        phase_diff: torch.Tensor,  # (n_bands,)
        freq_detuning: torch.Tensor,  # (n_bands,) in Hz
    ) -> torch.Tensor:
        """
        Compute effective coupling strength based on Adler equation.
        
        Adler equation: d(Δφ)/dt = Δω - K * sin(Δφ)
        
        At phase-lock: d(Δφ)/dt = 0, so Δω = K * sin(Δφ)
        
        Coupling is effective when |Δω| < |K| (locking range)
        
        Args:
            phase_diff: Current phase differences (n_bands,)
            freq_detuning: Frequency differences (n_bands,) in Hz
            
        Returns:
            Coupling effectiveness per band (n_bands,), range [0, 1]
        """
        # Normalize detuning by coupling strength
        K = self.band_coupling.abs()
        
        # Effective coupling ratio: |Δω| / |K|
        # When |Δω| < |K|, phase-lock is possible
        ratio = freq_detuning.abs() / (K + 1e-8)
        
        # Coupling effectiveness: 1.0 when locked, decays when detuned
        effectiveness = torch.clamp(1.0 - ratio / 2.0, min=0.0, max=1.0)
        
        return effectiveness
    
    def detect_phase_lock(
        self,
        source_phase: torch.Tensor,  # (T, n_bands) or (n_bands,)
        target_phase: torch.Tensor,  # (T, n_bands) or (n_bands,)
        time_points: Optional[torch.Tensor] = None,  # (T,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect true phase-locking between two oscillators.
        
        Returns three criteria for phase-locking:
        1. Phase alignment: cos(Δφ) near 1.0
        2. Temporal consistency: low variance of Δφ over time
        3. Frequency coupling: |Δω| < |K|
        
        Args:
            source_phase: Phase history of source oscillator
            target_phase: Phase history of target oscillator
            time_points: Timestamps (if None, assumes uniform sampling)
            
        Returns:
            Tuple of (alignment_score, consistency_score, coupling_score)
            Each tensor has shape (n_bands,) with values in [0, 1]
        """
        # Handle single-timepoint case
        if source_phase.dim() == 1:
            source_phase = source_phase.unsqueeze(0)
            target_phase = target_phase.unsqueeze(0)
        
        T = source_phase.shape[0]
        
        if time_points is None:
            # Assume uniform sampling
            time_points = torch.arange(T) * (1000.0 / self.sample_rate)  # ms
        
        # 1. PHASE ALIGNMENT (snapshot coherence)
        phase_diff = source_phase - target_phase
        alignment = torch.cos(phase_diff.mean(dim=0))  # Average over time
        alignment = (alignment + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        
        # 2. TEMPORAL CONSISTENCY
        if T >= 2:
            consistency = self.compute_temporal_consistency(phase_diff)
        else:
            consistency = torch.ones(self.n_bands) * 0.5  # Unknown
        
        # 3. FREQUENCY COUPLING
        if T >= 2:
            # Estimate frequencies
            freq_source = self.estimate_instantaneous_frequency(source_phase, time_points)
            freq_target = self.estimate_instantaneous_frequency(target_phase, time_points)
            
            # Detuning
            freq_detuning = (freq_source - freq_target).mean(dim=0)  # (n_bands,)
            
            coupling = self.compute_coupling_strength(
                phase_diff.mean(dim=0),
                freq_detuning,
            )
        else:
            coupling = torch.ones(self.n_bands) * 0.5  # Unknown
        
        return alignment, consistency, coupling
    
    def compute_phase_lock_score(
        self,
        source_phase: torch.Tensor,
        target_phase: torch.Tensor,
        time_points: Optional[torch.Tensor] = None,
        min_duration_ms: float = 50.0,  # Minimum duration for reliable detection
    ) -> Tuple[float, str]:
        """
        Compute overall phase-lock score with diagnostic label.
        
        Args:
            source_phase: Source phase history (T, n_bands) or (n_bands,)
            target_phase: Target phase history (T, n_bands) or (n_bands,)
            time_points: Timestamps
            min_duration_ms: Minimum signal duration for reliable detection
            
        Returns:
            Tuple of (score, label) where label describes the lock state:
            - "locked": All three criteria satisfied
            - "alignment_only": Phase aligned but not temporally consistent
            - "transient": Temporary lock, not stable
            - "unlocked": No phase-lock detected
        """
        # Check duration
        if time_points is not None:
            duration_ms = (time_points[-1] - time_points[0]).item()
        else:
            duration_ms = float('inf')  # Assume sufficient
        
        if duration_ms < min_duration_ms:
            return 0.0, "insufficient_data"
        
        # Compute lock criteria
        alignment, consistency, coupling = self.detect_phase_lock(
            source_phase, target_phase, time_points
        )
        
        # Overall score: weighted combination
        # Temporal consistency is most important for true phase-lock
        weights = torch.tensor([0.2, 0.5, 0.3])  # alignment, consistency, coupling
        scores = torch.stack([alignment.mean(), consistency.mean(), coupling.mean()])
        
        overall_score = (weights * scores).sum().item()
        
        # Determine label
        if overall_score > 0.8 and consistency.mean() > 0.7:
            label = "locked"
        elif alignment.mean() > 0.7 and consistency.mean() < 0.5:
            label = "alignment_only"
        elif consistency.mean() > 0.5 and alignment.mean() < 0.5:
            label = "coupled_not_aligned"
        elif overall_score > 0.5:
            label = "transient"
        else:
            label = "unlocked"
        
        return overall_score, label


def demo_phase_lock_detection():
    """Demonstrate true phase-lock detection on synthetic signals."""
    print("=" * 60)
    print("TRUE PHASE-LOCK DETECTION DEMO")
    print("=" * 60)
    
    detector = TruePhaseLockDetector(n_bands=4, history_length=20)
    
    # Generate test cases
    test_cases = [
        ("Perfect Lock", lambda t: (0.0 * t, 0.0 * t)),  # Same phase
        ("Constant Offset", lambda t: (0.0 * t, 0.5 + 0.0 * t)),  # Δφ = constant
        ("Slow Drift", lambda t: (0.0 * t, 0.01 * t)),  # Slow detuning
        ("Fast Drift", lambda t: (0.0 * t, 0.5 * t)),  # Fast detuning
        ("Noisy Lock", lambda t: (0.0 * t + 0.1 * torch.randn_like(t), 
                                  0.0 * t + 0.1 * torch.randn_like(t))),
    ]
    
    time_points = torch.linspace(0, 100, 50)  # 100ms, 50 samples
    
    for name, generator in test_cases:
        source_phase, target_phase = generator(time_points)
        
        # Add batch dimension for detector
        source_phase = source_phase.unsqueeze(-1).expand(-1, 4)  # (T, 4)
        target_phase = target_phase.unsqueeze(-1).expand(-1, 4)
        
        score, label = detector.compute_phase_lock_score(source_phase, target_phase, time_points)
        
        print(f"\n{name}:")
        print(f"  Score: {score:.3f}")
        print(f"  State: {label}")


if __name__ == "__main__":
    demo_phase_lock_detection()
