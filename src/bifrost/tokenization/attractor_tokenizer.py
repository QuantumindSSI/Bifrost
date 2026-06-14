"""
Phase 2: Attractor Tokenization

Converts continuous spectral outputs → discrete attractor vocabulary (65K tokens)
Implements VQ-VAE style learned codebook for structural events.
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from bifrost.spectral_tensor import SpectralTensor


@dataclass
class AttractorFeatures:
    """Represents a single structural attractor"""
    time_idx: int          # Frame index
    frequency: float       # Hz
    amplitude: float       # 0-1
    phase: float           # 0-2π
    coherence: float       # 0-1
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to feature vector for tokenizer"""
        return torch.tensor([
            self.time_idx / 1000.0,  # Normalize to ~0-1
            self.amplitude,
            (self.phase % (2 * math.pi)) / (2 * math.pi),
            self.coherence,
        ], dtype=torch.float32)


@dataclass
class TokenSequence:
    """Sequence of discrete attractor tokens"""
    tokens: List[int]
    times: List[int]
    confidences: List[float]  # VQ confidence (distance to nearest centroid)
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensor format"""
        return (
            torch.tensor(self.tokens, dtype=torch.long),
            torch.tensor(self.times, dtype=torch.float32),
        )


def detect_attractors(coherence: torch.Tensor, spectrum: SpectralTensor, 
                      persistence_threshold: float = 0.8) -> List[AttractorFeatures]:
    """
    Identify stable frequency attractors from spectral coherence.
    
    Attractors = local maxima in coherence with high persistence (low collapse rate)
    
    Args:
        coherence: [time, freq] coherence matrix
        spectrum: SpectralTensor with amplitude/phase/frequency
        persistence_threshold: Min persistence to accept attractor (0-1)
    
    Returns:
        List of AttractorFeatures representing stable peaks
    """
    attractors = []
    
    time_len, freq_len = coherence.shape
    window_size = 8  # Look at ±4 frames
    
    # Find local maxima in coherence
    for t in range(window_size, time_len - window_size):
        for f in range(window_size, freq_len - window_size):
            # Check if local maximum
            center_val = coherence[t, f]
            
            # Compare to neighbors
            window = coherence[t-window_size:t+window_size+1, f-window_size:f+window_size+1]
            is_max = center_val >= window.max() - 1e-6
            
            if not is_max or center_val < 0.1:  # Threshold on coherence
                continue
            
            # Check persistence: does this peak stay stable over time?
            persistence = 0.0
            for dt in range(-2, 3):
                if 0 <= t + dt < time_len:
                    if coherence[t + dt, f] > center_val * 0.8:
                        persistence += 1
            persistence /= 5.0  # Normalize to 0-1
            
            if persistence < persistence_threshold:
                continue
            
            # Extract attractor features
            attractor = AttractorFeatures(
                time_idx=t,
                frequency=spectrum.frequency[f] if hasattr(spectrum, 'frequency') else float(f),
                amplitude=spectrum.amplitude[t, f].item() if hasattr(spectrum, 'amplitude') else 0.5,
                phase=spectrum.phase[t, f].item() if hasattr(spectrum, 'phase') else 0.0,
                coherence=center_val.item(),
            )
            
            attractors.append(attractor)
    
    return attractors


class AttractorTokenizer(nn.Module):
    """
    Vector quantized codebook for attractors.
    Learned mapping from continuous attractor features → discrete tokens.
    
    Similar to VQ-VAE but optimized for spectral structure.
    """
    
    def __init__(self, vocab_size: int = 65536, feature_dim: int = 4, 
                 latent_dim: int = 256, num_clusters: int = 1000):
        """
        Args:
            vocab_size: Size of token vocabulary
            feature_dim: Input attractor feature dimension (time, amp, phase, coherence)
            latent_dim: Latent representation dimension
            num_clusters: Number of K-means clusters for initialization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        
        # Encoder: attractor features → latent
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        
        # Learnable codebook (VQ codebook)
        self.register_buffer(
            'codebook',
            torch.randn(vocab_size, latent_dim)
        )
        self.codebook_initialized = False
        
        # Exponential moving average for codebook updates
        self.register_buffer('cluster_size', torch.zeros(vocab_size))
        self.register_buffer('w_avg', torch.zeros(vocab_size, latent_dim))
        
        # Decoder: latent → reconstructed features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        
        self.commitment_cost = 0.25  # VQ-VAE commitment loss weight
    
    def initialize_codebook(self, attractors: List[List[AttractorFeatures]]):
        """
        Initialize codebook using K-means on a sample of attractors.
        
        Args:
            attractors: List of [List of AttractorFeatures] from multiple samples
        """
        logger = __import__('logging').getLogger(__name__)
        logger.info("Initializing codebook with K-means...")
        
        # Flatten all attractors
        all_features = []
        for sample in attractors:
            for attr in sample:
                feature_vec = attr.to_tensor()
                all_features.append(feature_vec)
        
        if len(all_features) == 0:
            logger.warning("No attractors for initialization, using random")
            return
        
        all_features = torch.stack(all_features)  # [N, 4]
        
        # Encode to latent space
        with torch.no_grad():
            latents = self.encoder(all_features)  # [N, latent_dim]
        
        # Simple K-means initialization
        # (In production: use sklearn or torch_kmeans)
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=min(self.vocab_size, len(all_features)), 
                       n_init=10, random_state=42)
        kmeans.fit(latents.cpu().numpy())
        
        # Copy centers to codebook
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        self.codebook[:centers.shape[0]] = centers
        
        self.codebook_initialized = True
        logger.info(f"Codebook initialized with {centers.shape[0]} clusters")
    
    def encode(self, attractors: List[AttractorFeatures], 
              return_confidence: bool = True) -> TokenSequence:
        """
        Encode attractor sequence to token sequence.
        
        Args:
            attractors: List of AttractorFeatures
            return_confidence: Include VQ confidence scores
        
        Returns:
            TokenSequence with tokens, times, and confidences
        """
        if len(attractors) == 0:
            return TokenSequence(tokens=[], times=[], confidences=[])
        
        # Convert to tensor
        features = torch.stack([a.to_tensor() for a in attractors])  # [N, 4]
        times = torch.tensor([a.time_idx for a in attractors])
        
        # Encode to latent
        with torch.no_grad():
            latents = self.encoder(features)  # [N, latent_dim]
        
        # Vector quantize: find nearest codebook entry
        tokens = []
        confidences = []
        
        with torch.no_grad():
            for i in range(latents.shape[0]):
                # L2 distances to all codebook entries
                distances = torch.norm(self.codebook - latents[i:i+1], dim=1)  # [vocab_size]
                
                # Nearest token
                token_id = distances.argmin().item()
                tokens.append(token_id)
                
                # Confidence: inverse distance (higher = more confident)
                if return_confidence:
                    min_dist = distances.min().item()
                    confidence = 1.0 / (1.0 + min_dist)
                    confidences.append(confidence)
        
        return TokenSequence(
            tokens=tokens,
            times=times.tolist(),
            confidences=confidences,
        )
    
    def decode(self, tokens: List[int]) -> List[AttractorFeatures]:
        """
        Decode token sequence back to attractor features.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            List of reconstructed AttractorFeatures
        """
        if len(tokens) == 0:
            return []
        
        token_indices = torch.tensor(tokens, dtype=torch.long)
        
        with torch.no_grad():
            # Get codebook entries
            latents = self.codebook[token_indices]  # [N, latent_dim]
            
            # Decode
            features = self.decoder(latents)  # [N, 4]
        
        # Convert back to AttractorFeatures
        attractors = []
        for i in range(features.shape[0]):
            f = features[i]
            attractors.append(AttractorFeatures(
                time_idx=int(f[0].item() * 1000),
                amplitude=torch.sigmoid(f[1]).item(),
                phase=(f[2].item() * 2 * math.pi) % (2 * math.pi),
                coherence=torch.sigmoid(f[3]).item(),
                frequency=0.0,  # Not reconstructed
            ))
        
        return attractors
    
    def compute_loss(self, attractors: List[AttractorFeatures], 
                    batch_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute VQ-VAE loss.
        
        Returns:
            Dictionary with loss components
        """
        if len(attractors) == 0:
            return {'total': torch.tensor(0.0), 'codebook': torch.tensor(0.0)}
        
        # Convert to tensor
        features = torch.stack([a.to_tensor() for a in attractors])  # [N, 4]
        
        # Encode
        latents = self.encoder(features)  # [N, latent_dim]
        
        # Vector quantize
        with torch.no_grad():
            # Find nearest codebook entries
            distances = torch.cdist(latents, self.codebook)  # [N, vocab_size]
            token_ids = distances.argmin(dim=1)  # [N]
            quantized = self.codebook[token_ids]  # [N, latent_dim]
        
        # Loss 1: Reconstruction loss (how well decoded matches input)
        reconstructed = self.decoder(quantized)
        loss_reconstruction = F.mse_loss(reconstructed, features)
        
        # Loss 2: Codebook loss (update codebook to match encodings)
        loss_codebook = F.mse_loss(quantized.detach(), latents)
        
        # Loss 3: Commitment loss (keep encoder committed to codebook)
        loss_commitment = self.commitment_cost * F.mse_loss(
            quantized.detach(), latents
        )
        
        # Update codebook (exponential moving average)
        self._update_codebook(latents, token_ids)
        
        total_loss = loss_reconstruction + loss_codebook + loss_commitment
        
        return {
            'total': total_loss,
            'reconstruction': loss_reconstruction,
            'codebook': loss_codebook,
            'commitment': loss_commitment,
        }
    
    def _update_codebook(self, latents: torch.Tensor, token_ids: torch.Tensor):
        """Update codebook using exponential moving average"""
        decay = 0.99
        
        # Count cluster assignments
        updated_cluster_size = self.cluster_size * decay + \
                              (1 - decay) * torch.bincount(token_ids, minlength=self.vocab_size)
        
        # Update cluster centers
        dw = torch.zeros_like(self.codebook)
        dw.scatter_add_(0, token_ids.unsqueeze(1).expand(-1, self.latent_dim),
                       latents)
        
        updated_w = self.w_avg * decay + (1 - decay) * dw
        
        # Normalize
        n = updated_cluster_size.sum()
        updated_cluster_size = (
            (updated_cluster_size + 1e-5) /
            (n + self.vocab_size * 1e-5) * n
        )
        
        normalised_updated_w = updated_w / updated_cluster_size.unsqueeze(1)
        self.codebook.data.copy_(normalised_updated_w)
        
        self.cluster_size.data.copy_(updated_cluster_size)
        self.w_avg.data.copy_(updated_w)


class AttractorTokenizerTrainer:
    """Training loop for AttractorTokenizer (Phase 2)"""
    
    def __init__(self, tokenizer: AttractorTokenizer, lr: float = 1e-3, 
                 weight_decay: float = 1e-5):
        self.tokenizer = tokenizer
        self.optimizer = AdamW(tokenizer.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100_000)
        
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def training_step(self, batch: List[List[AttractorFeatures]]) -> Dict[str, float]:
        """
        Single training step on a batch of attractor sequences.
        
        Args:
            batch: List of [List of AttractorFeatures]
        
        Returns:
            Dictionary of loss values
        """
        batch_loss = {'total': 0.0, 'reconstruction': 0.0, 'codebook': 0.0, 'commitment': 0.0}
        
        for sample_idx, attractors in enumerate(batch):
            if len(attractors) == 0:
                continue
            
            losses = self.tokenizer.compute_loss(attractors, sample_idx)
            
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate
            for key in batch_loss:
                batch_loss[key] += losses[key].item()
        
        # Average
        n_samples = sum(1 for s in batch if len(s) > 0)
        if n_samples > 0:
            for key in batch_loss:
                batch_loss[key] /= n_samples
        
        return batch_loss
    
    @torch.no_grad()
    def evaluate(self, batch: List[List[AttractorFeatures]]) -> Dict[str, float]:
        """Evaluate on validation batch"""
        metrics = {
            'reconstruction': 0.0,
            'token_perplexity': 0.0,
            'encoding_accuracy': 0.0,
        }
        
        all_tokens = []
        all_confidences = []
        
        for attractors in batch:
            if len(attractors) == 0:
                continue
            
            # Encode
            token_seq = self.tokenizer.encode(attractors)
            all_tokens.extend(token_seq.tokens)
            all_confidences.extend(token_seq.confidences)
            
            # Reconstruct
            reconstructed = self.tokenizer.decode(token_seq.tokens)
            
            # Reconstruction error
            orig_features = torch.stack([a.to_tensor() for a in attractors])
            recon_features = torch.stack([a.to_tensor() for a in reconstructed])
            
            metrics['reconstruction'] += F.mse_loss(recon_features, orig_features).item()
        
        # Token perplexity (entropy of token distribution)
        if all_tokens:
            token_counts = {}
            for t in all_tokens:
                token_counts[t] = token_counts.get(t, 0) + 1
            
            probs = torch.tensor([count / len(all_tokens) for count in token_counts.values()])
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            metrics['token_perplexity'] = 2 ** entropy
            
            # Mean encoding confidence
            if all_confidences:
                metrics['encoding_accuracy'] = sum(all_confidences) / len(all_confidences)
        
        # Average
        n_samples = sum(1 for s in batch if len(s) > 0)
        if n_samples > 0:
            metrics['reconstruction'] /= n_samples
        
        return metrics
    
    def train_epoch(self, train_loader, val_loader=None, epoch: int = 0) -> Dict:
        """Train for one epoch"""
        self.tokenizer.train()
        
        train_losses = {'total': [], 'reconstruction': [], 'codebook': [], 'commitment': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            losses = self.training_step(batch)
            
            for key in losses:
                train_losses[key].append(losses[key])
            
            pbar.set_postfix({
                'loss': losses['total'],
                'perp': losses.get('codebook', 0),
            })
        
        # Validation
        val_metrics = {}
        if val_loader:
            self.tokenizer.eval()
            val_metrics = self._evaluate_loader(val_loader)
        
        return {
            'train_losses': train_losses,
            'val_metrics': val_metrics,
        }
    
    def _evaluate_loader(self, loader) -> Dict:
        """Evaluate on full loader"""
        metrics = {
            'reconstruction': [],
            'token_perplexity': [],
            'encoding_accuracy': [],
        }
        
        for batch in tqdm(loader, desc="Validation"):
            batch_metrics = self.evaluate(batch)
            for key in metrics:
                metrics[key].append(batch_metrics[key])
        
        # Average
        return {k: sum(v) / len(v) for k, v in metrics.items() if v}


# Phase 2 CLI Integration
def phase2_tokenize_command(args):
    """CLI command for Phase 2 tokenization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Attractor Tokenization")
    parser.add_argument("--data-dir", type=str, default="./datasets/multimodal_corpus",
                       help="Path to curated multi-modal corpus")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/phase2",
                       help="Directory to save tokenizer checkpoint")
    parser.add_argument("--vocab-size", type=int, default=65536,
                       help="Tokenizer vocabulary size")
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load corpus
    print(f"Loading multi-modal corpus from {args.data_dir}...")
    # TODO: Implement corpus loading
    
    # Initialize tokenizer
    tokenizer = AttractorTokenizer(vocab_size=args.vocab_size)
    trainer = AttractorTokenizerTrainer(tokenizer)
    
    print(f"Starting Phase 2 tokenization training...")
    print(f"Epochs: {args.num_epochs}, Batch size: {args.batch_size}, Device: {args.device}")
    
    # Training loop
    # TODO: Implement training loop


if __name__ == "__main__":
    print("Phase 2: Attractor Tokenization")
    print("Use 'bifrost phase2 tokenize --help' for CLI usage")
