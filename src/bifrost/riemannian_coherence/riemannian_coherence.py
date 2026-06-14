"""
Riemannian Semantic Coherence Module

Implements learned Riemannian manifold structure on frequency attractor space
to enable semantic coherence measurement through geodesic distances.

Architecture:
    Input: FrequencyAttractors from phase-lock bridge
    ├── RiemannianMetricLearner: Learn metric tensor g_ij on attractor space
    ├── GeodesicComputer: Compute shortest paths on learned manifold
    ├── CoherenceScorer: Map geodesic distances to semantic coherence
    └── ManifoldProjector: Project attractors to 2D/3D for visualization
    
    Training: Triplet loss (synonyms close, antonyms far, unrelated farther)

Mathematical Foundation:
    - Metric tensor g(x) defines local distances: ds² = g_ij dx^i dx^j
    - Geodesic: shortest path γ(t) minimizing ∫√(g_ij γ̇^i γ̇^j) dt
    - Semantic coherence inversely proportional to geodesic distance

References:
    - Riemannian Geometry (Do Carmo, 1992)
    - Metric Learning for Semantic Similarity (Weinberger & Saul, 2009)
    - Graph-Based Manifold Learning (Tenenbaum et al., 2000)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from collections import defaultdict

from ..phase_lock_bridge import FrequencyAttractor


@dataclass
class SemanticCoherenceOutput:
    """Output container for Riemannian semantic coherence computation.
    
    Attributes:
        coherence_scores: (n_attractors, n_attractors) pairwise coherence [0, 1]
        geodesic_distances: (n_attractors, n_attractors) geodesic distances
        manifold_coords: (n_attractors, manifold_dim) low-dimensional projection
        metric_tensor: (n_attractors, d_model, d_model) learned metric g_ij
        semantic_clusters: Dict[str, List[int]] attractor indices by cluster
    """
    coherence_scores: torch.Tensor
    geodesic_distances: torch.Tensor
    manifold_coords: torch.Tensor
    metric_tensor: torch.Tensor
    semantic_clusters: Dict[str, List[int]]


class RiemannianMetricLearner(nn.Module):
    """Learns Riemannian metric tensor g_ij on frequency attractor space.
    
    The metric defines local distances: ds² = dx^T G(x) dx
    where G(x) is positive definite at each point x.
    
    Implementation uses a neural network to map attractor features to
    a lower-dimensional metric representation via Cholesky decomposition
    to ensure positive definiteness.
    
    Attributes:
        d_model: Dimension of attractor feature space
        metric_dim: Intrinsic manifold dimension (typically << d_model)
        
    Complexity:
        Forward: O(n_attractors * d_model²) for full metric computation
        Space: O(d_model²) for metric parameters
    """
    
    def __init__(
        self,
        d_model: int = 768,
        metric_dim: int = 64,
        n_hidden_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.metric_dim = metric_dim
        
        # Metric network: attractor features -> Cholesky factors
        # Ensures positive definite metric via L @ L.T
        layers = []
        in_dim = d_model
        for i in range(n_hidden_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        # Output: lower-triangular Cholesky factors (metric_dim*(metric_dim+1)//2)
        n_cholesky = metric_dim * (metric_dim + 1) // 2
        layers.append(nn.Linear(hidden_dim, n_cholesky))
        
        self.metric_net = nn.Sequential(*layers)
        
        # Learnable base metric (Euclidean-like initialization)
        self.register_buffer('base_metric', torch.eye(metric_dim))
        
        # Temperature for distance scaling
        self.distance_temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, attractors: List[FrequencyAttractor]) -> torch.Tensor:
        """Compute metric tensors for each attractor.
        
        Args:
            attractors: List of FrequencyAttractor objects
            
        Returns:
            metrics: (n_attractors, metric_dim, metric_dim) metric tensors g_ij
            Each metric is symmetric positive definite.
            
        Raises:
            ValueError: If attractor list is empty
        """
        if not attractors:
            raise ValueError("Empty attractor list provided")
        
        n_attractors = len(attractors)
        device = attractors[0].centroid.device
        
        # Stack attractor centroids: (n_attractors, d_model)
        features = torch.stack([a.centroid for a in attractors], dim=0)
        
        # Compute Cholesky factors for each attractor
        cholesky_flat = self.metric_net(features)  # (n_attractors, n_cholesky)
        
        # Build lower-triangular matrices
        metrics = []
        for i in range(n_attractors):
            L = torch.zeros(self.metric_dim, self.metric_dim, device=device)
            idx = 0
            for row in range(self.metric_dim):
                for col in range(row + 1):
                    if row == col:
                        # Diagonal: ensure positive via softplus
                        L[row, col] = F.softplus(cholesky_flat[i, idx]) + 1e-6
                    else:
                        L[row, col] = cholesky_flat[i, idx]
                    idx += 1
            
            # G = L @ L.T ensures positive definiteness
            G = L @ L.T
            metrics.append(G)
        
        return torch.stack(metrics, dim=0)
    
    def local_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Riemannian distance between two points under metric g.
        
        For nearby points, approximates geodesic distance via:
        d(x, y) ≈ sqrt((y-x)^T G(x) (y-x))
        
        Args:
            x: (d_model,) point on manifold
            y: (d_model,) point on manifold
            metric: (metric_dim, metric_dim) metric tensor at x
            
        Returns:
            distance: scalar Riemannian distance
            
        Complexity: O(metric_dim²)
        """
        # Project to metric space (use first metric_dim dimensions)
        dx = (y - x)[:self.metric_dim]
        
        # d² = dx^T G dx
        dist_sq = dx @ metric @ dx
        distance = torch.sqrt(dist_sq + 1e-8)
        
        return distance / F.softplus(self.distance_temp)


class GeodesicComputer(nn.Module):
    """Computes approximate geodesic distances on learned Riemannian manifold.
    
    Uses graph-based approximation:
    1. Build k-NN graph in attractor space with Riemannian edge weights
    2. Compute shortest paths (geodesic approximations) via Dijkstra/Floyd-Warshall
    
    This avoids expensive ODE integration for exact geodesics while
    preserving manifold structure for semantic coherence tasks.
    
    Attributes:
        k_neighbors: Number of nearest neighbors for graph construction
        algorithm: Shortest path algorithm ('dijkstra' or 'floyd')
        
    Complexity:
        Graph construction: O(n_attractors² * d_model)
        Shortest paths: O(n_attractors * (k * log n)) for Dijkstra from all nodes
        Space: O(n_attractors²) for distance matrix
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        algorithm: str = 'dijkstra',
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.algorithm = algorithm
    
    def forward(
        self,
        attractors: List[FrequencyAttractor],
        metrics: torch.Tensor,
        metric_learner: RiemannianMetricLearner,
    ) -> torch.Tensor:
        """Compute pairwise geodesic distances between all attractors.
        
        Args:
            attractors: List of FrequencyAttractor objects
            metrics: (n_attractors, metric_dim, metric_dim) metric tensors
            metric_learner: RiemannianMetricLearner for local distance computation
            
        Returns:
            distances: (n_attractors, n_attractors) geodesic distance matrix
            distances[i, j] = geodesic distance from attractor i to j
            
        Raises:
            ValueError: If fewer than 2 attractors provided
        """
        n_attractors = len(attractors)
        if n_attractors < 2:
            raise ValueError(f"Need at least 2 attractors, got {n_attractors}")
        
        device = attractors[0].centroid.device
        
        # Build k-NN graph with Riemannian edge weights
        adj_matrix = self._build_riemannian_graph(
            attractors, metrics, metric_learner, device
        )
        
        # Compute all-pairs shortest paths
        if self.algorithm == 'dijkstra':
            distances = self._dijkstra_all_pairs(adj_matrix)
        elif self.algorithm == 'floyd':
            distances = self._floyd_warshall(adj_matrix)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return distances
    
    def _build_riemannian_graph(
        self,
        attractors: List[FrequencyAttractor],
        metrics: torch.Tensor,
        metric_learner: RiemannianMetricLearner,
        device: torch.device,
    ) -> torch.Tensor:
        """Build k-NN graph with Riemannian distances as edge weights.
        
        Returns:
            adj_matrix: (n_attractors, n_attractors) adjacency with weights
            Non-edges have weight = inf
        """
        n = len(attractors)
        
        # Clamp k_neighbors to n-1 (max possible neighbors excluding self)
        effective_k = min(self.k_neighbors, n - 1)
        
        # Compute pairwise Euclidean distances for k-NN selection
        positions = torch.stack([a.centroid for a in attractors], dim=0)
        euclidean_dists = torch.cdist(positions, positions)
        
        # Get k nearest neighbors for each node (excluding self)
        knn_indices = torch.argsort(euclidean_dists, dim=-1)[:, 1:effective_k+1]
        
        # Compute Riemannian edge weights for k-NN edges
        adj_matrix = torch.full((n, n), float('inf'), device=device)
        
        for i in range(n):
            for j_idx in range(effective_k):
                j = knn_indices[i, j_idx].item()
                
                # Use metric at node i for edge weight
                dist = metric_learner.local_distance(
                    attractors[i].centroid,
                    attractors[j].centroid,
                    metrics[i],
                )
                adj_matrix[i, j] = dist
                adj_matrix[j, i] = dist  # Symmetric
        
        # Self-distances are 0
        adj_matrix.fill_diagonal_(0)
        
        return adj_matrix
    
    def _dijkstra_all_pairs(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Run Dijkstra from all nodes to get geodesic approximations.
        
        Complexity: O(n * (k * log n)) where k = k_neighbors
        """
        n = adj_matrix.shape[0]
        device = adj_matrix.device
        distances = torch.full((n, n), float('inf'), device=device)
        
        for source in range(n):
            dist = self._dijkstra_single(adj_matrix, source)
            distances[source] = dist
        
        return distances
    
    def _dijkstra_single(self, adj_matrix: torch.Tensor, source: int) -> torch.Tensor:
        """Dijkstra's algorithm from single source.
        
        Complexity: O((n + m) log n) where m = n * k_neighbors
        """
        n = adj_matrix.shape[0]
        device = adj_matrix.device
        
        dist = torch.full((n,), float('inf'), device=device)
        dist[source] = 0
        visited = torch.zeros(n, dtype=torch.bool, device=device)
        
        for _ in range(n):
            # Find unvisited node with minimum distance
            unvisited_dists = torch.where(~visited, dist, torch.tensor(float('inf'), device=device))
            u = torch.argmin(unvisited_dists).item()
            
            if dist[u] == float('inf'):
                break
            
            visited[u] = True
            
            # Update distances to neighbors
            for v in range(n):
                if not visited[v] and adj_matrix[u, v] < float('inf'):
                    alt = dist[u] + adj_matrix[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
        
        return dist
    
    def _floyd_warshall(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Floyd-Warshall algorithm for all-pairs shortest paths.
        
        Complexity: O(n³) - only suitable for small graphs (n < 100)
        """
        n = adj_matrix.shape[0]
        dist = adj_matrix.clone()
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        return dist


class CoherenceScorer(nn.Module):
    """Maps geodesic distances to semantic coherence scores.
    
    Coherence is inversely related to geodesic distance:
    - High coherence (close to 1): nearby on manifold (similar semantics)
    - Low coherence (close to 0): far apart (different semantics)
    
    Uses learnable sigmoid with temperature and bias to adapt
    to dataset-specific distance scales.
    
    Attributes:
        temperature: Controls steepness of coherence decay
        bias: Shifts inflection point
    """
    
    def __init__(self, init_temperature: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.bias = nn.Parameter(torch.tensor(init_bias))
    
    def forward(self, geodesic_distances: torch.Tensor) -> torch.Tensor:
        """Convert geodesic distances to coherence scores.
        
        Args:
            geodesic_distances: (n, n) pairwise geodesic distances
            
        Returns:
            coherence: (n, n) coherence scores in [0, 1]
            
        Formula: coherence = sigmoid(-(distance - bias) / temperature)
        """
        # Clip infinite distances for numerical stability
        finite_distances = torch.clamp(geodesic_distances, max=1e6)
        
        # Coherence decreases with distance: sigmoid(-(d - bias) / temp)
        # Small distance -> high coherence (close to 1)
        # Large distance -> low coherence (close to 0)
        temp = F.softplus(self.temperature) + 0.1  # Ensure positive with minimum
        coherence = torch.sigmoid(-(finite_distances - self.bias) / temp)
        
        # Self-coherence is always 1
        coherence.fill_diagonal_(1.0)
        
        return coherence


class ManifoldProjector(nn.Module):
    """Projects high-dimensional attractors to 2D/3D for visualization.
    
    Uses PCA on the geodesic distance matrix (classical Multidimensional Scaling)
    to preserve manifold structure in low dimensions.
    
    Alternative: t-SNE or UMAP on geodesic distances for non-linear projection.
    
    Attributes:
        n_components: Output dimension (2 or 3)
        method: Projection method ('mds', 'tsne', 'umap')
    """
    
    def __init__(
        self,
        n_components: int = 2,
        method: str = 'mds',
    ):
        super().__init__()
        self.n_components = n_components
        self.method = method
    
    def forward(
        self,
        geodesic_distances: torch.Tensor,
        attractor_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project attractors to low-dimensional space.
        
        Args:
            geodesic_distances: (n, n) pairwise geodesic distance matrix
            attractor_features: Optional (n, d_model) for feature-based projection
            
        Returns:
            coords: (n, n_components) low-dimensional coordinates
            
        Raises:
            ValueError: If method is not supported
        """
        if self.method == 'mds':
            return self._classical_mds(geodesic_distances)
        elif self.method == 'pca' and attractor_features is not None:
            return self._pca_projection(attractor_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _classical_mds(self, distances: torch.Tensor) -> torch.Tensor:
        """Classical Multidimensional Scaling on distance matrix.
        
        Algorithm:
        1. Convert distances to inner products: B = -0.5 * H D² H
        2. Eigendecomposition of B
        3. Take top k eigenvectors scaled by sqrt(eigenvalues)
        
        Complexity: O(n³) for eigendecomposition
        """
        n = distances.shape[0]
        device = distances.device
        
        # Double centering: B = -0.5 * H D² H
        D_sq = distances ** 2
        H = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
        B = -0.5 * H @ D_sq @ H
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(B)
        
        # Take top k positive eigenvalues
        idx = torch.argsort(eigenvalues, descending=True)
        top_k = idx[:self.n_components]
        
        # Coordinates: sqrt(λ) * v
        coords = eigenvectors[:, top_k] * torch.sqrt(torch.clamp(eigenvalues[top_k], min=0))
        
        return coords
    
    def _pca_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Simple PCA on attractor features."""
        centered = features - features.mean(dim=0)
        U, S, V = torch.svd(centered)
        return U[:, :self.n_components] * S[:self.n_components]


class TripletSemanticLoss(nn.Module):
    """Triplet loss for semantic similarity learning on Riemannian manifold.
    
    Training objective:
    - Pull synonyms together (small geodesic distance)
    - Push antonyms apart (large geodesic distance)
    - Push random pairs even farther
    
    Loss = max(d(a, p) - d(a, n) + margin, 0)
    where a=anchor, p=positive (synonym), n=negative (antonym/random)
    
    Attributes:
        margin: Minimum separation between pos/neg pairs
        distance_metric: 'geodesic' or 'euclidean'
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = 'geodesic',
    ):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(
        self,
        geodesic_distances: torch.Tensor,
        anchor_idx: torch.Tensor,
        positive_idx: torch.Tensor,
        negative_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss for semantic training.
        
        Args:
            geodesic_distances: (n, n) pairwise distance matrix
            anchor_idx: (batch_size,) anchor attractor indices
            positive_idx: (batch_size,) positive (synonym) indices
            negative_idx: (batch_size,) negative (antonym) indices
            
        Returns:
            loss: scalar triplet loss (mean over batch)
            
        Raises:
            ValueError: If indices are out of bounds
        """
        n = geodesic_distances.shape[0]
        
        if anchor_idx.max() >= n or positive_idx.max() >= n or negative_idx.max() >= n:
            raise ValueError(f"Indices out of bounds (max should be < {n})")
        
        d_anchor_positive = geodesic_distances[anchor_idx, positive_idx]
        d_anchor_negative = geodesic_distances[anchor_idx, negative_idx]
        
        losses = F.relu(d_anchor_positive - d_anchor_negative + self.margin)
        
        return losses.mean()
    
    def compute_semantic_accuracy(
        self,
        geodesic_distances: torch.Tensor,
        synonym_pairs: List[Tuple[int, int]],
        antonym_pairs: List[Tuple[int, int]],
    ) -> float:
        """Compute accuracy: % of pairs where synonyms closer than antonyms.
        
        Returns:
            accuracy: float in [0, 1]
        """
        correct = 0
        total = 0
        
        for anchor, synonym in synonym_pairs:
            for _, antonym in antonym_pairs:
                if geodesic_distances[anchor, synonym] < geodesic_distances[anchor, antonym]:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0


class RiemannianSemanticCoherence(nn.Module):
    """Complete Riemannian semantic coherence system.
    
    Integrates metric learning, geodesic computation, coherence scoring,
    and manifold projection into a unified module for semantic analysis.
    
    Takes FrequencyAttractors from phase-lock bridge and produces semantic
    coherence metrics.
    
    Example:
        >>> coherence = RiemannianSemanticCoherence(d_model=768, metric_dim=64)
        >>> attractors = [FrequencyAttractor(...), ...]
        >>> output = coherence(attractors)
        >>> print(output.coherence_scores)
        >>> print(output.manifold_coords)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        metric_dim: int = 64,
        n_heads: int = 4,
        k_neighbors: int = 5,
        manifold_dim: int = 2,
        triplet_margin: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.metric_dim = metric_dim
        
        self.metric_learner = RiemannianMetricLearner(d_model=d_model, metric_dim=metric_dim)
        self.geodesic_computer = GeodesicComputer(k_neighbors=k_neighbors)
        self.coherence_scorer = CoherenceScorer()
        self.manifold_projector = ManifoldProjector(n_components=manifold_dim)
        self.triplet_loss = TripletSemanticLoss(margin=triplet_margin)
        
        self.semantic_embedder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, metric_dim),
        )
    
    def forward(
        self,
        attractors: List[FrequencyAttractor],
        return_projection: bool = True,
    ) -> SemanticCoherenceOutput:
        """Compute semantic coherence for attractor set.
        
        Args:
            attractors: List of FrequencyAttractor from phase-lock bridge
            return_projection: Whether to compute low-D projection
            
        Returns:
            SemanticCoherenceOutput with coherence scores, distances,
            manifold coordinates, metric tensors, and empty clusters
            
        Raises:
            ValueError: If attractor list has fewer than 2 elements
        """
        if len(attractors) < 2:
            raise ValueError(f"Need >=2 attractors for coherence, got {len(attractors)}")
        
        metrics = self.metric_learner(attractors)
        geodesic_dists = self.geodesic_computer(attractors, metrics, self.metric_learner)
        coherence = self.coherence_scorer(geodesic_dists)
        
        manifold_coords = None
        if return_projection:
            manifold_coords = self.manifold_projector(geodesic_dists)
        
        return SemanticCoherenceOutput(
            coherence_scores=coherence,
            geodesic_distances=geodesic_dists,
            manifold_coords=manifold_coords if manifold_coords is not None else torch.zeros(len(attractors), 2),
            metric_tensor=metrics,
            semantic_clusters={},
        )
    
    def training_step(
        self,
        attractors: List[FrequencyAttractor],
        anchor_idx: torch.Tensor,
        positive_idx: torch.Tensor,
        negative_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, SemanticCoherenceOutput]:
        """Single training step with triplet loss.
        
        Args:
            attractors: List of FrequencyAttractor
            anchor_idx: (batch_size,) anchor indices
            positive_idx: (batch_size,) positive (synonym) indices
            negative_idx: (batch_size,) negative (antonym) indices
            
        Returns:
            loss: scalar triplet loss
            output: SemanticCoherenceOutput from forward pass
        """
        output = self.forward(attractors, return_projection=False)
        
        loss = self.triplet_loss(
            output.geodesic_distances,
            anchor_idx,
            positive_idx,
            negative_idx,
        )
        
        return loss, output
    
    def find_semantic_clusters(
        self,
        attractors: List[FrequencyAttractor],
        coherence_threshold: float = 0.5,
    ) -> Dict[str, List[int]]:
        """Cluster attractors by semantic coherence (simple threshold-based).
        
        Args:
            attractors: List of FrequencyAttractor
            coherence_threshold: Minimum coherence for cluster membership
            
        Returns:
            clusters: Dict mapping cluster_id to list of attractor indices
        """
        with torch.no_grad():
            output = self.forward(attractors, return_projection=False)
            coherence = output.coherence_scores
        
        n = len(attractors)
        visited = [False] * n
        clusters = {}
        cluster_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            cluster = []
            queue = [i]
            visited[i] = True
            
            while queue:
                node = queue.pop(0)
                cluster.append(node)
                
                for j in range(n):
                    if not visited[j] and coherence[node, j] > coherence_threshold:
                        visited[j] = True
                        queue.append(j)
            
            if cluster:
                clusters[f"cluster_{cluster_id:03d}"] = cluster
                cluster_id += 1
        
        return clusters
    
    def semantic_search(
        self,
        query_attractor: FrequencyAttractor,
        attractor_database: List[FrequencyAttractor],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find semantically similar attractors in database.
        
        Args:
            query_attractor: Query attractor
            attractor_database: Database of attractors to search
            top_k: Number of results to return
            
        Returns:
            results: List of (attractor_idx, coherence_score) tuples, sorted by score
        """
        all_attractors = [query_attractor] + attractor_database
        
        with torch.no_grad():
            output = self.forward(all_attractors, return_projection=False)
            coherence = output.coherence_scores
        
        query_coherence = coherence[0, 1:]
        top_k_values, top_k_indices = torch.topk(query_coherence, min(top_k, len(attractor_database)))
        
        results = [
            (idx.item(), score.item())
            for idx, score in zip(top_k_indices, top_k_values)
        ]
        
        return results


def create_triplets_from_labels(
    labels: List[str],
    synonym_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create triplet indices from string labels.
    
    Args:
        labels: List of semantic labels for each attractor
        synonym_map: Optional mapping from label to list of synonyms
        
    Returns:
        anchor_idx, positive_idx, negative_idx: Tensors of triplet indices
    """
    n = len(labels)
    
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)
    
    anchors = []
    positives = []
    negatives = []
    
    for anchor in range(n):
        anchor_label = labels[anchor]
        
        positive_candidates = label_to_indices[anchor_label].copy()
        if synonym_map and anchor_label in synonym_map:
            for syn in synonym_map[anchor_label]:
                if syn in label_to_indices:
                    positive_candidates.extend(label_to_indices[syn])
        
        positive_candidates = [p for p in positive_candidates if p != anchor]
        
        if not positive_candidates:
            continue
        
        positive = np.random.choice(positive_candidates)
        
        negative_candidates = []
        for label, indices in label_to_indices.items():
            if label != anchor_label:
                is_antonym = (
                    synonym_map and 
                    anchor_label in synonym_map and 
                    label in synonym_map[anchor_label]
                )
                if not is_antonym:
                    negative_candidates.extend(indices)
        
        if not negative_candidates:
            for label, indices in label_to_indices.items():
                if label != anchor_label:
                    negative_candidates.extend(indices)
        
        if not negative_candidates:
            continue
        
        negative = np.random.choice(negative_candidates)
        
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    
    return (
        torch.tensor(anchors, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )
