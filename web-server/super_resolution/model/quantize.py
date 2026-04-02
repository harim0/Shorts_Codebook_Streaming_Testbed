"""Factorized L2-Normalized Vector Quantization with Entropy Regularization.

Combines four stabilization techniques:
  1. Factorized projection (embed_dim -> codebook_dim) for efficient utilization
  2. L2 normalization on both encoder outputs and codebook entries
  3. EMA updates with dead code replacement
  4. Entropy regularization loss for uniform codebook usage
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedL2VQ(nn.Module):
    """Vector Quantization layer with factorized L2-normalized codebook.

    Args:
        embed_dim:    encoder output dimension (e.g. 64)
        codebook_dim: factorized low dimension for codebook (e.g. 8)
        n_embed:      codebook size (e.g. 1024)
        decay:        EMA decay rate
        commitment_cost: beta for commitment loss
        dead_code_threshold: min usage count before code is considered dead
    """

    def __init__(self, embed_dim=64, codebook_dim=8, n_embed=1024,
                 decay=0.99, commitment_cost=0.25, dead_code_threshold=2.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.codebook_dim = codebook_dim
        self.n_embed = n_embed
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.dead_code_threshold = dead_code_threshold

        # Factorized projections
        self.proj_down = nn.Conv2d(embed_dim, codebook_dim, 1)
        self.proj_up = nn.Conv2d(codebook_dim, embed_dim, 1)

        # Codebook embeddings (L2-normalized during forward)
        self.embedding = nn.Embedding(n_embed, codebook_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_embed, 1.0 / n_embed)

        # EMA tracking buffers
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", torch.zeros(n_embed, codebook_dim))
        self.register_buffer("usage_count", torch.zeros(n_embed))
        self.register_buffer("iter_count", torch.zeros(1))

    def _l2_normalize(self, x, dim=-1):
        return F.normalize(x, p=2, dim=dim)

    @torch.amp.autocast('cuda', enabled=False)
    def _ema_update(self, z_flat, indices):
        """Update codebook with Exponential Moving Average."""
        z_flat = z_flat.float()
        # One-hot encoding
        encodings = F.one_hot(indices, self.n_embed).float()  # (N, K)

        # Cluster sizes
        batch_cluster_size = encodings.sum(dim=0)  # (K,)
        self.cluster_size.data.mul_(self.decay).add_(
            batch_cluster_size, alpha=1 - self.decay
        )

        # Embed averages (sum of assigned vectors per code)
        batch_embed_sum = encodings.T @ z_flat  # (K, D)
        self.embed_avg.data.mul_(self.decay).add_(
            batch_embed_sum, alpha=1 - self.decay
        )

        # Laplace smoothing to avoid division by zero
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + 1e-5) / (n + self.n_embed * 1e-5) * n
        )

        # Update codebook
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

        # Track usage
        self.usage_count.data.add_(batch_cluster_size)

    def replace_dead_codes(self, z_flat):
        """Replace dead codebook entries with encoder outputs that have high error.

        Called periodically during training (not every step).
        """
        dead_mask = self.cluster_size < self.dead_code_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        n_dead = int(n_dead)
        n_available = z_flat.size(0)
        if n_available == 0:
            return 0

        # Sample random encoder outputs to replace dead codes
        replace_idx = torch.randint(0, n_available, (n_dead,), device=z_flat.device)
        new_codes = z_flat[replace_idx].detach()

        # L2 normalize the replacements
        new_codes = self._l2_normalize(new_codes, dim=-1)

        self.embedding.weight.data[dead_mask] = new_codes
        self.cluster_size.data[dead_mask] = self.dead_code_threshold
        self.embed_avg.data[dead_mask] = (
            new_codes * self.dead_code_threshold
        )

        return n_dead

    def forward(self, z):
        """
        Args:
            z: encoder output (B, embed_dim, H, W)

        Returns:
            z_q:     quantized output projected back to embed_dim (B, embed_dim, H, W)
            vq_loss: commitment loss (scalar)
            indices: codebook indices (B, H, W)
            metrics: dict with perplexity, utilization, entropy_loss
        """
        B, C, H, W = z.shape
        self.iter_count += 1

        # 1. Factorized projection: embed_dim -> codebook_dim
        z_low = self.proj_down(z)  # (B, codebook_dim, H, W)

        # 2. L2 normalize (force float32 for precision in distance computation)
        # Reshape for distance computation
        z_low_flat = z_low.permute(0, 2, 3, 1).reshape(-1, self.codebook_dim).float()  # (BHW, D)
        z_low_norm = self._l2_normalize(z_low_flat, dim=-1)
        e_norm = self._l2_normalize(self.embedding.weight.float(), dim=-1)

        # 3. Nearest neighbor (cosine distance = L2 on normalized vectors)
        dist = torch.cdist(z_low_norm, e_norm)  # (BHW, K)
        indices = dist.argmin(dim=-1)  # (BHW,)

        # Quantized vectors (from normalized codebook)
        z_q_low_flat = e_norm[indices]  # (BHW, D)

        # 4. EMA update (training only)
        if self.training:
            self._ema_update(z_low_norm, indices)

        # 5. Commitment loss: push encoder output toward codebook
        commitment_loss = self.commitment_cost * F.mse_loss(
            z_low_norm, z_q_low_flat.detach()
        )

        # Straight-through estimator
        z_q_low_flat = z_low_norm + (z_q_low_flat - z_low_norm).detach()

        # Reshape back to spatial
        z_q_low = z_q_low_flat.reshape(B, H, W, self.codebook_dim).permute(0, 3, 1, 2)

        # 6. Project back to embed_dim
        z_q = self.proj_up(z_q_low)  # (B, embed_dim, H, W)

        # 7. Compute metrics
        indices_2d = indices.reshape(B, H, W)
        metrics = self._compute_metrics(indices)

        return z_q, commitment_loss, indices_2d, metrics

    def _compute_metrics(self, indices):
        """Compute codebook health metrics and entropy loss."""
        flat = indices.flatten()

        # Usage histogram
        usage = torch.bincount(flat, minlength=self.n_embed).float()
        usage_prob = usage / (usage.sum() + 1e-8)

        # Entropy
        log_prob = torch.log(usage_prob + 1e-8)
        entropy = -(usage_prob * log_prob).sum()
        max_entropy = math.log(self.n_embed)

        # Perplexity = exp(entropy)
        perplexity = torch.exp(entropy)

        # Utilization
        active = (usage > 0).sum().float()
        utilization = active / self.n_embed

        # Top-10 dominance
        top10 = usage_prob.topk(10).values.sum()

        # Entropy regularization loss: minimize (max_entropy - entropy)
        entropy_loss = max_entropy - entropy

        return {
            "entropy_loss": entropy_loss,
            "perplexity": perplexity.item(),
            "utilization": utilization.item(),
            "top10_dominance": top10.item(),
            "dead_codes": (usage == 0).sum().item(),
            "entropy": entropy.item(),
        }

    @torch.no_grad()
    def initialize_from_kmeans(self, z_samples):
        """Initialize codebook using K-means on projected encoder outputs.

        Args:
            z_samples: encoder outputs (N, embed_dim, H, W) from a data subset
        """
        # Project and normalize
        z_low = self.proj_down(z_samples)
        z_flat = z_low.permute(0, 2, 3, 1).reshape(-1, self.codebook_dim)
        z_flat = self._l2_normalize(z_flat, dim=-1).cpu().numpy()
        self._kmeans_fit(z_flat)

    @torch.no_grad()
    def _kmeans_fit(self, z_flat):
        """Run K-means on pre-projected, normalized numpy array and set codebook."""
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np

        # Subsample if too many points
        max_points = 500_000
        if z_flat.shape[0] > max_points:
            idx = np.random.permutation(z_flat.shape[0])[:max_points]
            z_flat = z_flat[idx]

        print(f"  K-means on {z_flat.shape[0]} points, K={self.n_embed} ...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_embed,
            batch_size=4096,
            max_iter=100,
            n_init=1,
        )
        kmeans.fit(z_flat)

        # Set codebook to centroids (L2 normalized)
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(
            self.embedding.weight.device
        )
        centroids = self._l2_normalize(centroids, dim=-1)
        self.embedding.weight.data.copy_(centroids)
        self.embed_avg.data.copy_(centroids * self.dead_code_threshold)
        self.cluster_size.data.fill_(self.dead_code_threshold)

        print(f"  K-means initialization done. Inertia: {kmeans.inertia_:.4f}")
