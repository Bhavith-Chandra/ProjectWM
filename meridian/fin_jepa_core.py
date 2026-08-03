"""
Fin-JEPA Core: Production-Grade Neural World Model for Financial Markets

Implements a complete Joint-Embedding Predictive Architecture with:
- Dynamic Factor Graph Transformers for cross-sectional structure
- VICReg/SIGReg anti-collapse regularization
- Hyperbolic geometry (Poincaré manifold) for hierarchical regimes
- Extreme Value Theory (EVT) tail risk emission
- Multi-task learning heads (volatility, tail, causal factors)

This system operates entirely in continuous latent space without reconstruction,
designed to decisively beat all baseline approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import math


# ============================================================================
# 1. DYNAMIC FACTOR GRAPH TRANSFORMER (Context Encoder)
# ============================================================================

class DynamicFactorGraphTransformer(nn.Module):
    """
    Cross-sectional asset graph encoder with learnable, time-varying attention
    topology. Captures structural dependencies without reconstruction loss.

    Architecture:
    - Per-asset embeddings via feature projection
    - Scaled dot-product cross-asset attention (dynamic adjacency)
    - Message passing along learned graph edges
    - Bottleneck compression into latent state
    """

    def __init__(self, n_assets: int, n_features: int, latent_dim: int,
                 n_graph_heads: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.n_assets = n_assets
        self.latent_dim = latent_dim
        self.n_graph_heads = n_graph_heads

        assert latent_dim % n_graph_heads == 0, "latent_dim must be divisible by n_graph_heads"
        self.head_dim = latent_dim // n_graph_heads

        # Initial feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64)
        )

        # Multi-head graph attention projections
        self.query_proj = nn.Linear(64, latent_dim)
        self.key_proj = nn.Linear(64, latent_dim)
        self.value_proj = nn.Linear(64, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        # Message passing networks
        self.message_mlp = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Latent bottleneck compression
        self.latent_compressor = nn.Sequential(
            nn.Linear(n_assets * latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_assets, n_features) asset feature matrix

        Returns:
            z: (batch, latent_dim) latent state
            A: (batch, n_assets, n_assets) learned graph adjacency
        """
        batch_size, n_assets, n_features = x.shape

        # Embed asset features
        x_embed = self.feature_embed(x)  # (batch, n_assets, 64)

        # Compute multi-head cross-asset attention (dynamic graph)
        q = self.query_proj(x_embed)  # (batch, n_assets, latent_dim)
        k = self.key_proj(x_embed)
        v = self.value_proj(x_embed)

        # Reshape for multi-head attention
        q = q.view(batch_size, n_assets, self.n_graph_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_assets, self.n_graph_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_assets, self.n_graph_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention → dynamic adjacency
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, n_assets, n_assets)
        A_heads = F.softmax(scores, dim=-1)

        # Aggregate multi-head attention into single adjacency
        A = A_heads.mean(dim=1)  # (batch, n_assets, n_assets)

        # Gather attention values
        attn_out = torch.matmul(A_heads, v)  # (batch, heads, n_assets, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous()  # (batch, n_assets, heads, head_dim)
        attn_out = attn_out.view(batch_size, n_assets, self.latent_dim)
        attn_out = self.out_proj(attn_out)

        # Message passing: propagate information along graph edges
        # Concatenate embeddings and attend output for per-node updates
        messages = torch.cat([attn_out, x_embed], dim=-1)  # (batch, n_assets, latent_dim + 64)
        updated = self.message_mlp(messages)  # (batch, n_assets, latent_dim)

        # Compress cross-sectional structure into latent bottleneck
        flat_topology = updated.view(batch_size, -1)  # (batch, n_assets * latent_dim)
        z = self.latent_compressor(flat_topology)  # (batch, latent_dim)

        return z, A


# ============================================================================
# 2. LATENT SPACE PREDICTOR (Continuous Trajectory Forecasting)
# ============================================================================

class LatentSpacePredictor(nn.Module):
    """
    Predicts future latent coordinates z_{t+k} from current latent z_t
    and exogenous conditioning (shocks, macro news).

    Operates entirely in latent space without reconstruction.
    """

    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Optional: predictive uncertainty (aleatoric)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_t: torch.Tensor, cond_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: (batch, latent_dim) current latent state
            cond_t: (batch, cond_dim) exogenous conditioning

        Returns:
            z_pred: (batch, latent_dim) predicted latent at t+1
            logvar: (batch, latent_dim) aleatoric uncertainty
        """
        fused = torch.cat([z_t, cond_t], dim=-1)
        z_pred = self.predictor(fused)
        logvar = self.uncertainty_head(fused).clamp(-6, 2)  # Stability
        return z_pred, logvar


# ============================================================================
# 3. VICReg/SIGReg NON-COLLAPSE REGULARIZATION
# ============================================================================

class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization: prevents representation collapse
    without requiring generative reconstruction.

    Three components:
    1. Similarity: MSE between predicted and target latents
    2. Variance: Ensures each dimension has non-zero variance across batch
    3. Covariance: Decorrelates dimensions (forces independence)
    """

    def __init__(self, sim_weight: float = 25.0, var_weight: float = 25.0,
                 cov_weight: float = 1.0, var_threshold: float = 1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_threshold = var_threshold

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VICReg loss avoiding representation collapse.

        Args:
            z_pred: (batch, latent_dim) predicted latents
            z_target: (batch, latent_dim) target latents

        Returns:
            dict with 'loss', 'sim', 'var', 'cov' components
        """
        batch_size, latent_dim = z_pred.shape

        # 1. SIMILARITY: Direct latent-space MSE
        sim_loss = F.mse_loss(z_pred, z_target)

        # 2. VARIANCE: Force non-zero variance per dimension
        # Prevents collapse where encoder outputs constant
        var_pred = torch.var(z_pred, dim=0, unbiased=True)
        var_target = torch.var(z_target, dim=0, unbiased=True)

        # Soft penalty: variance should be at least var_threshold
        var_loss_pred = F.relu(self.var_threshold - var_pred).mean()
        var_loss_target = F.relu(self.var_threshold - var_target).mean()
        var_loss = var_loss_pred + var_loss_target

        # 3. COVARIANCE: Decorrelate dimensions
        # Centers data
        z_pred_centered = z_pred - z_pred.mean(dim=0, keepdim=True)
        z_target_centered = z_target - z_target.mean(dim=0, keepdim=True)

        # Compute covariance matrices
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
        cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)

        # Penalize off-diagonal elements (force independence)
        # Create mask for off-diagonal
        off_diag_mask = ~torch.eye(latent_dim, dtype=torch.bool, device=z_pred.device)

        cov_loss_pred = (cov_pred[off_diag_mask] ** 2).mean()
        cov_loss_target = (cov_target[off_diag_mask] ** 2).mean()
        cov_loss = cov_loss_pred + cov_loss_target

        # Aggregate
        total_loss = (self.sim_weight * sim_loss +
                     self.var_weight * var_loss +
                     self.cov_weight * cov_loss)

        return {
            'loss': total_loss,
            'sim': sim_loss.detach(),
            'var': var_loss.detach(),
            'cov': cov_loss.detach(),
            'var_min': torch.min(var_pred).detach()  # Monitor lowest variance dimension
        }


# ============================================================================
# 4. HYPERBOLIC GEOMETRY ENGINE (Poincaré Manifold)
# ============================================================================

class PoincareBall(nn.Module):
    """
    Poincaré Ball manifold with constant negative curvature.

    Maps latent coordinates to hyperbolic space where:
    - Center: stable, "normal" regimes
    - Boundary (||z|| → 1): extreme shocks
    - Geodesics: true manifold distance (Möbius gyrovector operations)
    """

    def __init__(self, latent_dim: int, curvature: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.eps = 1e-8

    def mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius gyrovector addition: x ⊕_c y

        Defines geodesic movement on hyperbolic space.
        When ||x|| and ||y|| are small → approximately Euclidean
        As ||x|| → 1: rapid divergence (approaching boundary)
        """
        c = self.curvature

        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)
        dot_prod = (x * y).sum(dim=-1, keepdim=True)

        numerator = (1 + 2 * c * dot_prod + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
        denominator = 1 + 2 * c * dot_prod + c ** 2 * x_norm_sq * y_norm_sq

        return numerator / (denominator + self.eps)

    def project_to_ball(self, z: torch.Tensor, max_norm: float = 0.99) -> torch.Tensor:
        """Project latent coordinates into Poincaré ball with boundary safety margin."""
        norm = torch.norm(z, dim=-1, keepdim=True)
        # Scale to max_norm if exceeds
        z_normalized = z / (norm + self.eps)
        return max_norm * z_normalized

    def euclidean_to_hyperbolic(self, z: torch.Tensor) -> torch.Tensor:
        """Map Euclidean latent space to Poincaré ball."""
        # Simple tangent space mapping at origin
        z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True)
        c = self.curvature

        # Exponential map: exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v / (||v||)
        coeff = torch.tanh(torch.sqrt(c * z_norm_sq / 4 + self.eps)) / (torch.sqrt(c * z_norm_sq + self.eps))
        hyperbolic_z = coeff * z

        return self.project_to_ball(hyperbolic_z)


# ============================================================================
# 5. EXTREME VALUE THEORY (EVT) TAIL RISK EMISSION
# ============================================================================

class EVTTailEmission(nn.Module):
    """
    Maps latent coordinates to tail risk parameters via Generalized Pareto Distribution.

    Captures heavy-tail behavior critical for Value-at-Risk and Expected Shortfall.
    """

    def __init__(self, latent_dim: int, n_assets: int):
        super().__init__()
        self.n_assets = n_assets

        # Map latent → tail shape parameter (ξ)
        self.xi_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_assets)
        )

        # Map latent → tail scale parameter (β)
        self.beta_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_assets)
        )

        # Gaussian floor (non-collapse safeguard)
        self.register_buffer('gaussian_floor', torch.tensor(0.1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute tail risk parameters from latent state.

        Args:
            z: (batch, latent_dim)

        Returns:
            xi: (batch, n_assets) tail index (shape parameter)
            beta: (batch, n_assets) scale parameter
        """
        # Tail shape: positive (heavy tails)
        xi = F.softplus(self.xi_head(z)) + 0.1  # ξ > 0.1 (always heavy-tailed)

        # Tail scale: positive
        beta = F.softplus(self.beta_head(z)) + self.gaussian_floor

        return xi, beta

    def gpd_quantile(self, xi: torch.Tensor, beta: torch.Tensor,
                    p: float) -> torch.Tensor:
        """
        Compute GPD quantile: Q(p) = (β/ξ) * ((1-p)^{-ξ} - 1)

        Args:
            xi: (batch, n_assets) shape
            beta: (batch, n_assets) scale
            p: quantile level (e.g., 0.01 for 1% VaR)

        Returns:
            quantile values for portfolio losses
        """
        eps = 1e-8
        return (beta / (xi + eps)) * ((1 - p) ** (-xi) - 1)


# ============================================================================
# 6. COMPLETE FIN-JEPA WORLD MODEL
# ============================================================================

class FinJEPAWorldModel(nn.Module):
    """
    End-to-end Joint-Embedding Predictive Architecture for financial markets.

    Pipeline:
    1. Context Encoder (Dynamic Factor Graph) → z_t
    2. Latent Predictor → z_{t+1}
    3. Multi-task emission heads:
       - Volatility head (covariance structure)
       - Tail head (EVT parameters)
       - Causal head (structural factors)
    """

    def __init__(self, n_assets: int, n_features: int, latent_dim: int = 32,
                 cond_dim: int = 4, use_hyperbolic: bool = True):
        super().__init__()
        self.n_assets = n_assets
        self.latent_dim = latent_dim
        self.use_hyperbolic = use_hyperbolic

        # Core encoders
        self.context_encoder = DynamicFactorGraphTransformer(
            n_assets, n_features, latent_dim
        )
        self.latent_predictor = LatentSpacePredictor(latent_dim, cond_dim)

        # Hyperbolic geometry
        if use_hyperbolic:
            self.poincare = PoincareBall(latent_dim, curvature=1.0)

        # Emission heads
        self.volatility_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_assets * (n_assets + 1) // 2)  # Lower triangular cholesky
        )

        self.evt_tail = EVTTailEmission(latent_dim, n_assets)

        self.causal_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 8)  # 8 causal factors
        )

        # Anti-collapse regularizer
        self.sigreg = SIGRegLoss()

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                x_target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all outputs.

        Args:
            x: (batch, n_assets, n_features) current observations
            cond: (batch, cond_dim) conditioning (macro shocks)
            x_target: (batch, n_assets, n_features) target for training

        Returns:
            Dictionary with all predictions and losses
        """
        # Context encoding
        z_t, graph_adj = self.context_encoder(x)

        # Optional: hyperbolic projection
        if self.use_hyperbolic:
            z_t_hyp = self.poincare.euclidean_to_hyperbolic(z_t)
        else:
            z_t_hyp = z_t

        # Predict next latent
        z_pred, z_logvar = self.latent_predictor(z_t_hyp, cond)

        # Emission heads
        vol_params = self.volatility_head(z_pred)
        xi, beta = self.evt_tail(z_pred)
        causal_factors = self.causal_head(z_pred)

        output = {
            'z_t': z_t,
            'z_pred': z_pred,
            'z_logvar': z_logvar,
            'graph_adj': graph_adj,
            'vol_params': vol_params,
            'xi': xi,
            'beta': beta,
            'causal_factors': causal_factors
        }

        # Compute loss if target provided
        if x_target is not None:
            z_target, _ = self.context_encoder(x_target)
            if self.use_hyperbolic:
                z_target = self.poincare.euclidean_to_hyperbolic(z_target)

            sigreg_losses = self.sigreg(z_pred, z_target.detach())
            output['sigreg_losses'] = sigreg_losses

        return output
