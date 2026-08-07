"""
Product Manifold Embedding: H^n x S^k x R^m
=============================================
Embed financial representations on a product of spaces with mixed curvature:
  - Lorentz hyperboloid H^n: market cap hierarchies, credit quality trees
  - Sphere S^k: business cycle rotation, sector rotation, seasonality
  - Euclidean R^m: yield levels, spread magnitudes, linear trends

Uses Lorentz model (NOT Poincare ball) for numerical stability:
  no 1/(1-||x||^2)^2 gradient blowup, no clamp to max_norm.
  Euclidean parametrization per Mishne et al. ICML 2023.

References:
  Gu, Sala et al. "Learning Mixed-Curvature Representations" ICLR 2019
  Nickel & Kiela "Poincare Embeddings" NeurIPS 2017
  Mishne et al. ICML 2023 (Lorentz Euclidean parametrization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class LorentzManifold(nn.Module):
    """
    Operations on the Lorentz hyperboloid H^n.
    H^n = {x in R^{n+1} : <x,x>_L = -1, x_0 > 0}
    Lorentz inner product: <x,y>_L = -x_0*y_0 + sum_{i>0} x_i*y_i
    """

    def __init__(self, dim: int, curvature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.ambient_dim = dim + 1
        self.c = nn.Parameter(torch.tensor(curvature), requires_grad=False)

    def lorentz_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Minkowski inner product: -x_0*y_0 + x_1*y_1 + ..."""
        xy = x * y
        xy[..., 0] = -xy[..., 0]
        return xy.sum(-1)

    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project ambient vector to H^n by adjusting x_0."""
        spatial = x[..., 1:]
        x_0 = (1.0 + (spatial * spatial).sum(-1, keepdim=True)).sqrt()
        return torch.cat([x_0, spatial], dim=-1)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: tangent vector at x -> point on H^n."""
        v_norm = self._tangent_norm(v).clamp(min=1e-7).unsqueeze(-1)
        return torch.cosh(v_norm) * x + torch.sinh(v_norm) * v / v_norm

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: point y -> tangent vector at x."""
        neg_ip = -self.lorentz_inner(x, y).clamp(max=-1.0 - 1e-7)
        dist = torch.acosh(neg_ip).unsqueeze(-1)
        direction = y + neg_ip.unsqueeze(-1) * x
        direction_norm = self._tangent_norm(direction).clamp(min=1e-7).unsqueeze(-1)
        return dist * direction / direction_norm

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        neg_ip = -self.lorentz_inner(x, y)
        return torch.acosh(neg_ip.clamp(min=1.0 + 1e-7))

    def _tangent_norm(self, v: torch.Tensor) -> torch.Tensor:
        return self.lorentz_inner(v, v).clamp(min=1e-8).sqrt()

    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project v onto tangent space at x."""
        return v + self.lorentz_inner(x, v).unsqueeze(-1) * x

    def origin(self, *shape, device=None) -> torch.Tensor:
        """Origin point on H^n: (1, 0, 0, ..., 0)."""
        o = torch.zeros(*shape, self.ambient_dim, device=device)
        o[..., 0] = 1.0
        return o


class SphericalManifold(nn.Module):
    """
    Operations on the unit sphere S^k embedded in R^{k+1}.
    Natural geometry for cyclical patterns.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ambient_dim = dim + 1

    def project_to_sphere(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        return torch.cos(v_norm) * x + torch.sin(v_norm) * v / v_norm

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos_angle = (x * y).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle)
        direction = y - cos_angle * x
        direction = F.normalize(direction, p=2, dim=-1)
        return angle * direction

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos_angle = (x * y).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos_angle)

    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v - (x * v).sum(-1, keepdim=True) * x

    def origin(self, *shape, device=None) -> torch.Tensor:
        """North pole: (1, 0, 0, ..., 0)."""
        o = torch.zeros(*shape, self.ambient_dim, device=device)
        o[..., 0] = 1.0
        return o


class MetricTensorNet(nn.Module):
    """
    Learned metric tensor that outputs local curvature based on market state.
    Curvature varies by regime: high curvature in crisis (tighter hierarchy),
    lower curvature in calm (flatter structure).
    """

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (kappa_h, kappa_s): curvature scalars for hyperbolic and spherical.
        kappa_h in (0.1, 10), kappa_s in (0.1, 10).
        """
        out = self.net(context)
        kappa_h = 0.1 + 9.9 * torch.sigmoid(out[..., 0])
        kappa_s = 0.1 + 9.9 * torch.sigmoid(out[..., 1])
        return kappa_h, kappa_s


class ProductManifold(nn.Module):
    """
    Product manifold embedding: M = H^{h_dim} x S^{s_dim} x R^{e_dim}.

    Input: Euclidean vectors from sheaf encoder (common_dim per asset).
    Output: points on the product manifold.

    For downstream processing (RG, RSSM), the log map sends manifold
    points back to the tangent space at the origin, which IS Euclidean.
    """

    def __init__(self, input_dim: int, h_dim: int = 64, s_dim: int = 32,
                 e_dim: int = 32, use_metric_tensor: bool = True):
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.tangent_dim = h_dim + s_dim + e_dim

        self.hyperbolic = LorentzManifold(h_dim)
        self.spherical = SphericalManifold(s_dim)

        self.proj_h = nn.Linear(input_dim, h_dim)
        self.proj_s = nn.Linear(input_dim, s_dim + 1)
        self.proj_e = nn.Linear(input_dim, e_dim)

        self.use_metric_tensor = use_metric_tensor
        if use_metric_tensor:
            self.metric = MetricTensorNet(input_dim)

    def embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (..., input_dim) — Euclidean vector from sheaf encoder.
        Returns: (h, s, e) — points on H^n, S^k, R^m respectively.
        """
        h_tangent = self.proj_h(x)
        origin_h = self.hyperbolic.origin(*x.shape[:-1], device=x.device)
        tangent_h = torch.cat([
            torch.zeros(*x.shape[:-1], 1, device=x.device),
            h_tangent
        ], dim=-1)
        tangent_h = self.hyperbolic.project_to_tangent(origin_h, tangent_h)
        h_point = self.hyperbolic.exp_map(origin_h, tangent_h)

        s_raw = self.proj_s(x)
        s_point = self.spherical.project_to_sphere(s_raw)

        e_point = self.proj_e(x)

        return h_point, s_point, e_point

    def to_tangent(self, h: torch.Tensor, s: torch.Tensor,
                   e: torch.Tensor) -> torch.Tensor:
        """
        Map product manifold points back to tangent space at origin.
        Returns a single Euclidean vector suitable for RG/RSSM input.
        """
        origin_h = self.hyperbolic.origin(*h.shape[:-1], device=h.device)
        origin_s = self.spherical.origin(*s.shape[:-1], device=s.device)

        v_h = self.hyperbolic.log_map(origin_h, h)
        v_h = v_h[..., 1:]

        v_s = self.spherical.log_map(origin_s, s)
        v_s = v_s[..., 1:]

        return torch.cat([v_h, v_s, e], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_assets, input_dim) or (batch, seq_len, n_assets, input_dim)
        returns: tangent space vector (batch, ..., tangent_dim)
        """
        h, s, e = self.embed(x)
        return self.to_tangent(h, s, e)

    def manifold_distance(self, x1: torch.Tensor,
                          x2: torch.Tensor) -> torch.Tensor:
        """Distance on product manifold between two input vectors."""
        h1, s1, e1 = self.embed(x1)
        h2, s2, e2 = self.embed(x2)

        d_h = self.hyperbolic.distance(h1, h2)
        d_s = self.spherical.distance(s1, s2)
        d_e = (e1 - e2).norm(dim=-1)

        return (d_h ** 2 + d_s ** 2 + d_e ** 2).sqrt()

    def curvature_diagnostic(self, x: torch.Tensor
                             ) -> Dict[str, torch.Tensor]:
        """Returns diagnostic info about the manifold embedding."""
        h, s, e = self.embed(x)

        origin_h = self.hyperbolic.origin(*h.shape[:-1], device=h.device)
        h_dist = self.hyperbolic.distance(origin_h, h)

        result = {
            'h_dist_from_origin': h_dist,
            'e_norm': e.norm(dim=-1),
        }

        if self.use_metric_tensor:
            kappa_h, kappa_s = self.metric(x)
            result['kappa_h'] = kappa_h
            result['kappa_s'] = kappa_s

        return result
