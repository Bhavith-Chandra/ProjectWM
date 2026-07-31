"""Meridian core: SSM belief state + JEPA prediction + SIGReg regularization.

Architecture
------------
  x_{t-L+1:t}  --(diagonal SSM belief core)-->  belief state h_t
       |                                              |
       |                                     +--------+--------+
       |                                     |                 |
   JEPA predictor g(h_t) -> z_hat_{t+1}   vol head        regime head
                                          log RV_{t+1}    K-state logits

Objectives (jointly trained):
  * supervised volatility: MSE( vol_head(h_t), log RV_{t+1} )
  * JEPA latent prediction: || g(h_t) - sg(target_enc(future)) ||^2   (energy / surprise)
  * SIGReg: push embeddings toward isotropic Gaussian (anti-collapse, LeJEPA-style)

The JEPA "energy" (latent prediction error) is the live **surprise score**.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MeridianConfig:
    n_features: int = 11
    window: int = 32
    d_model: int = 64
    d_state: int = 64
    n_layers: int = 2
    n_regimes: int = 3
    dropout: float = 0.1
    lambda_jepa: float = 1.0
    lambda_sig: float = 0.5
    ema: float = 0.99
    seed: int = 0
    loss_mode: str = "mse"     # "mse" (head=E[log RV]) or "qlike" (head=log variance)
    dual_vol: bool = False     # CF-JEPA test: 2nd vol head reads the EMA target encoder
    core_type: str = "ssm"     # "ssm" (DiagonalSSM) or "ode" (Neural-ODE / ODE-RNN)


# --------------------------------------------------------------------------- #
class DiagonalSSM(nn.Module):
    """Diagonal linear state-space layer (S4/Mamba-lite).

    h_t = a ⊙ h_{t-1} + b ⊙ x_t ;  y_t = C h_t + D x_t, with a = sigmoid(logit)
    giving stable decay in (0,1). Runs as a sequential scan (windows are short).
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.in_proj = nn.Linear(d_model, d_state)
        self.a_logit = nn.Parameter(torch.linspace(1.0, 4.0, d_state))  # decay ~sigmoid
        self.b = nn.Parameter(torch.ones(d_state) * 0.5)
        self.out_proj = nn.Linear(d_state, d_model)
        self.skip = nn.Linear(d_model, d_model)

    def forward(self, x):  # x: (B, L, d_model)
        B, L, _ = x.shape
        a = torch.sigmoid(self.a_logit)               # (d_state,)
        u = self.in_proj(x) * self.b                   # (B, L, d_state)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = a * h + u[:, t]
            outs.append(h)
        h_seq = torch.stack(outs, dim=1)               # (B, L, d_state)
        return self.out_proj(h_seq) + self.skip(x)


class Encoder(nn.Module):
    """Feature projection + stacked SSM belief core. Returns full belief sequence."""

    def __init__(self, cfg: MeridianConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.n_features, cfg.d_model)
        self.layers = nn.ModuleList(
            [DiagonalSSM(cfg.d_model, cfg.d_state) for _ in range(cfg.n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)])
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.LayerNorm(cfg.d_model)

    def forward(self, x):                              # (B, L, F) -> (B, L, d_model)
        h = self.inp(x)
        for ssm, norm in zip(self.layers, self.norms):
            h = h + self.drop(ssm(norm(h)))
        return self.head(h)


class Predictor(nn.Module):
    """JEPA predictor: belief state -> predicted next-window embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, h):
        return self.net(h)


# --------------------------------------------------------------------------- #
def sigreg_loss(z: torch.Tensor, n_proj: int = 64) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (LeJEPA-style, practical form).

    Pushes the embedding distribution toward an isotropic Gaussian by matching,
    along `n_proj` random 1-D projections, the first four standardized moments
    to those of N(0,1). Prevents representation collapse without stop-grad.
    """
    z = z - z.mean(0, keepdim=True)
    d = z.shape[1]
    P = torch.randn(d, n_proj, device=z.device, dtype=z.dtype)
    P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
    p = z @ P                                          # (N, n_proj)
    p = p / (p.std(0, keepdim=True) + 1e-6)
    m2 = (p ** 2).mean(0)                              # ->1
    m3 = (p ** 3).mean(0)                              # ->0
    m4 = (p ** 4).mean(0)                              # ->3
    return ((m2 - 1) ** 2 + m3 ** 2 + (m4 - 3) ** 2).mean()


class Meridian(nn.Module):
    def __init__(self, cfg: MeridianConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        _Enc = Encoder
        if cfg.core_type == "ode":
            from .continuous import ODERNNEncoder as _Enc
        self.encoder = _Enc(cfg)
        self.target_encoder = _Enc(cfg)                  # EMA target (no grad)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg.d_model)
        self.vol_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        # CF-JEPA test: a second vol head that reads the EMA target encoder.
        # It is a probe on the smoothed representation (stop-grad to the encoder),
        # trained on the same QLIKE loss so we can compare the two read-offs.
        self.vol_head_tgt = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        ) if cfg.dual_vol else None
        self.regime_head = nn.Linear(cfg.d_model, cfg.n_regimes)

    @torch.no_grad()
    def update_target(self):
        m = self.cfg.ema
        for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.mul_(m).add_(p, alpha=1 - m)

    def belief(self, x):
        """Return last belief state h_t for a window x: (B, L, F) -> (B, d_model)."""
        return self.encoder(x)[:, -1]

    def forward(self, x_ctx, x_fut=None):
        """x_ctx: (B, L, F) context window ending at t.
        x_fut:  (B, L, F) window ending at t+1 (for JEPA target); optional.
        """
        h_seq = self.encoder(x_ctx)
        h_t = h_seq[:, -1]                               # belief state
        vol = self.vol_head(h_t).squeeze(-1)             # predicted log RV_{t+1}
        regime_logits = self.regime_head(h_t)

        out = {"h": h_t, "vol": vol, "regime_logits": regime_logits,
               "h_seq": h_seq}

        if self.cfg.dual_vol:
            # read a vol forecast off the EMA target encoder (stop-grad to encoder)
            with torch.no_grad():
                h_tgt = self.target_encoder(x_ctx)[:, -1]
            out["vol_tgt"] = self.vol_head_tgt(h_tgt).squeeze(-1)

        if x_fut is not None:
            z_pred = self.predictor(h_t)                 # predicted future embedding
            with torch.no_grad():
                z_tgt = self.target_encoder(x_fut)[:, -1]
            energy = (z_pred - z_tgt).pow(2).mean(-1)    # per-sample surprise
            out.update({"z_pred": z_pred, "z_tgt": z_tgt, "energy": energy})
        return out

    def loss(self, batch):
        cfg = self.cfg
        out = self.forward(batch["x_ctx"], batch["x_fut"])
        y = batch["y"]                       # log RV_{t+1}

        def vol_objective(f):
            if cfg.loss_mode == "qlike":
                # f is the log-variance forecast; minimize exact QLIKE.
                # QLIKE = RV/s2 - log(RV/s2) - 1 = RV*exp(-f) + f - log RV - 1
                rv = torch.exp(y)
                return (rv * torch.exp(-f) + f - y - 1.0).mean()
            return F.mse_loss(f, y)          # f is E[log RV]

        vol_loss = vol_objective(out["vol"])
        jepa_loss = out["energy"].mean()
        sig = sigreg_loss(out["h"])
        total = vol_loss + cfg.lambda_jepa * jepa_loss + cfg.lambda_sig * sig
        logs = {"total": float(total), "vol": float(vol_loss),
                "jepa": float(jepa_loss), "sig": float(sig)}
        if cfg.dual_vol:
            vol_loss_tgt = vol_objective(out["vol_tgt"])
            total = total + vol_loss_tgt     # trains the target-readout head only
            logs["vol_tgt"] = float(vol_loss_tgt)
        return total, logs
