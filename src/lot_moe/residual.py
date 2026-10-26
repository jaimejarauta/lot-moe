"""Conformal Geodesic Residual Connections.
Bounds residual update magnitude, preventing representation explosion and gradient shattering.
"""

import torch
import torch.nn as nn


class ConformalGeodesicResidual(nn.Module):
    """Conformal Geodesic Residual:
    Squashes update vectors via a hyperbolic tangent metric, strictly bounding norm drift.
    """
    def __init__(self, d_model: int, max_step: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.max_step = max_step
        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """x: (B, S, D), delta: (B, S, D)"""
        norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True).clamp_min(1e-7)
        # Cap is relative to ||x||; an absolute 0.1 cap froze training (+0.5 nats, see git log)
        cap = self.max_step * torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-7)
        step = cap * torch.tanh(norm_delta / cap)
        clamped_delta = (step / norm_delta) * delta
        return self.norm(x + clamped_delta)

