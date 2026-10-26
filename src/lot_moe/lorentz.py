"""Numerically robust pseudo-Riemannian Minkowski spacetime operations
for the hyperboloid manifold H^{d, -c}.
"""

import torch
import torch.nn.functional as F


class LorentzMath:
    """Operations on the Lorentz / Hyperboloid model H^{d, -c} = { x in R^{d+1} : <x, x>_L = -1/c, x_0 > 0 }."""

    @staticmethod
    def spatial_to_lorentz(x_sp: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Lifts spatial coordinates R^d to the Lorentz hyperboloid H^{d, -c} in R^{d+1}.
        Strict invariant: <x, x>_L = -x_0^2 + ||x_sp||^2 == -1/c.
        Shape: (..., d) -> (..., d + 1)
        """
        inv_c = 1.0 / c
        norm_sq = torch.sum(x_sp ** 2, dim=-1, keepdim=True)
        x_0 = torch.sqrt(inv_c + norm_sq)
        return torch.cat([x_0, x_sp], dim=-1)

    @staticmethod
    def minkowski_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Minkowski bilinear inner product: <u, v>_L = -u_0 * v_0 + <u_sp, v_sp>.
        Shape: (..., d+1), (..., d+1) -> (..., 1)
        """
        time_part = -u[..., 0:1] * v[..., 0:1]
        space_part = torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        return time_part + space_part

    @staticmethod
    def lorentz_distance_sq(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Squared geodesic distance on H^{d, -c}:
        d_H^2(u, v) = (1/c) * arcosh^2( clamp( -c * <u, v>_L, min=1.0 + eps ) )
        """
        inner = LorentzMath.minkowski_inner(u, v)
        neg_c_inner = torch.clamp(-c * inner, min=1.0 + 1e-6)
        dist = torch.acosh(neg_c_inner) / (c ** 0.5)
        return dist ** 2
