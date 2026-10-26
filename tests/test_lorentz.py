"""Unit tests for Lorentz geometry mathematics on H^{d, -c}."""

import pytest
import torch
from lot_moe.lorentz import LorentzMath


def test_spatial_to_lorentz_invariant():
    c = 1.0
    x_sp = torch.randn(4, 16, 32)
    x_lor = LorentzMath.spatial_to_lorentz(x_sp, c=c)

    # Invariant: -x_0^2 + ||x_sp||^2 == -1/c
    x_0 = x_lor[..., 0]
    norm_sp_sq = torch.sum(x_lor[..., 1:] ** 2, dim=-1)
    lorentz_norm_sq = - (x_0 ** 2) + norm_sp_sq

    expected = -1.0 / c
    assert torch.allclose(lorentz_norm_sq, torch.full_like(lorentz_norm_sq, expected), atol=1e-5)


def test_minkowski_inner_properties():
    u = torch.randn(2, 5, 8)
    # Self-inner product of time-like vector
    u_lor = LorentzMath.spatial_to_lorentz(u, c=2.0)
    inner = LorentzMath.minkowski_inner(u_lor, u_lor)
    assert torch.allclose(inner, torch.full_like(inner, -0.5), atol=1e-5)


def test_lorentz_distance_metric():
    c = 1.0
    u_sp = torch.randn(4, 8)
    u = LorentzMath.spatial_to_lorentz(u_sp, c=c)

    # Distance to itself must be zero
    dist_self = LorentzMath.lorentz_distance_sq(u, u, c=c)
    assert torch.allclose(dist_self, torch.zeros_like(dist_self), atol=1e-4)

    # Distance to different vector must be positive
    v_sp = torch.randn(4, 8)
    v = LorentzMath.spatial_to_lorentz(v_sp, c=c)
    dist_uv = LorentzMath.lorentz_distance_sq(u, v, c=c)
    assert (dist_uv >= 0.0).all()
