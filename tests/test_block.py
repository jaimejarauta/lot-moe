"""Unit tests for DualCurvatureAttention and LOTMoEBlock."""

import pytest
import torch
from lot_moe.attention import DualCurvatureAttention
from lot_moe.block import LOTMoEBlock
from lot_moe.model import LOTMoETransformer


def test_dual_curvature_attention_shape():
    B, S, D, H = 2, 8, 32, 4
    attn = DualCurvatureAttention(d_model=D, num_heads=H, c=1.0, beta=1.0, hyperbolic_ratio=0.5)

    x = torch.randn(B, S, D)
    out = attn(x, causal=True)
    assert out.shape == (B, S, D)
    assert not torch.isnan(out).any()


def test_lot_moe_block_and_residual():
    B, S, D, H, E = 2, 6, 32, 4, 4
    block = LOTMoEBlock(d_model=D, num_heads=H, num_experts=E, top_k=2)

    x = torch.randn(B, S, D)
    out, P = block(x, causal=True)

    assert out.shape == (B, S, D)
    assert P.shape == (B * S, E)
    assert not torch.isnan(out).any()


def test_transformer_end_to_end_backprop():
    B, S, V, D, L = 2, 8, 64, 32, 2
    model = LOTMoETransformer(
        vocab_size=V,
        d_model=D,
        num_layers=L,
        num_heads=4,
        num_experts=4,
        top_k=2,
        max_seq_len=32,
    )

    tokens = torch.randint(0, V, (B, S))
    logits, plans = model(tokens)

    assert logits.shape == (B, S, V)
    assert len(plans) == L

    # Backward pass verification
    loss = logits.sum()
    loss.backward()

    # All parameters must receive valid gradients without NaNs
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN in gradient of {name}"



def test_block_training_routing_is_causal():
    from lot_moe.block import LOTMoEBlock
    torch.manual_seed(0)
    blk = LOTMoEBlock(d_model=32, num_heads=4, num_experts=4, top_k=2).train()
    x = torch.randn(2, 16, 32)
    x2 = x.clone()
    x2[:, 8:] = torch.randn(2, 8, 32)
    v0 = blk.router.amortized_v.clone()
    y1, _ = blk(x)
    blk.router.amortized_v.copy_(v0)  # the bias update is causal in time; reset it to isolate token coupling
    y2, _ = blk(x2)
    assert torch.allclose(y1[:, :8], y2[:, :8], atol=1e-5)


def test_residual_cap_scales_with_state_norm():
    from lot_moe.residual import ConformalGeodesicResidual
    res = ConformalGeodesicResidual(64)
    x = torch.randn(1, 4, 64) * 8.0
    delta = torch.randn(1, 4, 64) * 8.0
    raw = res.norm(x + delta)
    out = res(x, delta)
    # a delta of the same size as x must still move the output substantially
    assert (out - res.norm(x)).norm() > 0.3 * (raw - res.norm(x)).norm()
