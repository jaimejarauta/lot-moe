"""Unit tests for Log-Domain Sinkhorn-Knopp MoE Router."""

import pytest
import torch
from lot_moe.router import LogSinkhornRouter


def test_sinkhorn_birkhoff_marginals():
    N, D, E, K = 64, 32, 4, 2
    router = LogSinkhornRouter(d_model=D, num_experts=E, top_k=K, epsilon=0.05, num_iters=20)

    x = torch.randn(N, D)
    topk_weights, topk_indices, P = router(x, is_training=True)

    # Transport plan P shape: (N, E)
    assert P.shape == (N, E)

    # Row marginals: sum_j P_ij == 1/N
    row_sums = P.sum(dim=-1)
    expected_row = 1.0 / N
    assert torch.allclose(row_sums, torch.full_like(row_sums, expected_row), atol=1e-4)

    # Column marginals: sum_i P_ij == 1/E (Uniform load guarantee within entropic tolerance)
    col_sums = P.sum(dim=0)
    expected_col = 1.0 / E
    assert torch.allclose(col_sums, torch.full_like(col_sums, expected_col), atol=0.035)

    # Top-K weights sum to 1.0 per token
    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(N), atol=1e-5)


def test_amortized_inference_mode():
    N, D, E, K = 8, 16, 4, 2
    router = LogSinkhornRouter(d_model=D, num_experts=E, top_k=K)

    # Train step updates amortized_v
    x = torch.randn(N, D)
    router(x, is_training=True)

    # Eval mode uses amortized_v for single-token O(1) inference
    router.eval()
    x_single = torch.randn(1, D)
    topk_w, topk_idx, P_single = router(x_single, is_training=False)

    assert topk_w.shape == (1, K)
    assert topk_idx.shape == (1, K)
    assert torch.allclose(topk_w.sum(), torch.tensor(1.0), atol=1e-5)


def test_causal_routing_ignores_other_tokens():
    torch.manual_seed(0)
    router = LogSinkhornRouter(d_model=16, num_experts=4, top_k=2)
    x = torch.randn(64, 16)
    _, idx_a, _ = router(x, is_training=False, causal=True)
    x2 = x.clone()
    x2[32:] = torch.randn(32, 16)
    _, idx_b, _ = router(x2, is_training=False, causal=True)
    assert torch.equal(idx_a[:32], idx_b[:32])


def test_causal_topk_weights_not_saturated():
    torch.manual_seed(0)
    router = LogSinkhornRouter(d_model=64, num_experts=8, top_k=2)
    torch.nn.init.normal_(router.gate.weight, std=0.1)
    w, _, _ = router(torch.randn(2048, 64), is_training=True, causal=True)
    assert (w[:, 1] < 1e-3).float().mean() < 0.01


def test_causal_bias_balances_dispatch():
    torch.manual_seed(0)
    router = LogSinkhornRouter(d_model=16, num_experts=4, top_k=1, ema_gamma=0.1)
    with torch.no_grad():
        router.gate.weight[0] += 2.0  # skew toward expert 0
    x = torch.randn(1024, 16).abs()
    for _ in range(200):
        router(x, is_training=True, causal=True)
    _, idx, _ = router(x, is_training=False, causal=True)
    load = torch.bincount(idx.view(-1), minlength=4).float() / idx.numel()
    assert load.max() < 0.40
