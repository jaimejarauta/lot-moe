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
