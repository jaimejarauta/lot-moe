#!/usr/bin/env python3
"""Stage 2 Tree-Metric Distortion Benchmark: Lorentz Attention vs Flat Euclidean Attention.
Evaluates how accurately each geometry embeds hierarchical reasoning DAGs and syntax trees.
Metric: Distortion D = (1 / pairs) * sum_{i < j} |d_attn(i, j) - d_tree(i, j)| / d_tree(i, j).
Saves provenance manifest to results/tree_distortion_manifest.json.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
import torch
import torch.nn.functional as F

from lot_moe.lorentz import LorentzMath
from lot_moe.attention import DualCurvatureAttention


def build_binary_tree_graph(depth: int = 5):
    """Constructs a binary tree graph and computes all-pairs shortest path tree distances."""
    num_nodes = (2 ** (depth + 1)) - 1
    dist_matrix = torch.full((num_nodes, num_nodes), float("inf"))
    for i in range(num_nodes):
        dist_matrix[i, i] = 0.0

    for i in range(num_nodes):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < num_nodes:
            dist_matrix[i, left] = 1.0
            dist_matrix[left, i] = 1.0
        if right < num_nodes:
            dist_matrix[i, right] = 1.0
            dist_matrix[right, i] = 1.0

    # Floyd-Warshall algorithm for exact tree geodesic distances
    for k in range(num_nodes):
        dist_matrix = torch.minimum(dist_matrix, dist_matrix[:, k:k+1] + dist_matrix[k:k+1, :])

    return num_nodes, dist_matrix


def main():
    parser = argparse.ArgumentParser(description="Tree Metric Distortion Benchmark")
    parser.add_argument("--tree-depth", type=int, default=5, help="Binary tree depth (depth 5 = 63 nodes)")
    parser.add_argument("--d-model", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps to fit tree metric")
    args = parser.parse_args()

    device = torch.device("cpu")
    num_nodes, tree_dist = build_binary_tree_graph(args.tree_depth)

    print("=" * 76)
    print("STAGE 2: TREE-METRIC GEOMETRIC DISTORTION BENCHMARK")
    print(f"Tree Nodes: {num_nodes} (Depth {args.tree_depth}) | Latent Dimension: {args.d_model}")
    print("=" * 76)

    # 1. Optimize Flat Euclidean Embeddings to approximate tree distance: d_euc(u, v) = ||u - v||_2
    emb_euc = torch.nn.Parameter(torch.randn(num_nodes, args.d_model) * 0.1)
    opt_euc = torch.optim.Adam([emb_euc], lr=0.05)

    for _ in range(args.steps):
        diff = emb_euc.unsqueeze(1) - emb_euc.unsqueeze(0)  # (N, N, D)
        pred_dist = torch.norm(diff, p=2, dim=-1)
        loss = F.mse_loss(pred_dist, tree_dist)
        opt_euc.zero_grad()
        loss.backward()
        opt_euc.step()

    with torch.no_grad():
        diff = emb_euc.unsqueeze(1) - emb_euc.unsqueeze(0)
        euc_dist = torch.norm(diff, p=2, dim=-1)
        mask = torch.triu(torch.ones(num_nodes, num_nodes, dtype=torch.bool), diagonal=1)
        euc_distortion = (torch.abs(euc_dist[mask] - tree_dist[mask]) / tree_dist[mask]).mean().item()

    # 2. Optimize Lorentz Hyperbolic Embeddings to approximate tree distance
    c = 1.0
    emb_lor_sp = torch.nn.Parameter(torch.randn(num_nodes, args.d_model) * 0.1)
    opt_lor = torch.optim.Adam([emb_lor_sp], lr=0.05)

    for _ in range(args.steps):
        lor_points = LorentzMath.spatial_to_lorentz(emb_lor_sp, c=c)
        # Compute pairwise Lorentz distance: d_H = (1/√c) * arcosh(-c <u, v>_L)
        time_m = torch.matmul(lor_points[:, 0:1], lor_points[:, 0:1].t())
        space_m = torch.matmul(lor_points[:, 1:], lor_points[:, 1:].t())
        minkowski = -time_m + space_m
        neg_c_minkowski = torch.clamp(-c * minkowski, min=1.0 + 1e-6)
        pred_lor_dist = torch.acosh(neg_c_minkowski) / (c ** 0.5)

        loss = F.mse_loss(pred_lor_dist, tree_dist)
        opt_lor.zero_grad()
        loss.backward()
        opt_lor.step()

    with torch.no_grad():
        lor_points = LorentzMath.spatial_to_lorentz(emb_lor_sp, c=c)
        time_m = torch.matmul(lor_points[:, 0:1], lor_points[:, 0:1].t())
        space_m = torch.matmul(lor_points[:, 1:], lor_points[:, 1:].t())
        minkowski = -time_m + space_m
        neg_c_minkowski = torch.clamp(-c * minkowski, min=1.0 + 1e-6)
        lor_dist = torch.acosh(neg_c_minkowski) / (c ** 0.5)
        lor_distortion = (torch.abs(lor_dist[mask] - tree_dist[mask]) / tree_dist[mask]).mean().item()

    print(f"\n{'Geometry':<26} | {'Target Structure':<20} | {'Mean Metric Distortion D':<24}")
    print("-" * 76)
    print(f"{'Flat Euclidean Space (R^d)':<26} | {'Binary Tree Graph':<20} | {euc_distortion:<24.4f}")
    print(f"{'Lorentz Hyperboloid (H^d)':<26} | {'Binary Tree Graph':<20} | {lor_distortion:<24.4f}")
    print("-" * 76)

    distortion_reduction = (euc_distortion - lor_distortion) / max(1e-5, euc_distortion)
    gate_pass = lor_distortion < euc_distortion * 0.70

    print(f"Metric Distortion Reduction: {distortion_reduction * 100:.1f}%")
    print(f"Pre-Registered Gate (Lorentz Distortion < 70% Euclidean): {'PASS' if gate_pass else 'FAIL'}")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    manifest_path = os.path.join(results_dir, "tree_distortion_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "benchmark": "tree_metric_distortion",
            "date": datetime.now(timezone.utc).isoformat(),
            "nodes": num_nodes,
            "euclidean_distortion": euc_distortion,
            "lorentz_distortion": lor_distortion,
            "distortion_reduction_pct": distortion_reduction * 100.0,
            "gate_passed": gate_pass,
        }, f, indent=2)

    print(f"Manifest saved to {manifest_path}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())

