# LOT-MoE: Lorentzian Optimal Transport Mixture of Experts

Resolving Routing Collapse, Metric Distortion, and Residual Saturation in Frontier Transformers.

## Core Breakthrough
Overcomes three structural failure modes of modern Mixture-of-Experts and ultra-deep Transformers:
1. **Zero Auxiliary Loss Load Balancing ($\mathcal{L}_{\text{aux}} \equiv 0$)**: Replaces heuristic softmax gating and disruptive auxiliary loss tug-of-wars with exact entropic Optimal Transport solved via log-domain Sinkhorn-Knopp iterations. Mathematically guarantees uniform $1/E$ expert capacity allocation without degrading task loss.
2. **Dual-Curvature Lorentz Attention**: Combines Euclidean attention heads with non-Euclidean geodesic attention on the Lorentz hyperboloid $H^{d, -c}$, eliminating the exponential tree volume bottleneck when representing hierarchical token semantics and syntax DAGs.
3. **Conformal Geodesic Residuals**: Replaces unconstrained Euclidean addition with conformal updates squashed via $\tanh$, bounding representation norm and preventing gradient shattering across 100+ layers.

## Key Invariants
- Minkowski Inner Product: $\langle u, v \rangle_{\mathcal{L}} = -u_0 v_0 + \langle u_{\text{sp}}, v_{\text{sp}} \rangle$
- Geodesic Distance: $d_{\mathbb{H}}^2(u, v) = \frac{1}{c}\operatorname{arcosh}^2(-c \langle u, v \rangle_{\mathcal{L}})$
- Optimal Transport Invariant: $\sum_j P_{ij} = \frac{1}{N}$, $\sum_i P_{ij} = \frac{1}{E}$
- Auxiliary Balancing Loss: $\mathcal{L}_{\text{aux}} \equiv 0.0$

## Quickstart
```bash
python scripts/run_smoke.py --steps 150 --num-experts 4 --top-k 2
pytest tests/
```

