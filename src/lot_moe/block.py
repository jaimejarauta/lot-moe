"""LOT-MoE Transformer Block integrating Dual-Curvature Attention,
Log-Sinkhorn Optimal Transport Routing, and Conformal Geodesic Residuals.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from lot_moe.attention import DualCurvatureAttention
from lot_moe.router import LogSinkhornRouter
from lot_moe.residual import ConformalGeodesicResidual


class ExpertFFN(nn.Module):
    """SwiGLU Expert Feed-Forward Network."""
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LOTMoEBlock(nn.Module):
    """Unified LOT-MoE Transformer Block."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int,
        top_k: int = 2,
        d_ffn: Optional[int] = None,
        c: float = 1.0,
        beta: float = 1.0,
        epsilon: float = 0.05,
        max_step: float = 0.1,
    ):
        super().__init__()
        d_ffn = d_ffn or int(4 * d_model * 2 / 3)
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        NormCls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm

        # Sublayer 1: Dual-Curvature Attention
        self.attn_norm = NormCls(d_model)
        self.attn = DualCurvatureAttention(d_model, num_heads, c=c, beta=beta)
        self.res1 = ConformalGeodesicResidual(d_model, max_step=max_step)

        # Sublayer 2: Log-Sinkhorn MoE
        self.moe_norm = NormCls(d_model)
        self.router = LogSinkhornRouter(d_model, num_experts, top_k=top_k, epsilon=epsilon)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ffn) for _ in range(num_experts)
        ])
        self.res2 = ConformalGeodesicResidual(d_model, max_step=max_step)

    def forward(self, x: torch.Tensor, causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape

        # 1. Attention
        attn_out = self.attn(self.attn_norm(x), causal=causal)
        x = self.res1(x, attn_out)

        # 2. Routing
        norm_x = self.moe_norm(x)
        flat_x = norm_x.view(B * S, D)
        topk_weights, topk_indices, P_dense = self.router(flat_x, is_training=self.training)

        # 3. Sparse Expert Dispatch
        flat_moe_out = torch.zeros_like(flat_x)
        for e_idx in range(self.num_experts):
            for k in range(self.top_k):
                mask = (topk_indices[:, k] == e_idx)
                if mask.any():
                    tokens_for_expert = flat_x[mask]
                    weights = topk_weights[mask, k:k+1]
                    expert_out = self.experts[e_idx](tokens_for_expert)
                    flat_moe_out[mask] = flat_moe_out[mask] + weights * expert_out

        moe_out = flat_moe_out.view(B, S, D)
        x = self.res2(x, moe_out)
        return x, P_dense

