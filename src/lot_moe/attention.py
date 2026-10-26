"""Dual-Curvature Attention: Hyperbolic Geodesic Attention + Euclidean Dot-Product."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from lot_moe.lorentz import LorentzMath


class DualCurvatureAttention(nn.Module):
    """Dual-Curvature Attention Layer:
    - H_hyp heads compute non-Euclidean geodesic attention on Lorentz manifold H^{d_k, -c}.
    - H_eucl heads compute standard scaled dot-product attention in R^{d_k}.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        c: float = 1.0,
        beta: float = 1.0,
        hyperbolic_ratio: float = 0.5,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.c = c
        self.beta = beta

        self.num_hyp_heads = int(num_heads * hyperbolic_ratio)
        self.num_euc_heads = num_heads - self.num_hyp_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, S, D = x.shape
        H, d_k = self.num_heads, self.d_k

        q = self.q_proj(x).view(B, S, H, d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, d_k).transpose(1, 2)

        attn_outputs = []

        # 1. Hyperbolic Lorentz Heads
        if self.num_hyp_heads > 0:
            q_h = q[:, :self.num_hyp_heads]
            k_h = k[:, :self.num_hyp_heads]
            v_h = v[:, :self.num_hyp_heads]

            inv_c = 1.0 / self.c
            # Cancellation-free Lorentzian inner product: -c<q,k>_L = 1 + c/2 (||q-k||^2 - (q0-k0)^2)
            # Evaluated in fp32 accumulation to prevent catastrophic cancellation in half/bfloat16
            qf, kf = q_h.float(), k_h.float()
            qn = (qf * qf).sum(-1, keepdim=True)
            kn = (kf * kf).sum(-1, keepdim=True)
            q0 = torch.sqrt(inv_c + qn)
            k0 = torch.sqrt(inv_c + kn)
            d_sp = (qn + kn.transpose(-2, -1) - 2 * torch.matmul(qf, kf.transpose(-2, -1))).clamp_min(0.0)
            d_0 = (qn - kn.transpose(-2, -1)) / (q0 + k0.transpose(-2, -1))
            neg_c_inner = (1.0 + 0.5 * self.c * (d_sp - d_0 ** 2)).clamp_min(1.0 + 1e-6)

            dist_sq = (torch.acosh(neg_c_inner) ** 2) / self.c
            hyp_logits = (-self.beta * dist_sq).to(q_h.dtype)

            if causal:
                mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
                hyp_logits = hyp_logits + mask.unsqueeze(0).unsqueeze(0)

            hyp_weights = F.softmax(hyp_logits, dim=-1)
            out_hyp = torch.matmul(hyp_weights, v_h)
            attn_outputs.append(out_hyp)

        # 2. Euclidean Heads
        if self.num_euc_heads > 0:
            q_e = q[:, self.num_hyp_heads:]
            k_e = k[:, self.num_hyp_heads:]
            v_e = v[:, self.num_hyp_heads:]

            euc_logits = torch.matmul(q_e, k_e.transpose(-2, -1)) / math.sqrt(d_k)
            if causal:
                mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
                euc_logits = euc_logits + mask.unsqueeze(0).unsqueeze(0)

            euc_weights = F.softmax(euc_logits, dim=-1)
            out_euc = torch.matmul(euc_weights, v_e)
            attn_outputs.append(out_euc)

        full_out = torch.cat(attn_outputs, dim=1)
        full_out = full_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(full_out)

