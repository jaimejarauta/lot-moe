"""Full Stack LOT-MoE Transformer for Language Modeling and Reasoning."""

from typing import List, Tuple
import torch
import torch.nn as nn

from lot_moe.block import LOTMoEBlock


class LOTMoETransformer(nn.Module):
    """Complete Deep Reasoning Transformer featuring LOT-MoE blocks."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        max_seq_len: int = 512,
        c: float = 1.0,
        beta: float = 1.0,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            LOTMoEBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_experts=num_experts,
                top_k=top_k,
                c=c,
                beta=beta,
                epsilon=epsilon,
            ) for _ in range(num_layers)
        ])

        NormCls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.final_norm = NormCls(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, causal: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)

        x = self.tok_embeddings(input_ids) + self.pos_embeddings(pos)
        transport_plans = []

        for layer in self.layers:
            x, P = layer(x, causal=causal)
            transport_plans.append(P)

        logits = self.lm_head(self.final_norm(x))
        return logits, transport_plans
