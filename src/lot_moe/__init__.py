"""Lorentzian Optimal Transport Mixture of Experts (LOT-MoE).
Resolving Routing Collapse, Metric Distortion, and Residual Saturation in Transformers.
"""

from lot_moe.lorentz import LorentzMath
from lot_moe.attention import DualCurvatureAttention
from lot_moe.router import LogSinkhornRouter
from lot_moe.residual import ConformalGeodesicResidual
from lot_moe.block import ExpertFFN, LOTMoEBlock
from lot_moe.model import LOTMoETransformer

__all__ = [
    "LorentzMath",
    "DualCurvatureAttention",
    "LogSinkhornRouter",
    "ConformalGeodesicResidual",
    "ExpertFFN",
    "LOTMoEBlock",
    "LOTMoETransformer",
]
