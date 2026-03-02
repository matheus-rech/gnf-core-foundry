"""Evaluation module — responsiveness and convergent validity."""

from .responsiveness import EffectSizeResult, compute_effect_size, compute_group_effect_size
from .convergent_validity import CorrelationResult, compute_correlation, compute_correlation_matrix

__all__ = [
    "EffectSizeResult",
    "compute_effect_size",
    "compute_group_effect_size",
    "CorrelationResult",
    "compute_correlation",
    "compute_correlation_matrix",
]
