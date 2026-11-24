"""Coalition masks and masked-input helpers for distributional Q surrogates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from xrl._masking import CoalitionMaskSampler, CoalitionMaskSamplerConfig, masked_inputs


class MaskedQValueModel(Protocol):
    """Small surface needed by the exact surrogate explainer."""

    num_actions: int
    support: Sequence[float]

    def q_values(
        self, observation: np.ndarray | Iterable[int | float], known_mask: np.ndarray | Iterable[int]
    ) -> np.ndarray:
        """Return expected Q-values for every action under one coalition."""


def masked_surrogate_inputs(states: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
    """Encode masked states as ``[visible_values, known_mask]``.

    Unknown cells use value ``0`` in the visible-value channel, but they remain
    distinguishable from known empty cells because the known-mask channel is
    concatenated to the input.
    """

    return masked_inputs(states, known_masks)


__all__ = [
    "CoalitionMaskSampler",
    "CoalitionMaskSamplerConfig",
    "MaskedQValueModel",
    "masked_surrogate_inputs",
]
