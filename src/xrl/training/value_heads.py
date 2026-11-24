"""The vocabulary for how a Q-function represents a value.

A value head is the output layer's *representation* of ``Q(s, a)``: either a
single scalar, or a probability distribution over a fixed support of value
bins. The loss follows from that choice — mean squared error for a scalar,
cross-entropy against a projected target for a categorical head — which is why
this axis is named for the representation rather than for the loss.

This is deliberately independent of the learning algorithm. ``--algo`` selects
the function approximator and the update mechanics (online TD versus replay
buffer plus target network); the value head selects what the network emits.
Only DQN currently offers a choice, but nothing about a categorical head is
specific to replay.
"""

from __future__ import annotations

__all__ = [
    "CATEGORICAL_VALUE_HEADS",
    "SCALAR",
    "VALUE_HEADS",
    "is_categorical",
    "normalize_value_head",
]

SCALAR = "scalar"
TWO_HOT = "two-hot"
HL_GAUSS = "hl-gauss"

VALUE_HEADS = (SCALAR, TWO_HOT, HL_GAUSS)

# Heads that emit a distribution over support bins. These are the only runs
# that can be distilled into a masked distributional surrogate.
CATEGORICAL_VALUE_HEADS = (TWO_HOT, HL_GAUSS)

# Names used before the value head became its own concept, when this axis was
# called ``loss`` and mixed underscores with hyphens. Saved runs still carry
# them, and a trained model is data we do not get to invalidate by renaming a
# flag, so they are accepted on read and normalized here.
_LEGACY_ALIASES = {
    "regression": SCALAR,
    "two_hot": TWO_HOT,
    "hl_gauss": HL_GAUSS,
}


def normalize_value_head(name: str) -> str:
    """Return the canonical value-head name, accepting legacy spellings.

    Raises ``ValueError`` naming the valid options when ``name`` is neither.
    """

    canonical = _LEGACY_ALIASES.get(name, name)
    if canonical not in VALUE_HEADS:
        raise ValueError(f"value_head must be one of {', '.join(VALUE_HEADS)}, got {name!r}")
    return canonical


def is_categorical(name: str) -> bool:
    """Whether this head emits a distribution over value bins rather than a scalar."""

    return normalize_value_head(name) in CATEGORICAL_VALUE_HEADS
