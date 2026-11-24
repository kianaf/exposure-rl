"""Tic-tac-toe opponent specifications shared by the oracle, TD training and exact evaluation.

``"random"``  uniformly random legal moves.
``"pm05"``    xrl's perturbed minimax at a 5% random-move rate (minimax otherwise), the evaluation
              opponent of Philipp's setup. ``pmNN`` gives NN% in general.

Training against perturbed minimax can anneal the random-move rate from 1 to the target rate, as the xrl
CLI does (``--opponent perturbed_minimax``); the oracle and all evaluation use the target rate.
"""

from __future__ import annotations

import re

import numpy as np
from xrl.explainability.tictactoe import opponent_action_distribution


def parse(spec: str) -> tuple[str, float | None]:
    """``(xrl opponent name, perturbed-minimax random-move rate or None)``."""

    if spec in ("random", "minimax"):
        return spec, None
    m = re.fullmatch(r"pm(\d+)", spec)
    if m is None:
        raise ValueError(f"unknown opponent spec {spec!r}")
    return "perturbed_minimax", int(m.group(1)) / 100.0


def distribution(spec: str, state, rules, rate: float | None = None) -> np.ndarray:
    """Opponent move probabilities; ``rate`` overrides the random-move rate (annealed training)."""

    name, p = parse(spec)
    if name == "perturbed_minimax":
        return opponent_action_distribution(
            name, state, perturbed_minimax_probability=p if rate is None else rate, rules=rules
        )
    return opponent_action_distribution(name, state, rules=rules)
