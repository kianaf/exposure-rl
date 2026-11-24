"""Categorical value-support projection helpers for neural Q-learning."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from xrl._optional import jax, jnp

DEFAULT_V_MIN = -1.0
DEFAULT_V_MAX = 1.0
DEFAULT_NUM_BINS = 51
DEFAULT_HL_GAUSS_SIGMA_BINS = 0.75
_SIGMA_FLOOR = 1e-6
_PROBABILITY_FLOOR = 1e-8


def support_values(
    v_min: float = DEFAULT_V_MIN,
    v_max: float = DEFAULT_V_MAX,
    num_bins: int = DEFAULT_NUM_BINS,
) -> np.ndarray:
    """Return categorical bin centers used for projection and expected-value decoding."""

    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if v_max <= v_min:
        raise ValueError("v_max must be greater than v_min")
    return np.linspace(v_min, v_max, num_bins, dtype=np.float32)


def two_hot_projection(values: np.ndarray | Iterable[float], support: np.ndarray) -> np.ndarray:
    """Project scalar targets onto adjacent support bins."""

    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    support = np.asarray(support, dtype=np.float32).reshape(-1)
    clipped = np.clip(values_array, support[0], support[-1])
    width = float(support[1] - support[0])
    positions = (clipped - support[0]) / width
    lower = np.floor(positions).astype(np.int32)
    upper = np.ceil(positions).astype(np.int32)
    lower = np.clip(lower, 0, len(support) - 1)
    upper = np.clip(upper, 0, len(support) - 1)

    probs = np.zeros((len(values_array), len(support)), dtype=np.float32)
    upper_weight = positions - lower
    lower_weight = 1.0 - upper_weight
    rows = np.arange(len(values_array))
    probs[rows, lower] += lower_weight
    probs[rows, upper] += upper_weight
    return probs


def hl_gauss_projection(
    values: np.ndarray | Iterable[float],
    support: np.ndarray,
    *,
    sigma_bins: float = DEFAULT_HL_GAUSS_SIGMA_BINS,
) -> np.ndarray:
    """Project scalar targets to Gaussian-smoothed probabilities over support bin centers."""

    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    support = np.asarray(support, dtype=np.float32).reshape(-1)
    width = float(support[1] - support[0])
    sigma = max(float(sigma_bins) * width, _SIGMA_FLOOR)
    clipped = np.clip(values_array, support[0], support[-1])
    edges = np.concatenate(
        (
            [support[0] - width / 2],
            (support[:-1] + support[1:]) / 2,
            [support[-1] + width / 2],
        )
    ).astype(np.float32)
    z = (edges[None, :] - clipped[:, None]) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    probs = np.diff(cdf, axis=1).astype(np.float32)
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.maximum(row_sums, _PROBABILITY_FLOOR)


def project_values_jax(values, support, method: str, sigma_bins: float):
    """JAX version of two-hot / HL-Gauss projection for jit-ed train steps."""

    if jax is None or jnp is None:
        raise RuntimeError("JAX projection requires the optional 'neural' extra")

    if method == "two-hot":
        clipped = jnp.clip(values, support[0], support[-1])
        width = support[1] - support[0]
        positions = (clipped - support[0]) / width
        lower = jnp.clip(jnp.floor(positions).astype(jnp.int32), 0, support.shape[0] - 1)
        upper = jnp.clip(jnp.ceil(positions).astype(jnp.int32), 0, support.shape[0] - 1)
        upper_weight = positions - lower
        lower_weight = 1.0 - upper_weight
        probs = jnp.zeros((values.shape[0], support.shape[0]), dtype=jnp.float32)
        rows = jnp.arange(values.shape[0])
        probs = probs.at[rows, lower].add(lower_weight)
        probs = probs.at[rows, upper].add(upper_weight)
        return probs

    if method == "hl-gauss":
        width = support[1] - support[0]
        sigma = jnp.maximum(sigma_bins * width, _SIGMA_FLOOR)
        clipped = jnp.clip(values, support[0], support[-1])
        edges = jnp.concatenate(
            (
                jnp.asarray([support[0] - width / 2], dtype=jnp.float32),
                (support[:-1] + support[1:]) / 2,
                jnp.asarray([support[-1] + width / 2], dtype=jnp.float32),
            )
        )
        z = (edges[None, :] - clipped[:, None]) / sigma
        cdf = 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))
        probs = jnp.diff(cdf, axis=1)
        return probs / jnp.maximum(jnp.sum(probs, axis=1, keepdims=True), _PROBABILITY_FLOOR)

    raise ValueError(f"unknown classification method {method!r}")


# Backward-compatible private name used inside older neural_q internals.
_project_values_jax = project_values_jax
