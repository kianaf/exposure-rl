"""Small dependency-free NumPy array helpers shared across packages."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return a numerically-stable softmax of ``logits`` along ``axis``."""

    shifted = np.asarray(logits) - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
