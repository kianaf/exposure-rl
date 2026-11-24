"""Centralized imports and guards for the optional ``neural`` extra (jax/flax/optax).

Modules that need the neural stack import the symbols from here (directly or re-exported via
``xrl.training.neural_q``) instead of repeating the ``try/except ModuleNotFoundError`` dance.
When the extra is not installed, the symbols are ``None`` and :func:`require_neural_dependencies`
raises :class:`MissingNeuralDependenciesError`.

The optional ``explain`` extra (shap/scikit-learn) is guarded here too, but only by presence
checks — shap is heavy to import, so consumers import it lazily inside functions after calling
:func:`require_explain_dependencies`.

The optional ``storage`` extra (pyarrow) backs every persisted study table. It follows the neural
pattern rather than the explain one: the symbols are imported eagerly and are ``None`` when the
extra is absent, because the writers need the module object itself, not just a presence check.
"""

from __future__ import annotations

from importlib.util import find_spec

try:  # Optional neural dependency group.
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax import serialization
except ModuleNotFoundError:  # pragma: no cover - exercised by importing without the extra.
    jax = None
    jnp = None
    optax = None
    nn = None
    serialization = None

try:  # Optional storage dependency group.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - exercised by importing without the extra.
    pa = None
    pq = None


class MissingNeuralDependenciesError(RuntimeError):
    """Raised when an operation needs the optional 'neural' extra and it is not installed.

    Subclasses :class:`RuntimeError` for backward compatibility with callers that catch
    ``RuntimeError``, while letting newer callers catch the specific type.
    """


def neural_dependencies_available() -> bool:
    """Return whether the optional neural stack (jax/flax/optax) is importable."""

    return None not in (jax, jnp, nn, optax, serialization)


def require_neural_dependencies(feature: str = "This feature") -> None:
    """Raise :class:`MissingNeuralDependenciesError` if the neural stack is missing.

    Args:
        feature: Human-readable name used in the error message.
    """

    if not neural_dependencies_available():
        raise MissingNeuralDependenciesError(f"{feature} requires the optional 'neural' extra: install xrl[neural]")


class MissingStorageDependenciesError(RuntimeError):
    """Raised when an operation needs the optional 'storage' extra and it is not installed."""


def storage_dependencies_available() -> bool:
    """Return whether the optional storage stack (pyarrow) is importable."""

    return None not in (pa, pq)


def require_storage_dependencies(feature: str = "This feature") -> None:
    """Raise :class:`MissingStorageDependenciesError` if pyarrow is missing.

    Args:
        feature: Human-readable name used in the error message.
    """

    if not storage_dependencies_available():
        raise MissingStorageDependenciesError(f"{feature} requires the optional 'storage' extra: install xrl[storage]")


class MissingExplainDependenciesError(RuntimeError):
    """Raised when an operation needs the optional 'explain' extra and it is not installed."""


def explain_dependencies_available() -> bool:
    """Return whether the optional explain stack (shap/scikit-learn) is importable."""

    return find_spec("shap") is not None and find_spec("sklearn") is not None


def require_explain_dependencies(feature: str = "This feature") -> None:
    """Raise :class:`MissingExplainDependenciesError` if the explain stack is missing.

    Args:
        feature: Human-readable name used in the error message.
    """

    if not explain_dependencies_available():
        raise MissingExplainDependenciesError(f"{feature} requires the optional 'explain' extra: install xrl[explain]")
