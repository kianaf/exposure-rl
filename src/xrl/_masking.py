"""Shared masked-input encoding and coalition sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def masked_inputs(states: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
    """Encode states as ``[visible_values, known_mask]``.

    Unknown feature values are zero-filled in the visible channel.  The
    concatenated binary channel keeps an unknown zero distinct from a known
    feature whose observed value is actually zero.
    """

    states_array = np.asarray(states, dtype=np.float32)
    masks_array = np.asarray(known_masks, dtype=np.float32)
    if states_array.ndim == 1:
        states_array = states_array[None, :]
    if masks_array.ndim == 1:
        masks_array = masks_array[None, :]
    if states_array.shape != masks_array.shape:
        raise ValueError(
            f"states and known_masks must have the same shape, got {states_array.shape} and {masks_array.shape}"
        )
    visible = np.where(masks_array > 0.0, states_array, 0.0).astype(np.float32)
    return np.concatenate((visible, masks_array.astype(np.float32)), axis=1)


def _periodic_count(interval: int | None, example_offset: int, batch_size: int) -> int:
    """Count scheduled anchors inside a contiguous example batch.

    Counting examples instead of optimizer updates keeps the conditional mask
    law invariant when a caller changes its batch size or mask ratio.
    """

    if interval is None:
        return 0
    return (example_offset + batch_size) // interval - example_offset // interval


def validate_mask_mixture(
    *,
    shapley_probability: float,
    uniform_probability: float,
    empty_mask_example_interval: int | None = None,
    full_mask_example_interval: int | None = None,
) -> None:
    """Validate the coalition-mask knobs shared by every masked-Q trainer.

    Lives beside the sampler that owns the invariant, so a trainer config and
    the sampler cannot drift apart on what counts as a valid mixture.
    """

    for name, interval in (
        ("full_mask_example_interval", full_mask_example_interval),
        ("empty_mask_example_interval", empty_mask_example_interval),
    ):
        if interval is not None and interval < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(shapley_probability) or not math.isfinite(uniform_probability):
        raise ValueError("mask mixture probabilities must be finite")
    if shapley_probability < 0 or uniform_probability < 0:
        raise ValueError("mask mixture probabilities must be non-negative")
    if shapley_probability + uniform_probability <= 0:
        raise ValueError("at least one default mask sampler must have positive probability")


@dataclass(frozen=True, slots=True)
class CoalitionMaskSamplerConfig:
    """Coalition mask settings shared by every masked-Q trainer.

    Endpoint intervals count sampled mask examples. ``None`` disables the
    corresponding anchor.
    """

    feature_count: int = 9
    full_mask_example_interval: int | None = 512
    empty_mask_example_interval: int | None = 512
    shapley_probability: float = 0.7
    uniform_probability: float = 0.3
    guided_probability: float = 0.25
    guided_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.feature_count < 2:
            raise ValueError("feature_count must be at least 2")
        validate_mask_mixture(
            full_mask_example_interval=self.full_mask_example_interval,
            empty_mask_example_interval=self.empty_mask_example_interval,
            shapley_probability=self.shapley_probability,
            uniform_probability=self.uniform_probability,
        )
        if not 0 <= self.guided_probability <= 1:
            raise ValueError("guided_probability must be in [0, 1]")
        if self.guided_epsilon <= 0:
            raise ValueError("guided_epsilon must be positive")


class CoalitionMaskSampler:
    """Sample feature coalitions for masked-Q training and distillation.

    Every masked-Q arm draws from this one sampler. The arms are compared with
    each other, and that comparison only isolates the model when they share a
    coalition measure. See ADR 0026.
    """

    def __init__(self, config: CoalitionMaskSamplerConfig, *, seed: int = 0) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._features = tuple(range(config.feature_count))
        self._kernel_sizes = tuple(range(1, config.feature_count))
        kernel_weights = [1.0 / (size * (config.feature_count - size)) for size in self._kernel_sizes]
        total = sum(kernel_weights)
        self._kernel_probabilities = [weight / total for weight in kernel_weights]

    def sample_batch(
        self,
        batch_size: int,
        *,
        example_offset: int,
        attribution_magnitudes: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return a batch of binary known-feature masks."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if example_offset < 0:
            raise ValueError("example_offset must be non-negative")
        masks: list[np.ndarray] = []
        full_count = _periodic_count(self.config.full_mask_example_interval, example_offset, batch_size)
        empty_count = _periodic_count(self.config.empty_mask_example_interval, example_offset, batch_size)
        masks.extend(np.ones(self.config.feature_count, dtype=np.int8) for _ in range(full_count))
        masks.extend(np.zeros(self.config.feature_count, dtype=np.int8) for _ in range(empty_count))

        while len(masks) < batch_size:
            if (
                attribution_magnitudes is not None
                and len(masks) + 1 < batch_size
                and self._rng.random() < self.config.guided_probability
            ):
                selected, complement = self._guided_pair(attribution_magnitudes)
                masks.extend((selected, complement))
                continue
            masks.append(self._default_mask())
        return np.stack(masks[:batch_size]).astype(np.int8)

    def _default_mask(self) -> np.ndarray:
        total = self.config.shapley_probability + self.config.uniform_probability
        if self._rng.random() < self.config.shapley_probability / total:
            return self._shapley_kernel_mask()
        return self._uniform_size_mask()

    def _shapley_kernel_mask(self) -> np.ndarray:
        size = self._rng.choices(self._kernel_sizes, weights=self._kernel_probabilities, k=1)[0]
        return self._mask_from_indices(self._rng.sample(self._features, size))

    def _uniform_size_mask(self) -> np.ndarray:
        size = self._rng.randint(1, self.config.feature_count - 1)
        return self._mask_from_indices(self._rng.sample(self._features, size))

    def _guided_pair(self, attribution_magnitudes: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        size = self._rng.randint(1, self.config.feature_count - 1)
        weights = [abs(float(value)) + self.config.guided_epsilon for value in attribution_magnitudes]
        if len(weights) != self.config.feature_count:
            raise ValueError(f"expected {self.config.feature_count} attribution magnitudes, got {len(weights)}")
        selected = self._weighted_sample_without_replacement(weights, size)
        mask = self._mask_from_indices(selected)
        return mask, (1 - mask).astype(np.int8)

    def _weighted_sample_without_replacement(self, weights: Sequence[float], size: int) -> list[int]:
        available = list(self._features)
        available_weights = [max(float(weight), 0.0) for weight in weights]
        selected: list[int] = []
        for _ in range(size):
            total = sum(available_weights[index] for index in available)
            if total <= 0:
                choice = self._rng.choice(available)
            else:
                threshold = self._rng.random() * total
                running = 0.0
                choice = available[-1]
                for index in available:
                    running += available_weights[index]
                    if running >= threshold:
                        choice = index
                        break
            selected.append(choice)
            available.remove(choice)
        return selected

    def _mask_from_indices(self, indices: Iterable[int]) -> np.ndarray:
        mask = np.zeros(self.config.feature_count, dtype=np.int8)
        for index in indices:
            mask[int(index)] = 1
        return mask
