"""Cohort selection over rollout background datasets.

Cohort-conditional Shapley narrows the reference distribution to background
states similar to the explained state (Mase-Owen-Seiler style), so masked
features are imputed from comparable game situations instead of the whole
visitation distribution.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from xrl.explainability.backgrounds import BackgroundDataset

CohortKeyFn = Callable[[np.ndarray], int]


def cohort_background(
    background: BackgroundDataset,
    observation: np.ndarray,
    key_fn: CohortKeyFn,
    *,
    min_size: int = 20,
) -> BackgroundDataset:
    """Restrict a background to rows whose cohort key is near the observation's.

    Starts from exact key matches and widens to the next-nearest keys by
    absolute key distance until at least ``min_size`` rows are selected. A
    background smaller than ``min_size`` is returned unchanged.
    """

    if min_size < 1:
        raise ValueError(f"min_size must be positive, got {min_size}")
    if background.size <= min_size:
        return background
    target = int(key_fn(np.asarray(observation)))
    keys = np.asarray([int(key_fn(row)) for row in background.observations], dtype=np.int64)
    distances = np.abs(keys - target)
    selected = np.zeros(background.size, dtype=bool)
    for distance in sorted(set(distances.tolist())):
        selected |= distances == distance
        if int(selected.sum()) >= min_size:
            break
    indices = np.flatnonzero(selected)
    return BackgroundDataset(
        env_id=background.env_id,
        observations=background.observations[indices],
        action_masks=background.action_masks[indices],
        move_numbers=background.move_numbers[indices],
        episode_ids=background.episode_ids[indices],
    )
