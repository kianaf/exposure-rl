"""Rollout-collected background state datasets for attribution baselines.

Background datasets are the reference distribution marginal (interventional)
Shapley methods marginalize over. Unlike the probe-state collectors in
``xrl.training``, these collectors keep duplicates — repeated observations
encode visitation frequency, which is exactly the occupancy weighting a
rollout-based background should carry — and record raw environment
observations rather than encoded state keys, so attributions stay in
environment feature space regardless of the agent's encoder.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xrl.training.interfaces import PolicyView


@dataclass(frozen=True, slots=True)
class BackgroundDataset:
    """Observations visited by rollouts, with metadata for cohort selection."""

    env_id: str
    observations: np.ndarray
    action_masks: np.ndarray
    move_numbers: np.ndarray
    episode_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.observations.ndim != 2:
            raise ValueError(f"observations must be 2-D (states, features), got shape {self.observations.shape}")
        rows = self.observations.shape[0]
        for name in ("action_masks", "move_numbers", "episode_ids"):
            value = getattr(self, name)
            if value.shape[0] != rows:
                raise ValueError(f"{name} must have {rows} rows, got {value.shape[0]}")

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            env_id=np.asarray(self.env_id),
            observations=self.observations,
            action_masks=self.action_masks,
            move_numbers=self.move_numbers,
            episode_ids=self.episode_ids,
        )

    @classmethod
    def load(cls, path: Path) -> BackgroundDataset:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                env_id=str(data["env_id"]),
                observations=np.asarray(data["observations"], dtype=np.float32),
                action_masks=np.asarray(data["action_masks"], dtype=np.int8),
                move_numbers=np.asarray(data["move_numbers"], dtype=np.int32),
                episode_ids=np.asarray(data["episode_ids"], dtype=np.int32),
            )


def background_cache_path(run_dir: str | Path, *, policy: str, seed: int, states: int) -> Path:
    """Return the canonical cache location for an auto-collected background."""

    return Path(run_dir) / "explain" / f"background_{policy}_s{seed}_n{states}.npz"


def resolve_background(
    adapter: Any,
    policy: PolicyView | None,
    config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    background_path: Path | None = None,
    states: int = 256,
    seed: int = 1234,
    policy_name: str = "greedy",
    epsilon: float = 0.1,
    save: bool = False,
) -> tuple[BackgroundDataset, str]:
    """Load, reuse, or collect a background for a run; returns (dataset, source).

    ``adapter`` is an ``EnvExplanationAdapter`` (duck-typed here to avoid a
    circular import with :mod:`xrl.explainability.adapters`). The cache naming
    is shared by the CLI and the browser UI, so either can reuse the other's
    collected background.
    """

    if background_path is not None:
        return BackgroundDataset.load(background_path), str(background_path)
    cache = background_cache_path(run_dir, policy=policy_name, seed=seed, states=states)
    if cache.exists():
        return BackgroundDataset.load(cache), str(cache)
    dataset = adapter.collect_background(
        policy,
        config,
        states=states,
        seed=seed,
        policy_epsilon=epsilon if policy_name == "epsilon" else 0.0,
        random_policy=policy_name == "random",
    )
    if save:
        dataset.save(cache)
        return dataset, str(cache)
    return dataset, "collected"


def collect_gymnasium_background(
    make_env: Callable[[], Any],
    policy: PolicyView | None,
    *,
    env_id: str,
    states: int,
    seed: int,
    policy_epsilon: float = 0.0,
    random_policy: bool = False,
) -> BackgroundDataset:
    """Collect pre-action observations from single-agent Gymnasium rollouts.

    Actions are sampled from ``policy.action_distribution`` (greedy when
    ``policy_epsilon`` is 0) unless ``random_policy`` is set or ``policy`` is
    ``None``, in which case every move is uniform over legal actions. The
    episode budget is capped at ``10 * states`` so short episodes yield smaller
    datasets instead of looping forever.
    """

    if states < 1:
        raise ValueError(f"states must be positive, got {states}")
    rows = _Rows()
    rng = np.random.default_rng(seed)
    for index in range(10 * states):
        if rows.count >= states:
            break
        env = make_env()
        try:
            observation, _info = env.reset(seed=seed + index)
            move = 0
            while rows.count < states:
                obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
                mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
                rows.add(obs_vector, mask, move, index)
                action = _select_action(
                    policy,
                    obs_vector,
                    mask,
                    rng=rng,
                    policy_epsilon=policy_epsilon,
                    random_policy=random_policy,
                )
                if action is None:
                    break
                observation, _reward, terminated, truncated, _info = env.step(action)
                move += 1
                if terminated or truncated:
                    break
        finally:
            env.close()
    return rows.build(env_id)


def collect_aec_background(
    make_env: Callable[[], Any],
    policy: PolicyView | None,
    *,
    env_id: str,
    states: int,
    seed: int,
    policy_epsilon: float = 0.0,
    random_policy: bool = False,
) -> BackgroundDataset:
    """Collect pre-action observations from AEC self-play rollouts.

    Every seat plays by the same rule (the policy, or uniform random when
    ``random_policy`` is set), and every non-terminal turn's observation is
    recorded — each is a decision state in the mover's own perspective, which
    is the frame explainers operate in. Budget and sampling semantics match
    :func:`collect_gymnasium_background`.
    """

    if states < 1:
        raise ValueError(f"states must be positive, got {states}")
    rows = _Rows()
    rng = np.random.default_rng(seed)
    for index in range(10 * states):
        if rows.count >= states:
            break
        env = make_env()
        try:
            env.reset(seed=seed + index)
            move = 0
            for _agent_id in env.agent_iter():
                observation, _reward, terminated, truncated, _info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue
                obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
                mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
                if rows.count < states:
                    rows.add(obs_vector, mask, move, index)
                action = _select_action(
                    policy,
                    obs_vector,
                    mask,
                    rng=rng,
                    policy_epsilon=policy_epsilon,
                    random_policy=random_policy,
                )
                if action is None:
                    env.step(None)
                    continue
                env.step(action)
                move += 1
        finally:
            env.close()
    return rows.build(env_id)


@dataclass(slots=True)
class _Rows:
    """Accumulates background rows before array conversion."""

    observations: list[np.ndarray] = field(default_factory=list)
    action_masks: list[np.ndarray] = field(default_factory=list)
    move_numbers: list[int] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.observations)

    def add(self, observation: np.ndarray, mask: np.ndarray, move: int, episode: int) -> None:
        self.observations.append(observation)
        self.action_masks.append(mask)
        self.move_numbers.append(move)
        self.episode_ids.append(episode)

    def build(self, env_id: str) -> BackgroundDataset:
        if not self.observations:
            raise ValueError("no background states collected; check the environment and seed")
        return BackgroundDataset(
            env_id=env_id,
            observations=np.stack(self.observations).astype(np.float32),
            action_masks=np.stack(self.action_masks).astype(np.int8),
            move_numbers=np.asarray(self.move_numbers, dtype=np.int32),
            episode_ids=np.asarray(self.episode_ids, dtype=np.int32),
        )


def _select_action(
    policy: PolicyView | None,
    observation: np.ndarray,
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    policy_epsilon: float,
    random_policy: bool,
) -> int | None:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return None
    if random_policy or policy is None:
        return int(rng.choice(legal))
    probabilities = np.asarray(policy.action_distribution(observation, mask, epsilon=policy_epsilon), dtype=np.float64)
    total = probabilities.sum()
    if total <= 0.0:
        return int(rng.choice(legal))
    return int(rng.choice(probabilities.size, p=probabilities / total))
