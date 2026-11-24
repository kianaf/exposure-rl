"""Shared functional episode driver for AEC training loops."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from xrl.training.metrics import ProbeSetBuilder, probe_q_stats


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Metrics captured for one episode."""

    episode: int
    opponent: str
    seed: int
    winner: str | None
    draw: bool
    learner_return: float
    episode_length: int
    epsilon: float
    illegal_moves: int
    elapsed_seconds: float
    td_error_mean: float = 0.0
    td_error_max: float = 0.0
    unique_states_seen: int = 0
    q_table_rows: int = 0
    gradient_updates: int = 0
    replay_size: int = 0
    win: int = 0
    steps_per_second: float = 0.0
    rolling_win_rate: float = 0.0
    rolling_draw_rate: float = 0.0
    rolling_learner_return: float = 0.0
    rolling_episode_length: float = 0.0
    opponent_random_move_probability: float = 0.0


@dataclass(frozen=True, slots=True)
class LearnerStep:
    """A learner decision in algorithm-native coordinates."""

    state: Any
    action: int
    env_action: int
    next_action_mask: Any


@dataclass(frozen=True, slots=True)
class LearnerStats:
    """Optional learner counters included in episode metrics."""

    unique_states_seen: int = 0
    q_table_rows: int = 0
    gradient_updates: int = 0
    replay_size: int = 0


SelectLearnerAction = Callable[[dict[str, Any], float, bool], LearnerStep]
ObserveLearnerTransition = Callable[[LearnerStep, float, LearnerStep | None, bool], float | None]
LearnerStatsFn = Callable[[], LearnerStats]


def choose_learner_id(seed: int) -> str:
    """Choose which PettingZoo agent id the learner controls for one episode."""

    rng = random.Random(seed)
    return "player_0" if rng.random() < 0.5 else "player_1"


def epsilon_for_episode(
    episode: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
) -> float:
    """Linearly decay epsilon over episodes."""

    if epsilon_decay_episodes <= 0:
        return epsilon_end
    fraction = min(max(episode - 1, 0) / epsilon_decay_episodes, 1.0)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


def winner_agent(winner: int | None) -> str | None:
    """Map core winner marks to PettingZoo agent ids."""

    if winner == 1:
        return "player_0"
    if winner == -1:
        return "player_1"
    return None


def run_aec_episode(
    *,
    env,
    opponent: object,
    episode: int,
    seed: int,
    epsilon: float,
    train: bool,
    select_learner_action: SelectLearnerAction,
    observe_learner_transition: ObserveLearnerTransition,
    learner_stats: LearnerStatsFn | None = None,
) -> EpisodeMetrics:
    """Run one AEC episode and delegate learner-specific work to callbacks."""

    env.reset(seed=seed)
    learner_id = choose_learner_id(seed)
    pending: dict[str, LearnerStep] = {}
    learner_return = 0.0
    episode_length = 0
    td_errors: list[float] = []
    started = time.perf_counter()

    try:
        for agent_id in env.agent_iter():
            observation, reward, terminated, truncated, _info = env.last()
            done = bool(terminated or truncated)
            is_learner = agent_id == learner_id

            if is_learner:
                learner_return += float(reward)

            if is_learner:
                if done:
                    if train and agent_id in pending:
                        td_error = observe_learner_transition(pending.pop(agent_id), float(reward), None, True)
                        if td_error is not None:
                            td_errors.append(td_error)
                    action = None
                else:
                    learner_step = select_learner_action(observation, epsilon if train else 0.0, train)
                    if train and agent_id in pending:
                        td_error = observe_learner_transition(pending[agent_id], float(reward), learner_step, False)
                        if td_error is not None:
                            td_errors.append(td_error)
                    pending[agent_id] = learner_step
                    action = learner_step.env_action
                    episode_length += 1
            else:
                if done:
                    action = None
                else:
                    action = opponent.select_action(
                        observation["observation"],
                        observation["action_mask"],
                        agent_id=agent_id,
                        epsilon=epsilon if train else 0.0,
                        env=env,
                    )
                    episode_length += 1

            # An illegal move from a policy is a bug; let the ValueError abort the episode loudly.
            env.step(action)

        stats = learner_stats() if learner_stats is not None else LearnerStats()
        elapsed_seconds = time.perf_counter() - started
        return EpisodeMetrics(
            episode=episode,
            opponent=getattr(opponent, "name", "unknown"),
            seed=seed,
            winner=winner_agent(env.unwrapped.state.winner),
            draw=bool(env.unwrapped.state.is_draw),
            learner_return=learner_return,
            episode_length=episode_length,
            epsilon=epsilon,
            illegal_moves=0,
            elapsed_seconds=elapsed_seconds,
            td_error_mean=sum(td_errors) / len(td_errors) if td_errors else 0.0,
            td_error_max=max(td_errors, default=0.0),
            unique_states_seen=stats.unique_states_seen,
            q_table_rows=stats.q_table_rows,
            gradient_updates=stats.gradient_updates,
            replay_size=stats.replay_size,
            win=int(learner_return > 0),
            steps_per_second=episode_length / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        )
    finally:
        env.close()


def collect_aec_probe_states(
    make_env: Callable[[], Any],
    learner: Any,
    *,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect distinct encoded observations from random-legal AEC rollouts.

    Runs episodes where every seat plays uniformly random legal moves, encoding
    each turn's observation through ``learner.encode_view`` and deduping via
    ``ProbeSetBuilder``. Episode seeds start at ``seed`` (use a base offset far
    from training/eval streams, see ``PROBE_SEED_OFFSET``); the episode budget
    is capped at ``10 * limit`` so tiny games yield smaller probe sets instead
    of looping forever.
    """

    if limit <= 0:
        return None
    builder = ProbeSetBuilder(limit)
    rng = np.random.default_rng(seed)
    for index in range(10 * limit):
        if builder.full:
            break
        env = make_env()
        try:
            env.reset(seed=seed + index)
            for _agent_id in env.agent_iter():
                observation, _reward, terminated, truncated, _info = env.last()
                if terminated or truncated:
                    env.step(None)
                    continue
                obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
                mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
                view = learner.encode_view(obs_vector, mask)
                builder.add(np.asarray(view.state_key, dtype=np.float32), np.asarray(view.action_mask))
                legal = np.flatnonzero(mask)
                if legal.size == 0:
                    env.step(None)
                    continue
                env.step(int(rng.choice(legal)))
        finally:
            env.close()
    return builder.build()


def aec_probe_hook(
    make_env: Callable[[], Any],
    *,
    limit: int,
    seed: int,
) -> Callable[[Any], dict[str, float]] | None:
    """Build a runner probe hook with a lazily collected, cached probe set.

    The probe set is collected on the first checkpoint and reused afterwards,
    so every later checkpoint scores the identical held-out states. Learners
    without a ``q_values_batch`` (e.g. the tabular agent) yield no stats.
    """

    if limit <= 0:
        return None
    cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def probe(learner: Any) -> dict[str, float]:
        q_values_batch = getattr(learner, "q_values_batch", None)
        if q_values_batch is None:
            return {}
        if "states" not in cache:
            cache["states"] = collect_aec_probe_states(make_env, learner, limit=limit, seed=seed)
        built = cache["states"]
        if built is None:
            return {}
        states, masks = built
        return probe_q_stats(q_values_batch(states), masks)

    return probe
