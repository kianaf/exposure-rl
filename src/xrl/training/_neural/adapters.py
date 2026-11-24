"""Shared learner-adapter helpers for the composable AEC runner."""

from __future__ import annotations

import numpy as np

from xrl.training.episode import LearnerStep, SelectLearnerAction


def neural_select_learner_action(learner) -> SelectLearnerAction:
    """Build the per-step action selector shared by the neural learner adapters.

    The closure flattens the observation, encodes it through the learner's
    encoder, picks an epsilon-greedy action, and returns a :class:`LearnerStep`
    in algorithm-native coordinates. The tabular adapter uses its own variant
    because it additionally tracks the set of unmasked states it has seen.
    """

    def select_learner_action(observation, step_epsilon: float, step_train: bool) -> LearnerStep:
        obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
        mask = observation["action_mask"]
        view = learner.encode_view(obs_vector, mask)
        env_action = learner.select_action(obs_vector, mask, epsilon=step_epsilon)
        return LearnerStep(
            state=np.asarray(view.state_key, dtype=np.float32),
            action=view.to_table_action(env_action),
            env_action=env_action,
            next_action_mask=np.asarray(view.action_mask, dtype=np.int8),
        )

    return select_learner_action
