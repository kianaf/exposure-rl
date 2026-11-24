"""Exact local SVERL explainers for Tic Tac Toe."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from threading import RLock
from typing import Any

import numpy as np

from xrl.explainability.core import (
    CachedCharacteristicFunction,
    Coalition,
    ExactShapleyEstimator,
    ExplanationResult,
    ExplanationSemantics,
    FeatureSpec,
    validate_action,
)
from xrl.tictactoe.agents import MinimaxAgent
from xrl.tictactoe.core import GameState, O, PlayerMark, TicTacToeRules, X
from xrl.training.interfaces import PolicyView

CONDITIONAL_OCCUPANCY_ASSUMPTION = (
    "unknown features are marginalized over the conditional learner-decision occupancy distribution"
)


@dataclass(frozen=True, slots=True)
class TicTacToeDistributionPoint:
    """One weighted learner-decision state in the exact Tic Tac Toe occupancy distribution."""

    state: GameState
    learner_mark: PlayerMark
    observation: tuple[int, ...]
    action_mask: tuple[int, ...]
    weight: float

    @property
    def observation_array(self) -> np.ndarray:
        return np.asarray(self.observation, dtype=np.int8)

    @property
    def action_mask_array(self) -> np.ndarray:
        return np.asarray(self.action_mask, dtype=np.int8)


class TicTacToeStateDistribution:
    """Normalized occupancy over learner-decision states."""

    def __init__(self, points: Iterable[TicTacToeDistributionPoint], *, features: FeatureSpec):
        self.features = features
        raw_points = list(points)
        total = sum(point.weight for point in raw_points)
        if total <= 0:
            self.points = tuple(raw_points)
        else:
            self.points = tuple(
                TicTacToeDistributionPoint(
                    state=point.state,
                    learner_mark=point.learner_mark,
                    observation=point.observation,
                    action_mask=point.action_mask,
                    weight=point.weight / total,
                )
                for point in raw_points
            )
        self._matching_cache: dict[
            tuple[tuple[int | float, ...], tuple[int, ...]],
            tuple[tuple[TicTacToeDistributionPoint, ...], float],
        ] = {}
        self._matching_cache_lock = RLock()
        self._action_views: dict[int, TicTacToeStateDistribution] = {}
        self._support_arrays_cache: tuple[np.ndarray, np.ndarray] | None = None

    def legal_action_view(self, action: int) -> TicTacToeStateDistribution:
        """Return this occupancy conditioned on ``action`` being legal.

        The fixed-action Q oracle conditions on ``{X_C = x_C} and {a legal}``,
        a strictly finer event than the one :meth:`matching_points` memoizes.
        Keying that memo on the coarser event would admit an illegal donor, so
        the legality filter is part of the memo's *identity* -- one view per
        action -- rather than part of its key. See ADR 0028.
        """

        with self._matching_cache_lock:
            view = self._action_views.get(action)
            if view is not None:
                return view
        legal = tuple(point for point in self.points if point.action_mask[action])
        if sum(point.weight for point in legal) <= 0:
            raise ValueError(f"no learner-decision occupancy states make action {action} legal")
        built = TicTacToeStateDistribution(legal, features=self.features)
        with self._matching_cache_lock:
            return self._action_views.setdefault(action, built)

    def matching_points(
        self,
        target_features: Sequence[int | float],
        coalition: Coalition | Iterable[int],
    ) -> tuple[tuple[TicTacToeDistributionPoint, ...], float]:
        key = tuple(sorted(coalition))
        target_key = tuple(target_features[index] for index in key)
        cache_key = (target_key, key)
        with self._matching_cache_lock:
            cached = self._matching_cache.get(cache_key)
            if cached is not None:
                return cached
        matches = tuple(
            point for point in self.points if self.features.matches(point.observation, target_features, key)
        )
        result = (matches, sum(point.weight for point in matches))
        with self._matching_cache_lock:
            return self._matching_cache.setdefault(cache_key, result)

    def _support_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Observations and weights of this view's support, built once."""

        with self._matching_cache_lock:
            cached = self._support_arrays_cache
            if cached is None:
                observations = np.asarray([point.observation for point in self.points], dtype=np.float64).reshape(
                    len(self.points), -1
                )
                weights = np.asarray([point.weight for point in self.points], dtype=np.float64)
                cached = (observations, weights)
                self._support_arrays_cache = cached
            return cached

    def conditional_expectation_operator(self, target_features: Sequence[int | float]) -> np.ndarray:
        """Return the row-stochastic ``(2^n, N)`` operator for every coalition at once.

        Conditioning on ``{X_C = x_C}`` is linear in the quantity being
        averaged: ``v(C) = sum_p w_p 1{p_C = x_C} q_p / sum_p w_p 1{p_C = x_C}``.
        Stacking the coalitions turns the whole game into one matrix, so all
        ``2^n`` values are a single matrix-vector product rather than ``2^n``
        Python weighted sums over the support.

        Rows follow :func:`all_coalitions` order, which is bitmask order --
        feature ``i`` is bit ``i`` -- and that is what lets the match test be a
        bitwise one: coalition ``c`` matches point ``p`` exactly when every bit
        of ``c`` is set in ``p``'s agreement mask.
        """

        observations, weights = self._support_arrays()
        feature_count = observations.shape[1]
        target = np.asarray(target_features, dtype=np.float64).reshape(-1)
        if target.shape[0] != feature_count:
            raise ValueError(f"expected {feature_count} target features, got {target.shape[0]}")

        bit = (np.uint32(1) << np.arange(feature_count, dtype=np.uint32))[None, :]
        agreement = ((observations == target[None, :]) * bit).sum(axis=1, dtype=np.uint32)
        coalitions = np.arange(1 << feature_count, dtype=np.uint32)
        matches = (coalitions[:, None] & ~agreement[None, :]) == 0

        operator = matches * weights[None, :]
        totals = operator.sum(axis=1)
        # The grand coalition is answered from the model, not the support, so a
        # target outside the reference is not an error there. Every interior
        # coalition must be supported, and saying so here keeps a declared-
        # estimand failure from becoming a silent NaN.
        if np.any(totals[:-1] <= 0):
            raise ValueError(
                "conditional event has zero reference support; "
                "choose a supported state/reference pair instead of changing the estimand"
            )
        totals[totals <= 0] = 1.0
        return operator / totals[:, None]

    def conditional_expectation(
        self,
        target_features: Sequence[int | float],
        coalition: Coalition,
        value_fn: Callable[[TicTacToeDistributionPoint], float],
    ) -> float:
        matches, total = self.matching_points(target_features, coalition)
        if total <= 0:
            raise ValueError(
                "conditional event has zero reference support; "
                "choose a supported state/reference pair instead of changing the estimand"
            )
        return float(sum((point.weight / total) * value_fn(point) for point in matches))


class TicTacToeSVERLExplainer:
    """Exact local SVERL implementation for flat Tic Tac Toe observations."""

    def __init__(
        self,
        policy: PolicyView,
        *,
        opponent: str = "random",
        perturbed_minimax_probability: float | None = None,
        distribution: TicTacToeStateDistribution | None = None,
        rules: TicTacToeRules | None = None,
        estimator: ExactShapleyEstimator | None = None,
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.policy = policy
        self.opponent = opponent
        self.perturbed_minimax_probability = perturbed_minimax_probability
        self.rules = rules or TicTacToeRules()
        self.features = tictactoe_feature_spec()
        self.estimator = estimator or ExactShapleyEstimator(parallel=parallel, max_workers=max_workers)
        self.distribution = distribution or build_tictactoe_state_distribution(
            policy,
            opponent=opponent,
            perturbed_minimax_probability=perturbed_minimax_probability,
            rules=self.rules,
            features=self.features,
        )
        self._policy_distribution_cache: dict[tuple[tuple[int | float, ...], tuple[int, ...]], np.ndarray] = {}
        self._policy_state_value_cache: dict[tuple[tuple[int | float, ...], tuple[int, ...]], float] = {}
        self._policy_q_values_cache: dict[tuple[int | float, ...], np.ndarray] = {}
        self._policy_cache_lock = RLock()

    def explain_behavior(
        self,
        state: GameState,
        *,
        action: int,
        learner_mark: PlayerMark | None = None,
    ) -> ExplanationResult:
        observation, action_mask, _mark = self._target_view(state, learner_mark)
        target_action = self._validate_action(action, action_mask)
        full = float(self._policy_action_distribution(observation, action_mask)[target_action])
        characteristic = CachedCharacteristicFunction(
            self.features.feature_count,
            lambda coalition: self._conditional_scalar(
                observation,
                action_mask,
                coalition,
                full,
                lambda point: float(self._point_action_distribution(point)[target_action]),
            ),
        )
        return self._result("behavior", state, target_action, characteristic)

    def explain_prediction(
        self,
        state: GameState,
        *,
        learner_mark: PlayerMark | None = None,
    ) -> ExplanationResult:
        observation, action_mask, _mark = self._target_learner_turn_view(state, learner_mark)
        full = float(self._policy_state_value(observation, action_mask))
        characteristic = CachedCharacteristicFunction(
            self.features.feature_count,
            lambda coalition: self._conditional_scalar(
                observation,
                action_mask,
                coalition,
                full,
                lambda point: float(self._point_state_value(point)),
            ),
        )
        return self._result("prediction", state, None, characteristic)

    def explain_outcome(
        self,
        state: GameState,
        *,
        learner_mark: PlayerMark | None = None,
    ) -> ExplanationResult:
        observation, action_mask, mark = self._target_learner_turn_view(state, learner_mark)
        continuation_returns = self._action_continuation_returns(state, mark)
        full_distribution = self._policy_action_distribution(observation, action_mask)
        full = float(np.dot(full_distribution, continuation_returns))

        def value_for(coalition: Coalition) -> float:
            if len(coalition) == self.features.feature_count:
                return full
            partial_distribution = self._partial_action_distribution(observation, action_mask, coalition)
            return float(np.dot(partial_distribution, continuation_returns))

        characteristic = CachedCharacteristicFunction(self.features.feature_count, value_for)
        return self._result("outcome", state, None, characteristic)

    def q_value_characteristic(
        self,
        state: GameState,
        *,
        action: int,
        learner_mark: PlayerMark | None = None,
    ) -> CachedCharacteristicFunction:
        """Return the action-legal conditional-occupancy game for ``Q(s, a)``."""

        observation, action_mask, _mark = self._target_learner_turn_view(state, learner_mark)
        target_action = self._validate_action(action, action_mask)
        full = float(self._policy_q_values(observation)[target_action])
        view = self.distribution.legal_action_view(target_action)
        target_features = tuple(observation.tolist())

        def point_value(point: TicTacToeDistributionPoint) -> float:
            return float(self._point_q_values(point)[target_action])

        def value_for(coalition: Coalition) -> float:
            if len(coalition) == self.features.feature_count:
                return full
            return view.conditional_expectation(target_features, coalition, point_value)

        return CachedCharacteristicFunction(self.features.feature_count, value_for)

    def q_value_coalition_values(
        self,
        state: GameState,
        *,
        action: int,
        learner_mark: PlayerMark | None = None,
    ) -> np.ndarray:
        """Every coalition value of the fixed-action Q game, in one product.

        The dense sibling of :meth:`q_value_characteristic`: same estimand, same
        coalition order as :func:`all_coalitions`, but evaluated as
        ``W(x) @ q`` -- the conditioning operator times the support's Q column
        for ``action`` -- instead of one Python weighted sum per coalition.
        Callers that need only a handful of coalitions still want the
        characteristic function.
        """

        observation, action_mask, _mark = self._target_learner_turn_view(state, learner_mark)
        target_action = self._validate_action(action, action_mask)
        view = self.distribution.legal_action_view(target_action)
        operator = view.conditional_expectation_operator(tuple(observation.tolist()))
        support_q = np.asarray(
            [self._point_q_values(point)[target_action] for point in view.points],
            dtype=np.float64,
        )
        values = operator @ support_q
        # The grand coalition is the model's own answer, not an average over the
        # reference; it is the endpoint the fidelity metrics anchor on.
        values[-1] = float(self._policy_q_values(observation)[target_action])
        return values

    def _validate_action(self, action: int, action_mask: Sequence[int] | np.ndarray) -> int:
        return validate_action(action, action_mask, num_actions=int(self.rules.num_actions))

    def explain_q_value(
        self,
        state: GameState,
        *,
        action: int,
        learner_mark: PlayerMark | None = None,
    ) -> ExplanationResult:
        """Explain one legal fixed-action Q-value under conditional occupancy."""

        characteristic = self.q_value_characteristic(
            state,
            action=action,
            learner_mark=learner_mark,
        )
        return self._result("q_value", state, int(action), characteristic)

    def _target_view(
        self,
        state: GameState,
        learner_mark: PlayerMark | None,
    ) -> tuple[np.ndarray, np.ndarray, PlayerMark]:
        if state.is_terminal:
            raise ValueError("SVERL explanations require a non-terminal state")
        mark = state.current_player if learner_mark is None else learner_mark
        if mark not in (X, O):
            raise ValueError(f"learner_mark must be 1 or -1, got {mark}")
        observation = self.rules.observation(state, mark, "flat")
        return (
            np.asarray(observation["observation"], dtype=np.int8).reshape(-1),
            np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1),
            mark,
        )

    def _target_learner_turn_view(
        self,
        state: GameState,
        learner_mark: PlayerMark | None,
    ) -> tuple[np.ndarray, np.ndarray, PlayerMark]:
        observation, action_mask, mark = self._target_view(state, learner_mark)
        if mark != state.current_player:
            raise ValueError("SVERL explanations require a learner-decision state")
        return observation, action_mask, mark

    def _conditional_scalar(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        coalition: Coalition,
        full: float,
        value_fn: Callable[[TicTacToeDistributionPoint], float],
    ) -> float:
        del action_mask
        if len(coalition) == self.features.feature_count:
            return full
        return self.distribution.conditional_expectation(tuple(observation.tolist()), coalition, value_fn)

    def _partial_action_distribution(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        coalition: Coalition,
    ) -> np.ndarray:
        if len(coalition) == self.features.feature_count:
            probabilities = self._policy_action_distribution(observation, action_mask)
        else:
            matches, total = self.distribution.matching_points(tuple(observation.tolist()), coalition)
            if total <= 0:
                raise ValueError(
                    "conditional event has zero reference support; "
                    "choose a supported state/reference pair instead of changing the estimand"
                )
            probabilities = np.zeros(self.rules.num_actions, dtype=np.float64)
            for point in matches:
                probabilities += (point.weight / total) * self._point_action_distribution(point)
        return renormalize_legal_actions(probabilities, action_mask)

    def _policy_action_distribution(
        self,
        observation: Sequence[int | float] | np.ndarray,
        action_mask: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        key = _policy_cache_key(observation, action_mask)
        with self._policy_cache_lock:
            cached = self._policy_distribution_cache.get(key)
            if cached is not None:
                return cached
        probabilities = np.asarray(
            self.policy.action_distribution(
                np.asarray(observation),
                np.asarray(action_mask, dtype=np.int8),
                epsilon=0.0,
            ),
            dtype=np.float64,
        )
        with self._policy_cache_lock:
            return self._policy_distribution_cache.setdefault(key, probabilities)

    def _policy_state_value(
        self,
        observation: Sequence[int | float] | np.ndarray,
        action_mask: Sequence[int] | np.ndarray,
    ) -> float:
        key = _policy_cache_key(observation, action_mask)
        with self._policy_cache_lock:
            cached = self._policy_state_value_cache.get(key)
            if cached is not None:
                return cached
        value = float(
            self.policy.state_value(
                np.asarray(observation),
                np.asarray(action_mask, dtype=np.int8),
                epsilon=0.0,
            )
        )
        with self._policy_cache_lock:
            return self._policy_state_value_cache.setdefault(key, value)

    def _point_action_distribution(self, point: TicTacToeDistributionPoint) -> np.ndarray:
        return self._policy_action_distribution(point.observation, point.action_mask)

    def _point_state_value(self, point: TicTacToeDistributionPoint) -> float:
        return self._policy_state_value(point.observation, point.action_mask)

    def _policy_q_values(self, observation: Sequence[int | float] | np.ndarray) -> np.ndarray:
        key = tuple(np.asarray(observation).reshape(-1).tolist())
        with self._policy_cache_lock:
            cached = self._policy_q_values_cache.get(key)
            if cached is not None:
                return cached
            # Keep the model call inside the lock so parallel exact-Shapley
            # workers do not all execute the same neural query before the
            # first result reaches the cache.
            values = np.asarray(self.policy.q_values(np.asarray(observation)), dtype=np.float64)
            return self._policy_q_values_cache.setdefault(key, values)

    def _point_q_values(self, point: TicTacToeDistributionPoint) -> np.ndarray:
        return self._policy_q_values(point.observation)

    def _action_continuation_returns(self, state: GameState, learner_mark: PlayerMark) -> np.ndarray:
        returns = np.zeros(self.rules.num_actions, dtype=np.float64)

        # Cache is intentionally scoped to this call: it memoizes the recursive game-tree
        # expansion below (otherwise exponential). It is not hoisted to module scope because
        # `value` closes over `self`'s policy, which is mutable across explainer instances.
        @cache
        def value(current: GameState) -> float:
            if current.is_terminal:
                player_0_return, player_1_return = self.rules.returns(current)
                return float(player_0_return if learner_mark == X else player_1_return)

            if current.current_player == learner_mark:
                observation = self.rules.observation(current, learner_mark, "flat")
                probabilities = self._policy_action_distribution(
                    observation["observation"],
                    observation["action_mask"],
                )
            else:
                probabilities = opponent_action_distribution(
                    self.opponent,
                    current,
                    perturbed_minimax_probability=self.perturbed_minimax_probability,
                    rules=self.rules,
                )

            return float(
                sum(
                    float(probability) * value(self.rules.apply(current, action))
                    for action, probability in enumerate(probabilities)
                    if probability > 0
                )
            )

        for action in self.rules.legal_actions(state):
            returns[action] = value(self.rules.apply(state, action))
        return returns

    def _expected_return(
        self,
        state: GameState,
        learner_mark: PlayerMark,
        *,
        override_distribution: tuple[float, ...] | None,
    ) -> float:
        target_state = state
        override = None if override_distribution is None else tuple(float(value) for value in override_distribution)

        # Per-call cache (see ``_action_continuation_returns``): also depends on ``override``
        # and ``target_state``, so it must not be shared across calls.
        @cache
        def value(current: GameState) -> float:
            if current.is_terminal:
                player_0_return, player_1_return = self.rules.returns(current)
                return float(player_0_return if learner_mark == X else player_1_return)

            if current.current_player == learner_mark:
                if override is not None and current == target_state:
                    probabilities = np.asarray(override, dtype=np.float64)
                else:
                    observation = self.rules.observation(current, learner_mark, "flat")
                    probabilities = self._policy_action_distribution(
                        observation["observation"],
                        observation["action_mask"],
                    )
            else:
                probabilities = opponent_action_distribution(
                    self.opponent,
                    current,
                    perturbed_minimax_probability=self.perturbed_minimax_probability,
                    rules=self.rules,
                )

            total = 0.0
            for action, probability in enumerate(probabilities):
                if probability <= 0:
                    continue
                total += float(probability) * value(self.rules.apply(current, action))
            return total

        return value(state)

    def _result(
        self,
        target: str,
        state: GameState,
        action: int | None,
        characteristic: CachedCharacteristicFunction,
    ) -> ExplanationResult:
        shapley_values = self.estimator.shapley_values(characteristic)
        attributions = {
            feature_name: float(value) for feature_name, value in zip(self.features.names, shapley_values, strict=True)
        }
        semantics = {
            "behavior": ExplanationSemantics(
                target_quantity="action_probability",
                coalition_semantics="conditional_occupancy",
                value_model_source="policy",
                shapley_estimator="exact",
            ),
            "prediction": ExplanationSemantics(
                target_quantity="policy_weighted_q",
                coalition_semantics="conditional_occupancy",
                value_model_source="policy",
                shapley_estimator="exact",
            ),
            "outcome": ExplanationSemantics(
                target_quantity="expected_outcome",
                coalition_semantics="conditional_occupancy",
                value_model_source="environment_return",
                shapley_estimator="exact",
            ),
            "q_value": ExplanationSemantics(
                target_quantity="fixed_action_q",
                coalition_semantics="conditional_occupancy",
                value_model_source="base_q_function",
                shapley_estimator="exact",
            ),
        }[target]
        return ExplanationResult(
            target=target,
            state=state,
            action=action,
            baseline_value=characteristic.value(frozenset()),
            full_value=characteristic.value(frozenset(range(self.features.feature_count))),
            attributions=attributions,
            feature_names=self.features.names,
            characteristic_assumption=CONDITIONAL_OCCUPANCY_ASSUMPTION,
            semantics=semantics,
        )


def tictactoe_feature_spec() -> FeatureSpec:
    return FeatureSpec(
        names=tuple(f"cell_{index}" for index in range(9)),
        display_metadata=tuple({"row": index // 3, "col": index % 3} for index in range(9)),
    )


def _policy_cache_key(
    observation: Sequence[int | float] | np.ndarray,
    action_mask: Sequence[int] | np.ndarray,
) -> tuple[tuple[int | float, ...], tuple[int, ...]]:
    observation_key = tuple(np.asarray(observation).reshape(-1).tolist())
    mask_key = tuple(int(value) for value in np.asarray(action_mask, dtype=np.int8).reshape(-1))
    return observation_key, mask_key


def build_tictactoe_state_distribution(
    policy: PolicyView,
    *,
    opponent: str = "random",
    perturbed_minimax_probability: float | None = None,
    rules: TicTacToeRules | None = None,
    features: FeatureSpec | None = None,
    profile_weights: dict[PlayerMark, float] | None = None,
) -> TicTacToeStateDistribution:
    rules = rules or TicTacToeRules()
    features = features or tictactoe_feature_spec()
    weights = profile_weights or {X: 0.5, O: 0.5}
    points: list[TicTacToeDistributionPoint] = []

    for learner_mark, profile_weight in weights.items():
        frontier: dict[GameState, float] = {rules.initial_state(): float(profile_weight)}
        while frontier:
            next_frontier: dict[GameState, float] = defaultdict(float)
            for state, mass in frontier.items():
                if mass <= 0 or state.is_terminal:
                    continue
                if state.current_player == learner_mark:
                    observation = rules.observation(state, learner_mark, "flat")
                    obs_tuple = tuple(int(value) for value in np.asarray(observation["observation"]).reshape(-1))
                    mask_tuple = tuple(int(value) for value in np.asarray(observation["action_mask"]).reshape(-1))
                    points.append(
                        TicTacToeDistributionPoint(
                            state=state,
                            learner_mark=learner_mark,
                            observation=obs_tuple,
                            action_mask=mask_tuple,
                            weight=mass,
                        )
                    )
                    probabilities = policy.action_distribution(
                        observation["observation"],
                        observation["action_mask"],
                        epsilon=0.0,
                    )
                else:
                    probabilities = opponent_action_distribution(
                        opponent,
                        state,
                        perturbed_minimax_probability=perturbed_minimax_probability,
                        rules=rules,
                    )

                for action, probability in enumerate(probabilities):
                    if probability > 0:
                        next_frontier[rules.apply(state, action)] += mass * float(probability)
            frontier = next_frontier

    return TicTacToeStateDistribution(points, features=features)


def state_from_observation(
    observation: Sequence[int | float] | np.ndarray,
    *,
    rules: TicTacToeRules | None = None,
) -> GameState:
    """Rebuild the absolute position a mover-relative flat observation encodes.

    The flat observation is ``board * mover``
    (:meth:`~xrl.tictactoe.core.TicTacToeRules.observation` with
    ``obs_mode="flat"``), which drops who the mover is. Alternation restores it:
    X moves whenever an even number of cells is filled, so the mover is X
    exactly when the filled count is even, and ``board = observation * mover``
    because ``mover ** 2 == 1``.

    ``winner`` and ``is_draw`` are computed rather than left at their dataclass
    defaults. A board that is already won would otherwise present as
    non-terminal, and every SVERL target rejects terminal states — so the
    default would turn a bad probe state into a wrong answer instead of an
    error.

    This is the inverse of the mapping the explanation adapters apply, and it is
    what lets a study that only carries ``(observation, action_mask)`` reach the
    conditional-occupancy game, which is defined over positions rather than
    over observation vectors.
    """

    rules = rules or TicTacToeRules()
    cells = np.asarray(observation, dtype=np.float64).reshape(-1)
    if cells.size != rules.num_actions:
        raise ValueError(f"observation must hold {rules.num_actions} cells, got {cells.size}")
    rounded = np.rint(cells).astype(np.int8)
    if not np.array_equal(rounded, cells):
        raise ValueError(f"observation must hold integral cell marks, got {cells.tolist()}")
    invalid = sorted({int(value) for value in rounded} - {-1, 0, 1})
    if invalid:
        raise ValueError(f"observation cells must be -1, 0, or 1, got {invalid}")

    move_count = int(np.count_nonzero(rounded))
    mover: PlayerMark = X if move_count % 2 == 0 else O
    board = tuple(int(value) * mover for value in rounded)
    winner = rules.winner(board)
    return GameState(
        board=board,
        current_player=mover,
        move_count=move_count,
        winner=winner,
        is_draw=winner is None and move_count == rules.num_actions,
    )


def run_opponent_law(config: Mapping[str, Any]) -> tuple[str, float | None]:
    """Return the opponent law a finished run's policy faced, as ``(name, rate)``.

    ``perturbed_minimax`` anneals its random-move rate as ``r_t = discount ** (t
    - 1)`` (:class:`~xrl.tictactoe.agents.PerturbedMinimaxAgent`), so the
    occupancy a run visits is non-stationary. A post-hoc explanation of the
    *finished* policy has to marginalize over the occupancy that policy actually
    faces at the end of training, which is the terminal rate ``r_T = discount **
    (episodes - 1)`` — not the rate averaged over training, and not the initial
    one. ``None`` is returned for opponents that carry no rate.

    A ``discount`` of ``1.0`` (the default) leaves ``r_T = 1.0``: the opponent
    never anneals and is uniformly random for the whole run, which makes
    ``--opponent perturbed_minimax`` behaviorally identical to
    ``--opponent random``. Callers that report the derived rate let a reader see
    that rather than infer it.
    """

    opponent = str(config.get("opponent", "random"))
    if not opponent.lower().startswith("perturbed_minimax"):
        return opponent, None
    discount = float(config.get("perturbed_minimax_discount", 1.0))
    episodes = int(config.get("episodes", 1))
    return opponent, discount ** max(episodes - 1, 0)


def opponent_action_distribution(
    opponent: str,
    state: GameState,
    *,
    perturbed_minimax_probability: float | None = None,
    rules: TicTacToeRules | None = None,
) -> np.ndarray:
    rules = rules or TicTacToeRules()
    legal = rules.legal_actions(state)
    if not legal:
        return np.zeros(rules.num_actions, dtype=np.float64)

    normalized = opponent.lower()
    probabilities = np.zeros(rules.num_actions, dtype=np.float64)
    if normalized == "random":
        for action in legal:
            probabilities[action] = 1.0 / len(legal)
        return probabilities

    if normalized.startswith("perturbed_minimax"):
        probability = perturbed_minimax_probability
        if probability is None:
            raise ValueError("perturbed_minimax opponent requires perturbed_minimax_probability")
        minimax = opponent_action_distribution("minimax", state, rules=rules)
        random_policy = opponent_action_distribution("random", state, rules=rules)
        return (1.0 - float(probability)) * minimax + float(probability) * random_policy

    if normalized in {"minimax", "perfect"}:
        mark = state.current_player
        observation = rules.observation(state, mark, "flat")
        action = MinimaxAgent(rules)(
            observation,
            observation["action_mask"],
            state=state,
            player=mark,
        )
        probabilities[action] = 1.0
        return probabilities

    raise ValueError(f"unsupported Tic Tac Toe opponent for exact SVERL: {opponent!r}")


def renormalize_legal_actions(
    probabilities: Sequence[float] | np.ndarray,
    action_mask: Sequence[int] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError(f"probabilities and action_mask must have the same shape, got {values.shape} and {mask.shape}")
    legal = [index for index, value in enumerate(mask) if int(value) == 1]
    if not legal:
        raise ValueError("no legal actions available")
    normalized = np.zeros_like(values, dtype=np.float64)
    for action in legal:
        normalized[action] = max(float(values[action]), 0.0)
    total = float(normalized.sum())
    if total <= 0:
        for action in legal:
            normalized[action] = 1.0 / len(legal)
        return normalized
    return normalized / total
