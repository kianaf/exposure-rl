"""Exact Tic-Tac-Toe evaluation and declared study reference laws."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, cast

import numpy as np

from xrl.explainability.tictactoe import (
    TicTacToeDistributionPoint,
    TicTacToeStateDistribution,
    tictactoe_feature_spec,
)
from xrl.tictactoe.core import GameState, O, PlayerMark, TicTacToeRules, X
from xrl.training.analysis import minimax_action_values
from xrl.training.interfaces import PolicyView

RoleName = Literal["X", "O"]
OpeningClass = Literal["corner", "edge", "center"]
PolicyDistributionFn = Callable[[GameState, PlayerMark], np.ndarray]

_X_MARK = cast(PlayerMark, X)
_O_MARK = cast(PlayerMark, O)
_MARKS: tuple[PlayerMark, PlayerMark] = (_X_MARK, _O_MARK)
_ROLE_NAME: dict[PlayerMark, RoleName] = {_X_MARK: "X", _O_MARK: "O"}
_OPENING_CLASS: dict[int, OpeningClass] = {
    0: "corner",
    1: "edge",
    2: "corner",
    3: "edge",
    4: "center",
    5: "edge",
    6: "corner",
    7: "edge",
    8: "corner",
}


@dataclass(frozen=True, slots=True)
class ExactOutcome:
    """Exact terminal-outcome probabilities from one learner perspective."""

    win_probability: float
    draw_probability: float
    loss_probability: float

    def __post_init__(self) -> None:
        values = (self.win_probability, self.draw_probability, self.loss_probability)
        if any(not np.isfinite(value) or value < -1e-12 for value in values):
            raise ValueError(f"invalid outcome probabilities {values}")
        if not np.isclose(sum(values), 1.0, atol=1e-10):
            raise ValueError(f"outcome probabilities must sum to one, got {values}")

    @property
    def expected_return(self) -> float:
        return self.win_probability - self.loss_probability


@dataclass(frozen=True, slots=True)
class ExactRoleEvaluation:
    """Exact policy and minimax diagnostics for one learner role."""

    learner_role: RoleName
    outcome: ExactOutcome
    best_response_return: float
    exploitability: float
    optimal_action_mass: float
    mean_one_step_regret: float
    expected_decisions: float
    decision_support_states: int

    def to_row(self) -> dict[str, Any]:
        return {
            "learner_role": self.learner_role,
            "expected_return": self.outcome.expected_return,
            "win_probability": self.outcome.win_probability,
            "draw_probability": self.outcome.draw_probability,
            "loss_probability": self.outcome.loss_probability,
            "best_response_return": self.best_response_return,
            "exploitability": self.exploitability,
            "optimal_action_mass": self.optimal_action_mass,
            "mean_one_step_regret": self.mean_one_step_regret,
            "expected_decisions": self.expected_decisions,
            "decision_support_states": self.decision_support_states,
        }


@dataclass(frozen=True, slots=True)
class ExactOpeningEvaluation:
    """Exact continuation after forcing one of the nine first moves."""

    learner_role: RoleName
    opening_action: int
    opening_class: OpeningClass
    natural_opening_probability: float
    outcome: ExactOutcome
    best_response_return: float
    exploitability: float
    optimal_action_mass: float
    mean_one_step_regret: float

    def to_row(self) -> dict[str, Any]:
        return {
            "learner_role": self.learner_role,
            "opening_action": self.opening_action,
            "opening_class": self.opening_class,
            "natural_opening_probability": self.natural_opening_probability,
            "expected_return": self.outcome.expected_return,
            "win_probability": self.outcome.win_probability,
            "draw_probability": self.outcome.draw_probability,
            "loss_probability": self.outcome.loss_probability,
            "best_response_return": self.best_response_return,
            "exploitability": self.exploitability,
            "optimal_action_mass": self.optimal_action_mass,
            "mean_one_step_regret": self.mean_one_step_regret,
        }


@dataclass(frozen=True, slots=True)
class ExactPolicyEvaluation:
    """Role-balanced exact evaluation with every forced opening retained."""

    roles: tuple[ExactRoleEvaluation, ExactRoleEvaluation]
    openings: tuple[ExactOpeningEvaluation, ...]

    @property
    def role_balanced_row(self) -> dict[str, Any]:
        rows = [role.to_row() for role in self.roles]
        numeric = [key for key, value in rows[0].items() if key != "learner_role" and isinstance(value, (int, float))]
        return {
            "learner_role": "balanced",
            **{key: sum(float(row[key]) for row in rows) / len(rows) for key in numeric},
        }

    def role_rows(self, *, include_balanced: bool = True) -> tuple[dict[str, Any], ...]:
        rows = tuple(role.to_row() for role in self.roles)
        return (*rows, self.role_balanced_row) if include_balanced else rows

    def opening_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(opening.to_row() for opening in self.openings)


@dataclass(frozen=True, slots=True)
class RoleCoverage:
    """Support and captured mass for one role in a deterministic state sample."""

    learner_role: RoleName
    support_states: int
    selected_states: int
    total_mass: float
    captured_mass: float

    @property
    def state_coverage(self) -> float:
        return self.selected_states / self.support_states if self.support_states else 0.0

    def to_row(self) -> dict[str, Any]:
        return {
            "learner_role": self.learner_role,
            "support_states": self.support_states,
            "selected_states": self.selected_states,
            "state_coverage": self.state_coverage,
            "total_mass": self.total_mass,
            "captured_mass": self.captured_mass,
        }


@dataclass(frozen=True, slots=True)
class DistributionSample:
    """Deterministic weighted-without-replacement state sample and coverage."""

    points: tuple[TicTacToeDistributionPoint, ...]
    coverage: tuple[RoleCoverage, RoleCoverage]

    @property
    def captured_mass(self) -> float:
        return sum(role.captured_mass for role in self.coverage)


@dataclass(frozen=True, slots=True)
class ReferenceOccurrenceBuffer:
    """One role-balanced occurrence buffer and its collapsed empirical law."""

    source: str
    seed: int
    per_role: int
    occurrences: tuple[TicTacToeDistributionPoint, ...]
    distribution: TicTacToeStateDistribution

    def role_occurrences(self, learner_mark: PlayerMark) -> tuple[TicTacToeDistributionPoint, ...]:
        return tuple(point for point in self.occurrences if point.learner_mark == learner_mark)


class ExactTicTacToeEvaluator:
    """Enumerate the full game tree for one frozen policy."""

    def __init__(
        self,
        rules: TicTacToeRules | None = None,
        *,
        opponent_random_move_probability: float = 1.0,
    ):
        if not 0.0 <= opponent_random_move_probability <= 1.0:
            raise ValueError("opponent_random_move_probability must lie in [0, 1]")
        self.rules = rules or TicTacToeRules()
        self.opponent_random_move_probability = float(opponent_random_move_probability)

    def evaluate(self, policy: PolicyView) -> ExactPolicyEvaluation:
        @cache
        def policy_distribution(state: GameState, learner_mark: PlayerMark) -> np.ndarray:
            return _policy_distribution(policy, state, learner_mark, self.rules)

        roles = (
            self._evaluate_role(policy_distribution, _X_MARK),
            self._evaluate_role(policy_distribution, _O_MARK),
        )
        openings = tuple(
            self._evaluate_opening(policy_distribution, mark, action)
            for mark in _MARKS
            for action in range(self.rules.num_actions)
        )
        return ExactPolicyEvaluation(roles=roles, openings=openings)

    def _evaluate_role(
        self,
        policy_distribution: PolicyDistributionFn,
        learner_mark: PlayerMark,
    ) -> ExactRoleEvaluation:
        initial = self.rules.initial_state()
        outcome = self._outcome_from(policy_distribution, learner_mark, initial)
        best_response_return = self._best_response_return_from(policy_distribution, learner_mark, initial)
        stats = self._decision_stats(policy_distribution, learner_mark)
        return ExactRoleEvaluation(
            learner_role=_ROLE_NAME[learner_mark],
            outcome=outcome,
            best_response_return=best_response_return,
            exploitability=max(0.0, -best_response_return),
            optimal_action_mass=stats.optimal_action_mass,
            mean_one_step_regret=stats.mean_one_step_regret,
            expected_decisions=stats.expected_decisions,
            decision_support_states=stats.support_states,
        )

    def _evaluate_opening(
        self,
        policy_distribution: PolicyDistributionFn,
        learner_mark: PlayerMark,
        opening_action: int,
    ) -> ExactOpeningEvaluation:
        initial = self.rules.initial_state()
        child = self.rules.apply(initial, opening_action)
        if learner_mark == _X_MARK:
            opening_probability = float(policy_distribution(initial, _X_MARK)[opening_action])
        else:
            opening_probability = float(self._opponent_distribution(initial)[opening_action])
        outcome = self._outcome_from(policy_distribution, learner_mark, child)
        best_response_return = self._best_response_return_from(policy_distribution, learner_mark, child)
        stats = self._decision_stats(policy_distribution, learner_mark, forced_opening=opening_action)
        return ExactOpeningEvaluation(
            learner_role=_ROLE_NAME[learner_mark],
            opening_action=opening_action,
            opening_class=_OPENING_CLASS[opening_action],
            natural_opening_probability=opening_probability,
            outcome=outcome,
            best_response_return=best_response_return,
            exploitability=max(0.0, -best_response_return),
            optimal_action_mass=stats.optimal_action_mass,
            mean_one_step_regret=stats.mean_one_step_regret,
        )

    def _outcome_from(
        self,
        policy_distribution: PolicyDistributionFn,
        learner_mark: PlayerMark,
        start: GameState,
    ) -> ExactOutcome:
        @cache
        def visit(state: GameState) -> tuple[float, float, float]:
            if state.is_terminal:
                learner_return = self.rules.returns(state)[0 if learner_mark == _X_MARK else 1]
                if learner_return > 0:
                    return (1.0, 0.0, 0.0)
                if learner_return < 0:
                    return (0.0, 0.0, 1.0)
                return (0.0, 1.0, 0.0)
            probabilities = (
                policy_distribution(state, learner_mark)
                if state.current_player == learner_mark
                else self._opponent_distribution(state)
            )
            result = np.zeros(3, dtype=np.float64)
            for action in self.rules.legal_actions(state):
                probability = float(probabilities[action])
                if probability > 0:
                    result += probability * np.asarray(visit(self.rules.apply(state, action)))
            return (float(result[0]), float(result[1]), float(result[2]))

        return ExactOutcome(*visit(start))

    def _best_response_return_from(
        self,
        policy_distribution: PolicyDistributionFn,
        learner_mark: PlayerMark,
        start: GameState,
    ) -> float:
        @cache
        def visit(state: GameState) -> float:
            if state.is_terminal:
                return float(self.rules.returns(state)[0 if learner_mark == _X_MARK else 1])
            children = [visit(self.rules.apply(state, action)) for action in self.rules.legal_actions(state)]
            if state.current_player != learner_mark:
                return min(children)
            probabilities = policy_distribution(state, learner_mark)
            return float(
                sum(
                    float(probabilities[action]) * child
                    for action, child in zip(self.rules.legal_actions(state), children, strict=True)
                )
            )

        return visit(start)

    def _decision_stats(
        self,
        policy_distribution: PolicyDistributionFn,
        learner_mark: PlayerMark,
        *,
        forced_opening: int | None = None,
    ) -> _DecisionStats:
        initial = self.rules.initial_state()
        frontier: dict[GameState, float]
        optimal_mass_sum = 0.0
        regret_sum = 0.0
        decision_mass = 0.0
        support: set[GameState] = set()
        if forced_opening is None:
            frontier = {initial: 1.0}
        else:
            frontier = {self.rules.apply(initial, forced_opening): 1.0}
            if learner_mark == _X_MARK:
                mass, regret = _action_quality(initial, _one_hot(forced_opening, self.rules.num_actions))
                optimal_mass_sum += mass
                regret_sum += regret
                decision_mass += 1.0
                support.add(initial)

        while frontier:
            next_frontier: dict[GameState, float] = defaultdict(float)
            for state, state_mass in frontier.items():
                if state_mass <= 0 or state.is_terminal:
                    continue
                if state.current_player == learner_mark:
                    probabilities = policy_distribution(state, learner_mark)
                    optimal_mass, regret = _action_quality(state, probabilities)
                    optimal_mass_sum += state_mass * optimal_mass
                    regret_sum += state_mass * regret
                    decision_mass += state_mass
                    support.add(state)
                else:
                    probabilities = self._opponent_distribution(state)
                for action in self.rules.legal_actions(state):
                    probability = float(probabilities[action])
                    if probability > 0:
                        next_frontier[self.rules.apply(state, action)] += state_mass * probability
            frontier = next_frontier
        if decision_mass <= 0:
            raise ValueError(f"no learner decision states for role {_ROLE_NAME[learner_mark]}")
        return _DecisionStats(
            optimal_action_mass=optimal_mass_sum / decision_mass,
            mean_one_step_regret=regret_sum / decision_mass,
            expected_decisions=decision_mass,
            support_states=len(support),
        )

    def _opponent_distribution(self, state: GameState) -> np.ndarray:
        return _perturbed_minimax_distribution(
            state,
            self.rules,
            random_move_probability=self.opponent_random_move_probability,
        )


class ExactReturnQOracle:
    """Exact continuation-return Q and its conditional-occupancy game.

    The primary Phase 1 oracle is the frozen network's Q function. This
    secondary oracle keeps the state law and fixed-action conditioning exactly
    the same, but replaces the quantity averaged over matching states with the
    exact terminal return obtained after taking the fixed action and then
    following the frozen learner policy against the declared opponent.
    Consequently the difference between both exact Shapley vectors is base-Q
    error rather than a change of explanation law.
    """

    def __init__(
        self,
        policy: PolicyView,
        distribution: TicTacToeStateDistribution,
        *,
        rules: TicTacToeRules | None = None,
        opponent_random_move_probability: float = 1.0,
    ) -> None:
        if not 0.0 <= opponent_random_move_probability <= 1.0:
            raise ValueError("opponent_random_move_probability must lie in [0, 1]")
        self.policy = policy
        self.distribution = distribution
        self.rules = rules or TicTacToeRules()
        self.opponent_random_move_probability = float(opponent_random_move_probability)
        self._policy_cache: dict[tuple[GameState, PlayerMark], np.ndarray] = {}
        self._return_cache: dict[tuple[GameState, PlayerMark], float] = {}
        self._coalition_cache: dict[tuple[GameState, PlayerMark, int], np.ndarray] = {}

    def action_return(self, state: GameState, *, action: int, learner_mark: PlayerMark) -> float:
        """Return after forcing one legal learner action and following the policy."""

        if state.is_terminal or state.current_player != learner_mark:
            raise ValueError("return-Q requires a non-terminal learner-decision state")
        if action not in self.rules.legal_actions(state):
            raise ValueError(f"action {action} is illegal in the requested state")
        return self._state_return(self.rules.apply(state, action), learner_mark)

    def coalition_values(
        self,
        state: GameState,
        *,
        action: int,
        learner_mark: PlayerMark,
    ) -> np.ndarray:
        """All conditional exact-return-Q coalition values in bitmask order."""

        key = (state, learner_mark, int(action))
        cached = self._coalition_cache.get(key)
        if cached is not None:
            return cached.copy()
        target = self.rules.observation(state, learner_mark, "flat")
        view = self.distribution.legal_action_view(int(action))
        operator = view.conditional_expectation_operator(tuple(int(value) for value in target["observation"]))
        support = np.asarray(
            [
                self.action_return(point.state, action=int(action), learner_mark=point.learner_mark)
                for point in view.points
            ],
            dtype=np.float64,
        )
        values = operator @ support
        values[-1] = self.action_return(state, action=int(action), learner_mark=learner_mark)
        values.setflags(write=False)
        self._coalition_cache[key] = values
        return values.copy()

    def _state_return(self, state: GameState, learner_mark: PlayerMark) -> float:
        key = (state, learner_mark)
        cached = self._return_cache.get(key)
        if cached is not None:
            return cached
        if state.is_terminal:
            value = float(self.rules.returns(state)[0 if learner_mark == _X_MARK else 1])
        else:
            probabilities = (
                self._learner_distribution(state, learner_mark)
                if state.current_player == learner_mark
                else _perturbed_minimax_distribution(
                    state,
                    self.rules,
                    random_move_probability=self.opponent_random_move_probability,
                )
            )
            value = sum(
                float(probabilities[action]) * self._state_return(self.rules.apply(state, action), learner_mark)
                for action in self.rules.legal_actions(state)
                if probabilities[action] > 0
            )
        self._return_cache[key] = float(value)
        return float(value)

    def _learner_distribution(self, state: GameState, learner_mark: PlayerMark) -> np.ndarray:
        key = (state, learner_mark)
        cached = self._policy_cache.get(key)
        if cached is None:
            cached = _policy_distribution(self.policy, state, learner_mark, self.rules)
            cached.setflags(write=False)
            self._policy_cache[key] = cached
        return cached


@dataclass(frozen=True, slots=True)
class _DecisionStats:
    optimal_action_mass: float
    mean_one_step_regret: float
    expected_decisions: float
    support_states: int


def build_shared_reference_distribution(
    *,
    rules: TicTacToeRules | None = None,
) -> TicTacToeStateDistribution:
    """Equal-role learner-decision law under uniform legal play by both sides."""

    rules = rules or TicTacToeRules()
    return _build_reference_distribution(None, rules=rules)


def build_policy_reference_distribution(
    policy: PolicyView,
    *,
    rules: TicTacToeRules | None = None,
    softening: float = 0.0,
    opponent_random_move_probability: float = 1.0,
) -> TicTacToeStateDistribution:
    """Equal-role law of the frozen learner against a fixed opponent mixture.

    ``softening`` mixes uniform legal play into the learner's own moves. At the
    default of zero the law is the greedy occupancy the study predeclares; a
    positive value widens the support, which is the only way the surrogate's
    training-point count can vary, since that count is the law's full support.

    ``opponent_random_move_probability`` mixes uniform legal play into the
    deterministic minimax opponent. One preserves the historical random law;
    Phase 1 uses the terminal training value of 0.05.
    """

    rules = rules or TicTacToeRules()
    if not 0.0 <= opponent_random_move_probability <= 1.0:
        raise ValueError("opponent_random_move_probability must lie in [0, 1]")
    return _build_reference_distribution(
        policy,
        rules=rules,
        softening=softening,
        opponent_random_move_probability=opponent_random_move_probability,
    )


def sample_reference_distribution(
    distribution: TicTacToeStateDistribution,
    *,
    per_role: int = 64,
    seed: int = 0,
) -> DistributionSample:
    """Sample a fixed number of support states per role without replacement."""

    if per_role < 1:
        raise ValueError("per_role must be positive")
    rng = np.random.default_rng(seed)
    selected: list[TicTacToeDistributionPoint] = []
    coverage: list[RoleCoverage] = []
    for mark in _MARKS:
        candidates = sorted(
            (point for point in distribution.points if point.learner_mark == mark and point.weight > 0),
            key=lambda point: (point.state.move_count, point.state.board),
        )
        count = min(per_role, len(candidates))
        if count == len(candidates):
            chosen = candidates
        else:
            probabilities = np.asarray([point.weight for point in candidates], dtype=np.float64)
            probabilities /= probabilities.sum()
            indices = rng.choice(len(candidates), size=count, replace=False, p=probabilities)
            chosen = [candidates[int(index)] for index in sorted(indices)]
        selected.extend(chosen)
        total_mass = sum(point.weight for point in candidates)
        captured_mass = sum(point.weight for point in chosen)
        coverage.append(
            RoleCoverage(
                learner_role=_ROLE_NAME[mark],
                support_states=len(candidates),
                selected_states=len(chosen),
                total_mass=total_mass,
                captured_mass=captured_mass,
            )
        )
    return DistributionSample(points=tuple(selected), coverage=(coverage[0], coverage[1]))


def sample_stratified_reference_distribution(
    distribution: TicTacToeStateDistribution,
    *,
    total: int = 32,
    seed: int = 0,
) -> DistributionSample:
    """Sample equal roles and near-equal reachable depths without replacement."""

    if total < 2 or total % 2:
        raise ValueError("total must be a positive even number")
    per_role = total // 2
    rng = np.random.default_rng(seed)
    selected: list[TicTacToeDistributionPoint] = []
    coverage: list[RoleCoverage] = []
    for mark in _MARKS:
        candidates = sorted(
            (point for point in distribution.points if point.learner_mark == mark and point.weight > 0),
            key=lambda point: (point.state.move_count, point.state.board),
        )
        count = min(per_role, len(candidates))
        by_depth: dict[int, list[TicTacToeDistributionPoint]] = defaultdict(list)
        for point in candidates:
            by_depth[point.state.move_count].append(point)
        depths = sorted(by_depth)
        allocations = {depth: 0 for depth in depths}
        while sum(allocations.values()) < count:
            progressed = False
            for depth in depths:
                if allocations[depth] >= len(by_depth[depth]):
                    continue
                allocations[depth] += 1
                progressed = True
                if sum(allocations.values()) == count:
                    break
            if not progressed:
                break

        chosen: list[TicTacToeDistributionPoint] = []
        for depth in depths:
            depth_candidates = by_depth[depth]
            depth_count = allocations[depth]
            if depth_count == len(depth_candidates):
                chosen.extend(depth_candidates)
                continue
            probabilities = np.asarray([point.weight for point in depth_candidates], dtype=np.float64)
            probabilities /= probabilities.sum()
            indices = rng.choice(len(depth_candidates), size=depth_count, replace=False, p=probabilities)
            chosen.extend(depth_candidates[int(index)] for index in sorted(indices))
        chosen.sort(key=lambda point: (point.state.move_count, point.state.board))
        selected.extend(chosen)
        coverage.append(
            RoleCoverage(
                learner_role=_ROLE_NAME[mark],
                support_states=len(candidates),
                selected_states=len(chosen),
                total_mass=sum(point.weight for point in candidates),
                captured_mass=sum(point.weight for point in chosen),
            )
        )
    return DistributionSample(points=tuple(selected), coverage=(coverage[0], coverage[1]))


def build_nested_reference_occurrence_buffers(
    distribution: TicTacToeStateDistribution,
    *,
    per_role_sizes: Sequence[int],
    seed: int,
    source: str,
) -> tuple[ReferenceOccurrenceBuffer, ...]:
    """Draw nested role-balanced occurrences and collapse duplicates by count.

    One maximum-size IID sequence is drawn per role. Every smaller empirical law
    is a prefix of that same sequence, so buffer-size comparisons change sample
    count without changing earlier observations. A collapsed point receives
    weight ``count / (2 * per_role)``; this is exactly equivalent to retaining
    duplicates and then mixing the two learner roles equally.
    """

    sizes = tuple(int(size) for size in per_role_sizes)
    if not sizes or tuple(sorted(set(sizes))) != sizes or any(size < 1 for size in sizes):
        raise ValueError("per_role_sizes must be strictly increasing positive integers")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    maximum = sizes[-1]
    sequences: dict[PlayerMark, tuple[TicTacToeDistributionPoint, ...]] = {}
    for mark in _MARKS:
        candidates = tuple(point for point in distribution.points if point.learner_mark == mark and point.weight > 0)
        if not candidates:
            raise ValueError(f"reference distribution has no {_ROLE_NAME[mark]} support")
        probabilities = np.asarray([point.weight for point in candidates], dtype=np.float64)
        probabilities /= probabilities.sum()
        indices = rng.choice(len(candidates), size=maximum, replace=True, p=probabilities)
        sequences[mark] = tuple(candidates[int(index)] for index in indices)

    buffers: list[ReferenceOccurrenceBuffer] = []
    for size in sizes:
        occurrences = tuple(point for mark in _MARKS for point in sequences[mark][:size])
        counts = Counter(occurrences)
        collapsed = tuple(
            TicTacToeDistributionPoint(
                state=point.state,
                learner_mark=point.learner_mark,
                observation=point.observation,
                action_mask=point.action_mask,
                weight=count / (2.0 * size),
            )
            for point, count in sorted(
                counts.items(),
                key=lambda item: (0 if item[0].learner_mark == _X_MARK else 1, item[0].state.board),
            )
        )
        buffers.append(
            ReferenceOccurrenceBuffer(
                source=source,
                seed=seed,
                per_role=size,
                occurrences=occurrences,
                distribution=TicTacToeStateDistribution(collapsed, features=distribution.features),
            )
        )
    return tuple(buffers)


def _build_reference_distribution(
    policy: PolicyView | None,
    *,
    rules: TicTacToeRules,
    softening: float = 0.0,
    opponent_random_move_probability: float = 1.0,
) -> TicTacToeStateDistribution:
    points: list[TicTacToeDistributionPoint] = []
    for learner_mark in _MARKS:
        role_points: list[TicTacToeDistributionPoint] = []
        frontier: dict[GameState, float] = {rules.initial_state(): 1.0}
        while frontier:
            next_frontier: dict[GameState, float] = defaultdict(float)
            for state, mass in frontier.items():
                if mass <= 0 or state.is_terminal:
                    continue
                if state.current_player == learner_mark:
                    observation = rules.observation(state, learner_mark, "flat")
                    role_points.append(
                        TicTacToeDistributionPoint(
                            state=state,
                            learner_mark=learner_mark,
                            observation=tuple(int(value) for value in observation["observation"]),
                            action_mask=tuple(int(value) for value in observation["action_mask"]),
                            weight=mass,
                        )
                    )
                    probabilities = (
                        _uniform_distribution(state, rules)
                        if policy is None
                        else _policy_distribution(policy, state, learner_mark, rules, softening)
                    )
                else:
                    probabilities = _perturbed_minimax_distribution(
                        state,
                        rules,
                        random_move_probability=opponent_random_move_probability,
                    )
                for action in rules.legal_actions(state):
                    probability = float(probabilities[action])
                    if probability > 0:
                        next_frontier[rules.apply(state, action)] += mass * probability
            frontier = next_frontier
        total = sum(point.weight for point in role_points)
        if total <= 0:
            raise ValueError(f"empty reference support for role {_ROLE_NAME[learner_mark]}")
        points.extend(
            TicTacToeDistributionPoint(
                state=point.state,
                learner_mark=point.learner_mark,
                observation=point.observation,
                action_mask=point.action_mask,
                weight=0.5 * point.weight / total,
            )
            for point in role_points
        )
    return TicTacToeStateDistribution(points, features=tictactoe_feature_spec())


def _uniform_distribution(state: GameState, rules: TicTacToeRules) -> np.ndarray:
    legal = rules.legal_actions(state)
    probabilities = np.zeros(rules.num_actions, dtype=np.float64)
    if legal:
        probabilities[list(legal)] = 1.0 / len(legal)
    return probabilities


def _perturbed_minimax_distribution(
    state: GameState,
    rules: TicTacToeRules,
    *,
    random_move_probability: float,
) -> np.ndarray:
    """Exact law of :class:`PerturbedMinimaxAgent` frozen at one difficulty."""

    uniform = _uniform_distribution(state, rules)
    if random_move_probability == 1.0:
        return uniform
    _values, optimal_actions, _best = minimax_action_values(state)
    if not optimal_actions:
        return uniform
    probabilities = random_move_probability * uniform
    # The deployed MinimaxAgent breaks ties by the first legal optimal action.
    probabilities[optimal_actions[0]] += 1.0 - random_move_probability
    return probabilities


def _policy_distribution(
    policy: PolicyView,
    state: GameState,
    learner_mark: PlayerMark,
    rules: TicTacToeRules,
    softening: float = 0.0,
) -> np.ndarray:
    if state.current_player != learner_mark:
        raise ValueError("policy distribution requested outside the learner turn")
    observation = rules.observation(state, learner_mark, "flat")
    values = np.asarray(
        policy.action_distribution(
            observation["observation"],
            observation["action_mask"],
            epsilon=softening,
        ),
        dtype=np.float64,
    ).reshape(-1)
    if values.shape != (rules.num_actions,):
        raise ValueError(f"policy returned action distribution shape {values.shape}, expected {(rules.num_actions,)}")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("policy action distribution must contain finite non-negative probabilities")
    legal_mask = rules.action_mask(state).astype(bool)
    if np.any(values[~legal_mask] > 1e-12):
        raise ValueError("policy assigned positive probability to an illegal action")
    probabilities = np.where(legal_mask, np.maximum(values, 0.0), 0.0)
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("policy assigned zero mass to every legal action")
    return probabilities / total


def _action_quality(
    state: GameState,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    values, optimal_actions, best_value = minimax_action_values(state)
    optimal_mass = sum(float(probabilities[action]) for action in optimal_actions)
    expected_value = 0.0
    for action, value in enumerate(values):
        probability = float(probabilities[action])
        if value is not None and probability > 0:
            expected_value += probability * float(value)
    return optimal_mass, max(0.0, best_value - expected_value)


def _one_hot(action: int, size: int) -> np.ndarray:
    values = np.zeros(size, dtype=np.float64)
    values[action] = 1.0
    return values
