"""Diagnostics for tabular Tic Tac Toe Q-learning."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from xrl.tictactoe.agents import MinimaxAgent
from xrl.tictactoe.core import GameState, O, PlayerMark, TicTacToeRules
from xrl.training.encoders import make_encoder
from xrl.training.q_learning import QLearningAgent


def is_unmasked_tictactoe_key(key: tuple[int | float, ...], *, mask_token: int = 2) -> bool:
    """Return whether a state key is a standard unmasked 9-cell Tic Tac Toe key."""

    return len(key) == 9 and mask_token not in key


@cache
def reachable_flat_state_keys(encoder: str = "identity") -> frozenset[tuple[int | float, ...]]:
    """Return perspective-normalized flat keys for all reachable non-terminal states."""

    rules = TicTacToeRules()
    state_encoder = make_encoder(encoder, num_actions=rules.num_actions)
    keys: set[tuple[int | float, ...]] = set()
    seen_states: set[tuple[tuple[int, ...], int]] = set()

    def visit(state: GameState) -> None:
        identity = (state.board, state.current_player)
        if identity in seen_states:
            return
        seen_states.add(identity)
        if state.is_terminal:
            return
        observation = rules.observation(state, state.current_player, "flat")
        keys.add(state_encoder.encode(observation["observation"], observation["action_mask"]).state_key)
        for action in rules.legal_actions(state):
            visit(rules.apply(state, action))

    visit(rules.initial_state())
    return frozenset(keys)


@cache
def reachable_nonterminal_states() -> tuple[GameState, ...]:
    """Return all reachable non-terminal Tic Tac Toe states."""

    rules = TicTacToeRules()
    states: list[GameState] = []
    seen_states: set[tuple[tuple[int, ...], int]] = set()

    def visit(state: GameState) -> None:
        identity = (state.board, state.current_player)
        if identity in seen_states:
            return
        seen_states.add(identity)
        if state.is_terminal:
            return
        states.append(state)
        for action in rules.legal_actions(state):
            visit(rules.apply(state, action))

    visit(rules.initial_state())
    return tuple(states)


@cache
def _shared_minimax_agent() -> MinimaxAgent:
    """Return a process-wide MinimaxAgent whose negamax cache persists across calls.

    ``minimax_action_match_summary`` evaluates ~3000 reachable states; reusing one agent
    lets the per-instance ``_negamax`` ``lru_cache`` memoize transposing subtrees across
    states instead of recomputing them for every state.
    """

    return MinimaxAgent(TicTacToeRules())


def minimax_action_values(state: GameState) -> tuple[list[float | None], list[int], float]:
    """Return minimax action values, optimal actions, and best value for a state."""

    if state.is_terminal:
        return [None for _ in range(TicTacToeRules().num_actions)], [], 0.0
    rules = TicTacToeRules()
    minimax = _shared_minimax_agent()
    action_values: list[float | None] = [None for _ in range(rules.num_actions)]
    for action in rules.legal_actions(state):
        child = rules.apply(state, action)
        action_values[action] = -minimax._negamax(child, -state.current_player)
    best_value = max(value for value in action_values if value is not None)
    optimal_actions = [action for action, value in enumerate(action_values) if value == best_value]
    return action_values, optimal_actions, float(best_value)


def _greedy_q_matrix(policy: object, observations: list[np.ndarray]) -> np.ndarray:
    """Return one row of Q-values per observation, batched where the policy allows.

    This runs over every reachable state at every policy checkpoint, so the
    per-observation path costs one network query each. Policies that expose a
    batched surface answer all of them in one call; tabular policies have no
    such surface and keep the loop.
    """

    batched = getattr(policy, "q_values_observations", None)
    if callable(batched):
        return np.asarray(batched(np.stack(observations)))
    return np.stack([np.asarray(policy.q_values(observation)) for observation in observations])


def minimax_action_match_summary(policy: object) -> dict[str, object]:
    """Compare a greedy policy against minimax-optimal actions over all reachable states."""

    rules = TicTacToeRules()
    states = reachable_nonterminal_states()
    observations = [rules.observation(state, state.current_player, "flat") for state in states]
    q_matrix = _greedy_q_matrix(policy, [view["observation"] for view in observations])
    matches = 0
    regret_sum = 0.0
    for index, state in enumerate(states):
        observation = observations[index]
        legal_actions = [action for action, value in enumerate(observation["action_mask"]) if int(value) == 1]
        if not legal_actions:
            continue
        action_values, optimal_actions, best_value = minimax_action_values(state)
        q_values = q_matrix[index]
        greedy_action = min(
            legal_actions,
            key=lambda action: (-float(q_values[action]), action),
        )
        if greedy_action in optimal_actions:
            matches += 1
        greedy_value = action_values[greedy_action]
        regret_sum += 0.0 if greedy_value is None else best_value - greedy_value

    total = len(states)
    return {
        "states": total,
        "minimax_matches": matches,
        "minimax_action_match_rate": matches / total if total else 0.0,
        "mean_minimax_regret": regret_sum / total if total else 0.0,
    }


def q_table_population(agent: QLearningAgent, *, mask_token: int = 2) -> dict[str, Any]:
    """Summarize exact Q-table coverage over reachable unmasked Tic Tac Toe states."""

    reachable = reachable_flat_state_keys(agent.encoder)
    unmasked_keys = {key for key in agent.q_table if is_unmasked_tictactoe_key(key, mask_token=mask_token)}
    covered = len(unmasked_keys & reachable)
    reachable_count = len(reachable)
    return {
        "q_table_rows": len(agent.q_table),
        "reachable_state_count": reachable_count,
        "covered_reachable_states": covered,
        "coverage_rate": covered / reachable_count if reachable_count else 0.0,
    }


def minimax_gap(agent: QLearningAgent, state: GameState, *, player: PlayerMark = O) -> dict[str, Any] | None:
    """Compare the Q-greedy legal action with minimax values for a board."""

    if state.is_terminal:
        return None
    rules = TicTacToeRules()
    analysis_state = GameState(
        board=state.board,
        current_player=player,
        move_count=state.move_count,
        winner=state.winner,
        is_draw=state.is_draw,
    )
    action_mask = rules.action_mask(analysis_state)
    legal_actions = [index for index, value in enumerate(action_mask) if int(value) == 1]
    if not legal_actions:
        return None

    observation = rules.observation(analysis_state, player, "flat")["observation"]
    q_values = [agent.q_value(observation, action) for action in range(agent.num_actions)]
    q_action = max(legal_actions, key=lambda action: (q_values[action], -action))

    action_values, optimal_actions, best_value = minimax_action_values(analysis_state)
    q_action_minimax_value = action_values[q_action]
    regret = 0.0 if q_action_minimax_value is None else best_value - q_action_minimax_value
    return {
        "player": player,
        "q_action": q_action,
        "optimal_actions": optimal_actions,
        "minimax_values": action_values,
        "q_values": [q_values[action] if action in legal_actions else None for action in range(agent.num_actions)],
        "regret": regret,
    }


def bellwether_state_keys(encoder: str = "identity") -> dict[str, tuple[int | float, ...]]:
    """Return fixed early-game state keys for monitoring Q-value convergence."""

    rules = TicTacToeRules()
    state_encoder = make_encoder(encoder, num_actions=rules.num_actions)
    empty = rules.initial_state()
    center = rules.apply(empty, 4)
    corner = rules.apply(empty, 0)
    states = {"empty": empty, "x_center": center, "x_corner": corner}
    keys: dict[str, tuple[int | float, ...]] = {}
    for name, state in states.items():
        observation = rules.observation(state, state.current_player, "flat")
        keys[name] = state_encoder.encode(observation["observation"], observation["action_mask"]).state_key
    return keys


def bellwether_values(agent: QLearningAgent) -> dict[str, float]:
    """Return max Q-value for the fixed bellwether states."""

    values: dict[str, float] = {}
    for name, key in bellwether_state_keys(agent.encoder).items():
        row = agent.q_table.get(key)
        values[name] = max(row) if row else 0.0
    return values
