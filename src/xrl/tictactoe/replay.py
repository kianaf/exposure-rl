"""Replay recording and visualization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from xrl.tictactoe.core import GameState, O, TicTacToeRules, X


def agent_for_mark(mark: int) -> str:
    if mark == X:
        return "player_0"
    if mark == O:
        return "player_1"
    raise ValueError(f"unknown player mark {mark}")


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """One replay step."""

    agent: str
    action: int
    rewards: tuple[float, float]
    board: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Replay:
    """A complete deterministic Tic Tac Toe replay."""

    seed: int | None
    options: dict[str, Any]
    actions: tuple[int, ...]
    moves: tuple[MoveRecord, ...]
    final_board: tuple[int, ...]
    result: str

    def frames(self, rules: TicTacToeRules | None = None) -> list[dict[str, Any]]:
        """Return simple serializable frame dictionaries."""

        rules = rules or TicTacToeRules()
        state = rules.initial_state()
        output = [
            {
                "ply": 0,
                "agent": agent_for_mark(state.current_player),
                "action": None,
                "board": state.board,
                "status": status_text(state),
            }
        ]
        for ply, action in enumerate(self.actions, start=1):
            acting_agent = agent_for_mark(state.current_player)
            state = rules.apply(state, action)
            output.append(
                {
                    "ply": ply,
                    "agent": acting_agent,
                    "action": action,
                    "board": state.board,
                    "status": status_text(state),
                }
            )
        return output


@dataclass(slots=True)
class ReplayRecorder:
    """Record moves while applying them through the core rules."""

    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    rules: TicTacToeRules = field(default_factory=TicTacToeRules)
    state: GameState = field(init=False)
    _actions: list[int] = field(init=False, repr=False)
    _moves: list[MoveRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state = self.rules.initial_state()
        self._actions: list[int] = []
        self._moves: list[MoveRecord] = []

    def apply(self, action: int) -> GameState:
        agent = agent_for_mark(self.state.current_player)
        self.state = self.rules.apply(self.state, action)
        rewards = self.rules.returns(self.state) if self.state.is_terminal else (0.0, 0.0)
        self._actions.append(action)
        self._moves.append(MoveRecord(agent=agent, action=action, rewards=rewards, board=self.state.board))
        return self.state

    def build(self) -> Replay:
        return Replay(
            seed=self.seed,
            options=dict(self.options),
            actions=tuple(self._actions),
            moves=tuple(self._moves),
            final_board=self.state.board,
            result=status_text(self.state),
        )


@dataclass(slots=True)
class ReplayViewer:
    """Step forward/backward through a replay."""

    replay: Replay
    rules: TicTacToeRules = field(default_factory=TicTacToeRules)
    index: int = field(init=False, default=0)
    _states: list[GameState] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._states = [self.rules.initial_state()]
        state = self._states[0]
        for action in self.replay.actions:
            state = self.rules.apply(state, action)
            self._states.append(state)
        self.index = 0

    @property
    def state(self) -> GameState:
        return self._states[self.index]

    def next(self) -> GameState:
        self.index = min(self.index + 1, len(self._states) - 1)
        return self.state

    def previous(self) -> GameState:
        self.index = max(self.index - 1, 0)
        return self.state

    def render_ansi(self) -> str:
        return self.rules.render_ansi(self.state)


def replay_actions(
    actions: Iterable[int],
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    rules: TicTacToeRules | None = None,
) -> Replay:
    """Build a replay from an action history."""

    recorder = ReplayRecorder(seed=seed, options=options or {}, rules=rules or TicTacToeRules())
    for action in actions:
        recorder.apply(action)
    return recorder.build()


def status_text(state: GameState) -> str:
    if state.winner == X:
        return "X wins"
    if state.winner == O:
        return "O wins"
    if state.is_draw:
        return "Draw"
    return f"{'X' if state.current_player == X else 'O'} to move"
