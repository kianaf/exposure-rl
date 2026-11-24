"""PettingZoo AEC environment for Tic Tac Toe."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from xrl.tictactoe.core import GameState, O, ObsMode, PlayerMark, TicTacToeRules, X

AgentID = str

AGENTS: tuple[AgentID, AgentID] = ("player_0", "player_1")
AGENT_TO_MARK: dict[AgentID, PlayerMark] = {"player_0": X, "player_1": O}
MARK_TO_AGENT: dict[int, AgentID] = {X: "player_0", O: "player_1"}


def env(obs_mode: ObsMode = "planes", render_mode: str | None = None):
    """Create the wrapped PettingZoo environment."""

    return wrappers.OrderEnforcingWrapper(raw_env(obs_mode=obs_mode, render_mode=render_mode))


class raw_env(AECEnv):
    """Raw PettingZoo AEC environment.

    Most users should call :func:`env`, which applies PettingZoo's order
    enforcing wrapper.
    """

    metadata = {
        "name": "xrl_tictactoe_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": False,
    }

    def __init__(self, obs_mode: ObsMode = "planes", render_mode: str | None = None):
        super().__init__()
        if obs_mode not in ("planes", "flat"):
            raise ValueError(f"unsupported obs_mode {obs_mode!r}; expected 'planes' or 'flat'")
        if render_mode not in (None, "ansi"):
            raise ValueError(f"unsupported render_mode {render_mode!r}; expected None or 'ansi'")

        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.rules = TicTacToeRules()
        if obs_mode == "planes":
            observation_box = spaces.Box(low=0, high=1, shape=(4, 3, 3), dtype=np.int8)
        else:
            observation_box = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        # Spaces are identical for both agents; build once per instance instead of caching
        # on a bound method (which would pin every instance in a process-wide lru_cache).
        self._observation_space = spaces.Dict(
            {
                "observation": observation_box,
                "action_mask": spaces.MultiBinary(9),
            }
        )
        self._action_space = spaces.Discrete(9)
        self.state: GameState = self.rules.initial_state()
        self.agents: list[AgentID] = []
        self.agent_selection: AgentID = "player_0"
        self.rewards: dict[AgentID, float] = {}
        self._cumulative_rewards: dict[AgentID, float] = {}
        self.terminations: dict[AgentID, bool] = {}
        self.truncations: dict[AgentID, bool] = {}
        self.infos: dict[AgentID, dict[str, Any]] = {}

    def observation_space(self, agent: AgentID):
        self._validate_agent(agent)
        return self._observation_space

    def action_space(self, agent: AgentID):
        self._validate_agent(agent)
        return self._action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self.state = self.rules.initial_state()
        self.agents = list(self.possible_agents)
        self.agent_selection = "player_0"
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

    def observe(self, agent: AgentID) -> dict[str, np.ndarray]:
        self._validate_agent(agent)
        return self.rules.observation(self.state, AGENT_TO_MARK[agent], self.obs_mode)

    def step(self, action: int | None) -> None:
        if self.terminations.get(self.agent_selection, False) or self.truncations.get(self.agent_selection, False):
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("action cannot be None for an active agent")

        agent = self.agent_selection
        legal_actions = set(self.rules.legal_actions(self.state))
        if action not in legal_actions:
            raise ValueError(f"illegal action {action}; legal actions are {sorted(legal_actions)}")

        self._clear_rewards()
        self._cumulative_rewards[agent] = 0.0
        self.state = self.rules.apply(self.state, int(action))

        if self.state.is_terminal:
            returns = self.rules.returns(self.state)
            self.rewards = {"player_0": returns[0], "player_1": returns[1]}
            self.terminations = {agent_id: True for agent_id in self.possible_agents}
            self.agent_selection = self._next_agent(agent)
        else:
            self.agent_selection = MARK_TO_AGENT[self.state.current_player]

        self._accumulate_rewards()

    def render(self) -> str | None:
        rendered = self.rules.render_ansi(self.state)
        if self.render_mode == "ansi":
            return rendered
        return None

    def close(self) -> None:
        return None

    def _clear_rewards(self) -> None:
        self.rewards = {agent: 0.0 for agent in self.agents}

    def _next_agent(self, agent: AgentID) -> AgentID:
        return "player_1" if agent == "player_0" else "player_0"

    def _validate_agent(self, agent: AgentID) -> None:
        if agent not in AGENT_TO_MARK:
            raise ValueError(f"unknown agent {agent!r}; expected one of {self.possible_agents}")
