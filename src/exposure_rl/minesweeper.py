"""Exact belief-space oracle for small Minesweeper (default 3x3, 2 mines).

Minesweeper is a genuine POMDP: the agent observes clues, never the mine layout. For a
deterministic policy the posterior over layouts given an observation is uniform over the layouts
consistent with it (no mine on a revealed cell, every revealed clue matches), because the prior is
uniform and the observation history is a deterministic function of the layout. The first click is
always safe (xrl places mines after it), so at the all-hidden observation the layouts for action
``a`` are those not containing ``a``.

Rules, rewards and flood fill follow ``xrl.minesweeper.core``: a safe reveal pays
``newly_revealed / total_safe``, a mine pays ``-1`` and ends the episode, revealing every safe cell
wins. ``gamma = 1``. The learner plays the belief-optimal policy (ties broken uniformly), and

    Q(o, a)   = E_{L ~ B(o)}[r + V(o')]                   (exact expectimax over beliefs)
    M(o, a)   = E[G^2 | o, a]                              (exact second moment of the return)

so ``Var(G | o, a) = M - Q^2`` is the agent's own outcome uncertainty about hidden mines and its
future. ``action_moments`` and ``occupancy`` match the Tic Tac Toe oracle's interface, so the same
certainty games and scoring code apply unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import combinations

import numpy as np
from xrl.explainability.tictactoe import (
    TicTacToeDistributionPoint,
    TicTacToeStateDistribution,
    tictactoe_feature_spec,
)

HIDDEN = -1


@dataclass(frozen=True)
class Geometry:
    rows: int = 3
    columns: int = 3
    mines: int = 2

    @property
    def cells(self) -> int:
        return self.rows * self.columns


class MinesweeperOracle:
    def __init__(self, geometry: Geometry | None = None) -> None:
        self.g = geometry or Geometry()
        n = self.g.cells
        self.layouts = [frozenset(c) for c in combinations(range(n), self.g.mines)]
        self.neighbours = [self._neighbours(i) for i in range(n)]
        self.total_safe = n - self.g.mines
        self._moments = cache(self._moments_uncached)
        self._value = cache(self._value_uncached)

    # ---------------------------------------------------------------- rules

    def _neighbours(self, i: int) -> tuple[int, ...]:
        r, c = divmod(i, self.g.columns)
        return tuple(
            rr * self.g.columns + cc
            for rr in range(max(0, r - 1), min(self.g.rows, r + 2))
            for cc in range(max(0, c - 1), min(self.g.columns, c + 2))
            if (rr, cc) != (r, c)
        )

    def clue(self, i: int, layout: frozenset[int]) -> int:
        return sum(1 for j in self.neighbours[i] if j in layout)

    def reveal(self, obs: tuple[int, ...], a: int, layout: frozenset[int]) -> tuple[tuple[int, ...], float, bool]:
        """Safe reveal with flood fill; returns ``(next_obs, reward, done)``."""

        values = list(obs)
        queue, newly = [a], 0
        while queue:
            i = queue.pop()
            if values[i] != HIDDEN or i in layout:
                continue
            values[i] = self.clue(i, layout)
            newly += 1
            if values[i] == 0:
                queue.extend(j for j in self.neighbours[i] if values[j] == HIDDEN)
        nxt = tuple(values)
        done = sum(v != HIDDEN for v in nxt) == self.total_safe
        return nxt, newly / self.total_safe, done

    def belief(self, obs: tuple[int, ...], a: int | None = None) -> list[frozenset[int]]:
        """Layouts consistent with ``obs`` (uniform posterior). At the all-hidden board, exclude ``a``."""

        revealed = [i for i, v in enumerate(obs) if v != HIDDEN]
        if not revealed:
            return [L for L in self.layouts if a not in L]
        return [L for L in self.layouts if not (L & set(revealed)) and all(self.clue(i, L) == obs[i] for i in revealed)]

    @staticmethod
    def legal(obs) -> np.ndarray:
        return np.asarray([v == HIDDEN for v in obs], bool)

    # ---------------------------------------------------------------- exact values

    def _moments_uncached(self, obs: tuple[int, ...], a: int) -> tuple[float, float]:
        """``(Q, M) = (E[G], E[G^2])`` for revealing ``a`` at ``obs`` under the belief-optimal policy."""

        layouts = self.belief(obs, a)
        q = m = 0.0
        for L in layouts:
            if a in L:
                q, m = q - 1.0, m + 1.0
                continue
            nxt, r, done = self.reveal(obs, a, L)
            v_next, m_next = (0.0, 0.0) if done else self._value(nxt)
            q += r + v_next
            m += r * r + 2 * r * v_next + m_next
        return q / len(layouts), m / len(layouts)

    def _value_uncached(self, obs: tuple[int, ...]) -> tuple[float, float]:
        """``(V, E[G^2])`` at ``obs`` under the optimal policy, ties uniform."""

        moments = {a: self._moments(obs, a) for a in np.flatnonzero(self.legal(obs))}
        best = max(q for q, _ in moments.values())
        tied = [a for a, (q, _) in moments.items() if np.isclose(q, best)]
        return best, float(np.mean([moments[a][1] for a in tied]))

    def action_moments(self, observation) -> tuple[np.ndarray, np.ndarray]:
        obs = tuple(int(v) for v in np.asarray(observation).reshape(-1))
        first = np.full(self.g.cells, np.nan)
        second = np.full(self.g.cells, np.nan)
        for a in np.flatnonzero(self.legal(obs)):
            first[a], second[a] = self._moments(obs, int(a))
        return first, second

    def q_values(self, observation) -> np.ndarray:
        first, _ = self.action_moments(observation)
        return np.where(np.isnan(first), -np.inf, first)

    # ---------------------------------------------------------------- occupancy

    def occupancy(self) -> TicTacToeStateDistribution:
        """Decision observations reached by the belief-optimal policy, weighted by probability mass.

        Returned in the xrl occupancy container so the exact conditional-expectation machinery
        (ADR 0005) applies unchanged; ``state`` and ``learner_mark`` are unused here.
        """

        start = tuple([HIDDEN] * self.g.cells)
        frontier, points = {start: 1.0}, {}
        while frontier:
            nxt_frontier: dict[tuple[int, ...], float] = {}
            for obs, mass in frontier.items():
                points[obs] = points.get(obs, 0.0) + mass
                q = self.q_values(obs)
                tied = np.flatnonzero(np.isclose(q, q.max()))
                for a in tied:
                    layouts = self.belief(obs, int(a))
                    for L in layouts:
                        if a in L:
                            continue
                        nxt, _, done = self.reveal(obs, int(a), L)
                        if not done:
                            p = mass / len(tied) / len(layouts)
                            nxt_frontier[nxt] = nxt_frontier.get(nxt, 0.0) + p
            frontier = nxt_frontier
        return TicTacToeStateDistribution(
            (
                TicTacToeDistributionPoint(
                    state=None,
                    learner_mark=None,
                    observation=obs,
                    action_mask=tuple(int(v) for v in self.legal(obs)),
                    weight=w,
                )
                for obs, w in points.items()
            ),
            features=tictactoe_feature_spec(),
        )
