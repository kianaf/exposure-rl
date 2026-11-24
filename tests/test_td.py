"""TD training environment: the two checks that would have caught the Minesweeper bugs."""

import numpy as np

from exposure_rl.td import MinesweeperGame, TicTacToeVsOpponent


def test_minesweeper_mine_click_ends_episode_with_minus_one():
    rng = np.random.default_rng(0)
    env = MinesweeperGame(4, 4, 3, rng)
    for _ in range(50):
        env.reset()
        env.step(0)  # the first click is always safe; it places the mines
        if env.layout is None:
            continue
        mine = next(iter(env.layout))
        if env.obs[mine] != env.hidden:
            continue
        _, legal, reward, done = env.step(mine)
        assert reward == -1.0 and done and not legal.any()
        return
    raise AssertionError("no episode reached a clickable mine")


def test_minesweeper_returns_are_bounded_and_wins_sum_to_one():
    rng = np.random.default_rng(1)
    env = MinesweeperGame(4, 4, 3, rng)
    for _ in range(300):
        _, legal = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            _, legal, r, done = env.step(int(rng.choice(np.flatnonzero(legal))))
            total += r
            steps += 1
            assert steps <= 13, "an episode can reveal at most the 13 safe cells"
        assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
        if r > 0:  # won: every safe cell revealed, the reveal rewards sum to one
            assert abs(total - 1.0) < 1e-9


def test_tictactoe_return_is_terminal_reward():
    rng = np.random.default_rng(2)
    env = TicTacToeVsOpponent("random", rng)
    for _ in range(200):
        _, legal = env.reset()
        total, done = 0.0, False
        while not done:
            _, legal, r, done = env.step(int(rng.choice(np.flatnonzero(legal))))
            total += r
        assert total in (-1.0, 0.0, 1.0)
