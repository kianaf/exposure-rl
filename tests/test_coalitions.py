import numpy as np

from exposure_rl import coalitions
from exposure_rl.minesweeper import HIDDEN, MinesweeperOracle
from exposure_rl.oracle import certainty_games, uncertainty_reduction_shapley


def test_subset_sum_games_match_dense_operator():
    oracle = MinesweeperOracle()
    dist = oracle.occupancy()
    points = np.array([p.observation for p in dist.points])
    weights = np.array([p.weight for p in dist.points])
    moments = [oracle.action_moments(p) for p in points]
    q = np.stack([m[0] for m in moments])
    g2 = np.stack([m[1] for m in moments])
    for i in range(0, len(points), 7):
        a = int(np.nanargmax(q[i]))
        dense = certainty_games(oracle, dist, dist.points[i].observation, a)
        legal = points[:, a] == HIDDEN
        _, v_mask, v_state = coalitions.certainty_games(
            points[legal],
            weights[legal],
            q[legal, a],
            g2[legal, a] - q[legal, a] ** 2,
            points[i],
            q[i, a],
            g2[i, a] - q[i, a] ** 2,
        )
        np.testing.assert_allclose(v_mask, dense.mask, atol=1e-12)
        np.testing.assert_allclose(v_state, dense.opp, atol=1e-12)
        np.testing.assert_allclose(
            coalitions.uncertainty_reduction_shapley(v_mask, 9), uncertainty_reduction_shapley(dense.mask), atol=1e-12
        )


def test_shapley_efficiency_16_features():
    rng = np.random.default_rng(0)
    v = rng.normal(size=1 << 16)
    np.testing.assert_allclose(coalitions.shapley(v, 16).sum(), v[-1] - v[0], atol=1e-9)
