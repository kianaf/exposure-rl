import numpy as np
import pytest

from exposure_rl.oracle import (
    ExpectimaxOracle,
    certainty_games,
    shapley_effects,
    uncertainty_reduction_shapley,
)

# Minimax opponent keeps the occupancy small (144 learner-decision states).
OPPONENT = "minimax"


@pytest.fixture(scope="module")
def setup():
    oracle = ExpectimaxOracle(OPPONENT)
    return oracle, oracle.occupancy()


def greedy(oracle, observation):
    return int(np.argmax(oracle.q_values(observation)))


def test_total_variance_split_and_efficiency(setup):
    oracle, dist = setup
    for point in dist.points[:40]:
        a = greedy(oracle, point.observation)
        games = certainty_games(oracle, dist, point.observation, a)
        assert np.allclose(games.tot, games.opp + games.mask)
        assert np.all(games.mask >= 0) and np.all(games.opp >= -1e-12)
        phi = uncertainty_reduction_shapley(games.mask)
        assert np.isclose(phi.sum(), games.mask[0] - games.mask[-1])


def test_averaged_local_mask_shapley_equals_shapley_effects(setup):
    oracle, dist = setup
    action = 4
    view = dist.legal_action_view(action)
    local = sum(
        point.weight * uncertainty_reduction_shapley(certainty_games(oracle, dist, point.observation, action).mask)
        for point in view.points
    )
    assert np.allclose(local, shapley_effects(oracle, dist, action), atol=1e-10)
