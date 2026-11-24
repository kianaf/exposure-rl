import numpy as np
import pytest
from xrl.tictactoe.core import TicTacToeRules

from exposure_rl import opponents
from exposure_rl.scalar_surrogate import ScalarMaskedSurrogate, ScalarSurrogateConfig


def test_opponent_specs():
    assert opponents.parse("random") == ("random", None)
    assert opponents.parse("pm05") == ("perturbed_minimax", 0.05)
    with pytest.raises(ValueError):
        opponents.parse("pm")


def test_perturbed_minimax_mixes_random_and_minimax():
    rules = TicTacToeRules()
    state = rules.initial_state()
    legal = list(rules.legal_actions(state))
    p = opponents.distribution("pm05", state, rules)
    np.testing.assert_allclose(p.sum(), 1.0)
    assert np.all(p[legal] >= 0.05 / len(legal) - 1e-12)  # every legal move keeps its random share
    np.testing.assert_allclose(opponents.distribution("pm05", state, rules, rate=1.0)[legal], 1 / len(legal))


def test_scalar_surrogate_shapes():
    surrogate = ScalarMaskedSurrogate(ScalarSurrogateConfig(base_run_dir="unused"), num_actions=9, observation_size=9)
    q = surrogate.q_batch(np.zeros((4, 9)), np.ones((4, 9)))
    assert q.shape == (4, 9)
    # GPU float32 matmuls (TF32) round differently per batch size; agreement to 1e-3 is the contract
    np.testing.assert_allclose(surrogate.q_values(np.zeros(9), np.ones(9)), q[0], atol=1e-3)
