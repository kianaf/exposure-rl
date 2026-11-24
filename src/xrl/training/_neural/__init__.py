"""Internal neural-training building blocks.

Houses the replay buffer, distributional value-projection helpers, the
``BaseAECTrainer`` base class, and the learner adapters shared by the DQN
(:mod:`xrl.training.neural_q`) and Q-regression
(:mod:`xrl.training.q_regression`) trainers.

Private package: import the public trainers and their re-exported symbols from
``xrl.training.neural_q`` / ``xrl.training.q_regression`` rather than reaching
into here directly.
"""
