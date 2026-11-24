"""CLI entrypoint for distributional Q surrogate training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from xrl.explainability.distributional_q_model import DistributionalQSurrogateConfig
from xrl.explainability.distributional_q_trainer import DistributionalQSurrogateTrainer


def train_q_surrogate(
    base_run: Annotated[Path, typer.Option("--base-run", "--run", help="Saved categorical DQN run directory.")],
    out: Path = Path("runs/q_surrogate"),
    seed: int = 0,
    updates: int = 1_000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    loss: str = "cross_entropy",
    dataset: Annotated[
        str,
        typer.Option(help="Training states: 'exact' enumeration (tictactoe), 'rollout', or 'auto' by env."),
    ] = "auto",
    rollout_states: int = 2_048,
    rollout_epsilon: float = 0.1,
    opponent: str = "random",
    perturbed_minimax_probability: float | None = None,
    max_distribution_points: int | None = None,
    full_mask_example_interval: int = 512,
    empty_mask_example_interval: int = 512,
    shapley_probability: float = 0.7,
    uniform_probability: float = 0.3,
    guided_warmup_updates: int = 200,
    guided_refresh_interval: int = 100,
    guided_probability: float = 0.25,
    guided_epsilon: float = 1e-3,
    log_interval: int = 50,
) -> None:
    """Train a masked distributional surrogate for a saved categorical DQN run."""

    config = DistributionalQSurrogateConfig(
        base_run_dir=base_run,
        out_dir=out,
        seed=seed,
        updates=updates,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss=loss,
        dataset=dataset,
        rollout_states=rollout_states,
        rollout_epsilon=rollout_epsilon,
        opponent=opponent,
        perturbed_minimax_probability=perturbed_minimax_probability,
        max_distribution_points=max_distribution_points,
        full_mask_example_interval=full_mask_example_interval,
        empty_mask_example_interval=empty_mask_example_interval,
        shapley_probability=shapley_probability,
        uniform_probability=uniform_probability,
        guided_warmup_updates=guided_warmup_updates,
        guided_refresh_interval=guided_refresh_interval,
        guided_probability=guided_probability,
        guided_epsilon=guided_epsilon,
        log_interval=log_interval,
    )
    _surrogate, summary = DistributionalQSurrogateTrainer(config).train()
    print(json.dumps({"out_dir": str(config.out_dir), "summary": summary}, indent=2))
