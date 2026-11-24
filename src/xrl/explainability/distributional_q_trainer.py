"""Training and artifact I/O for distributional Q surrogates."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from xrl._provenance import provenance_payload
from xrl.explainability.adapters import TICTACTOE_ENV_ID, LoadedExplanationRun, load_explanation_run
from xrl.explainability.core import ExactShapleyEstimator, SampledShapleyEstimator
from xrl.explainability.distributional_q_dataset import (
    SurrogateTrainingDataset,
    build_rollout_surrogate_dataset,
    build_surrogate_training_dataset,
    load_categorical_dqn_agent,
    require_categorical_agent,
)
from xrl.explainability.distributional_q_explainer import EXACT_SHAPLEY_MAX_FEATURES, DistributionalQExplainer
from xrl.explainability.distributional_q_masks import CoalitionMaskSampler
from xrl.explainability.distributional_q_model import DistributionalQSurrogateConfig, MaskedDistributionalQSurrogate
from xrl.training.flops import learner_flops


class DistributionalQSurrogateTrainer:
    """Train and persist a masked distributional Q surrogate."""

    def __init__(self, config: DistributionalQSurrogateConfig) -> None:
        self.config = config

    def train(self) -> tuple[MaskedDistributionalQSurrogate, dict[str, Any]]:
        run = load_explanation_run(self.config.base_run_dir)
        base_agent = require_categorical_agent(run.policy)
        dataset = self._build_dataset(run)
        # The saved surrogate config must carry the true feature count so
        # load_surrogate can rebuild the network without the base dataset.
        config = replace(self.config, feature_count=int(dataset.observations.shape[1]))
        surrogate = MaskedDistributionalQSurrogate(
            config,
            num_actions=base_agent.num_actions,
            observation_size=dataset.observations.shape[1],
            support=np.asarray(base_agent.support, dtype=np.float32),
        )
        metrics = self._train_surrogate(surrogate, dataset)
        summary = self._write_artifacts(surrogate, dataset, metrics)
        return surrogate, summary

    def _build_dataset(self, run: LoadedExplanationRun) -> SurrogateTrainingDataset:
        mode = self.config.dataset
        if mode == "auto":
            mode = "exact" if run.env_id == TICTACTOE_ENV_ID else "rollout"
        if mode == "exact":
            if run.env_id != TICTACTOE_ENV_ID:
                raise ValueError(
                    f"the exact conditional-occupancy dataset requires enumerating every reachable state and "
                    f"is only tractable for {TICTACTOE_ENV_ID}; use dataset='rollout' for {run.env_id}"
                )
            return build_surrogate_training_dataset(
                run.policy,
                opponent=self.config.opponent,
                perturbed_minimax_probability=self.config.perturbed_minimax_probability,
                max_points=self.config.max_distribution_points,
                seed=self.config.seed,
            )
        return build_rollout_surrogate_dataset(
            run.policy,
            env_id=run.env_id,
            env_config=run.config,
            states=self.config.rollout_states,
            seed=self.config.seed,
            policy_epsilon=self.config.rollout_epsilon,
        )

    def load_surrogate(self, run_dir: Path | None = None) -> MaskedDistributionalQSurrogate:
        run_dir = run_dir or self.config.out_dir
        config = DistributionalQSurrogateConfig.from_json_dict(
            json.loads((run_dir / "surrogate_config.json").read_text(encoding="utf-8"))
        )
        base_agent = load_categorical_dqn_agent(config.base_run_dir)
        surrogate = MaskedDistributionalQSurrogate(
            config,
            num_actions=base_agent.num_actions,
            observation_size=config.feature_count,
            support=np.asarray(base_agent.support, dtype=np.float32),
        )
        surrogate.load_params(run_dir / "surrogate.msgpack")
        return surrogate

    def _train_surrogate(
        self,
        surrogate: MaskedDistributionalQSurrogate,
        dataset: SurrogateTrainingDataset,
    ) -> list[dict[str, float | int]]:
        rng = np.random.default_rng(self.config.seed)
        # The surrogate's config carries the dataset-derived feature count.
        sampler = CoalitionMaskSampler(surrogate.config.mask_sampler_config, seed=self.config.seed)
        metrics: list[dict[str, float | int]] = []
        attribution_magnitudes: np.ndarray | None = None
        for update in range(self.config.updates):
            if update >= self.config.guided_warmup_updates and update % self.config.guided_refresh_interval == 0:
                attribution_magnitudes = self._estimate_attribution_magnitudes(surrogate, dataset, rng)
            indices = rng.choice(dataset.size, size=self.config.batch_size, replace=True, p=dataset.weights)
            known_masks = sampler.sample_batch(
                self.config.batch_size,
                example_offset=update * self.config.batch_size,
                attribution_magnitudes=attribution_magnitudes,
            )
            loss = surrogate.update(
                dataset.observations[indices],
                known_masks,
                dataset.target_distributions[indices],
                dataset.action_masks[indices],
            )
            if update % self.config.log_interval == 0 or update == self.config.updates - 1:
                metrics.append({"update": update + 1, "loss": loss})
        return metrics

    def _estimate_attribution_magnitudes(
        self,
        surrogate: MaskedDistributionalQSurrogate,
        dataset: SurrogateTrainingDataset,
        rng: np.random.Generator,
    ) -> np.ndarray:
        point_index = int(rng.choice(dataset.size, p=dataset.weights))
        observation = dataset.observations[point_index]
        action_mask = dataset.action_masks[point_index]
        full_mask = np.ones(observation.shape[0], dtype=np.int8)
        q_values = surrogate.q_values(observation, full_mask)
        legal = [action for action, value in enumerate(action_mask) if int(value) == 1]
        action = max(legal, key=lambda candidate: float(q_values[candidate]))
        explainer = DistributionalQExplainer(surrogate, estimator=self._shapley_estimator(observation.shape[0]))
        explanation = explainer.explain_action(observation, action_mask, action=action)
        return np.asarray([abs(value) for value in explanation.attributions.values()], dtype=np.float64)

    def _shapley_estimator(self, feature_count: int) -> ExactShapleyEstimator | SampledShapleyEstimator:
        if feature_count <= EXACT_SHAPLEY_MAX_FEATURES:
            return ExactShapleyEstimator()
        return SampledShapleyEstimator(num_permutations=32, seed=self.config.seed)

    def _write_artifacts(
        self,
        surrogate: MaskedDistributionalQSurrogate,
        dataset: SurrogateTrainingDataset,
        metrics: list[dict[str, float | int]],
    ) -> dict[str, Any]:
        out_dir = self.config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_config = {
            **surrogate.config.to_json_dict(),
            **provenance_payload(surrogate.config.base_run_dir),
        }
        (out_dir / "surrogate_config.json").write_text(json.dumps(artifact_config, indent=2), encoding="utf-8")
        surrogate.save(out_dir / "surrogate.msgpack")
        self._write_metrics(out_dir / "surrogate_metrics.csv", metrics)
        summary = {
            "updates": self.config.updates,
            "training_points": dataset.size,
            "final_loss": float(metrics[-1]["loss"]) if metrics else math.nan,
            "base_run_dir": str(self.config.base_run_dir),
            "surrogate": "distributional_q",
            # Charged on top of the base DQN when comparing against direct
            # mask-aware training, which spends nothing on distillation.
            **learner_flops(surrogate),
        }
        (out_dir / "surrogate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _write_metrics(self, path: Path, metrics: list[dict[str, float | int]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["update", "loss"])
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)


def load_distributional_q_surrogate(run_dir: Path) -> MaskedDistributionalQSurrogate:
    """Load a saved distributional Q surrogate run."""

    config = DistributionalQSurrogateConfig.from_json_dict(
        json.loads((run_dir / "surrogate_config.json").read_text(encoding="utf-8"))
    )
    return DistributionalQSurrogateTrainer(config).load_surrogate(run_dir)
