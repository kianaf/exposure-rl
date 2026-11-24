"""Small local browser UI for playing or replaying Minesweeper DQN runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from xrl._optional import explain_dependencies_available, neural_dependencies_available
from xrl._provenance import RunInstanceRef, payload_matches_run
from xrl.envs import minesweeper_v0
from xrl.explainability.adapters import MINESWEEPER_ENV_ID, get_explanation_adapter
from xrl.explainability.backgrounds import BackgroundDataset, resolve_background
from xrl.explainability.results import explanation_document, json_safe, list_explanations, save_explanation
from xrl.minesweeper import EXPLODED_MINE, HIDDEN, MinesweeperConfig
from xrl.training.minesweeper_dqn import MinesweeperDQNConfig, MinesweeperDQNTrainer
from xrl.training.value_heads import SCALAR, is_categorical, normalize_value_head
from xrl.ui.minesweeper_page import HTML
from xrl.ui.server import resolve_runs_dir


class MinesweeperPolicy(Protocol):
    """Greedy Minesweeper policy loaded from a saved DQN run."""

    num_actions: int

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, *, epsilon: float) -> int:
        """Return one legal action."""

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return Q-values in environment action coordinates."""

    def q_distributions(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return categorical support and per-action probabilities, when available."""


@dataclass(frozen=True, slots=True)
class MinesweeperRun:
    """Metadata for one saved Minesweeper DQN run."""

    id: str
    path: Path
    model_type: str
    loadable: bool
    status: str
    config: dict[str, Any]
    summary: dict[str, Any]
    latest_evaluation: dict[str, Any]

    @property
    def label(self) -> str:
        rows = self.config.get("rows", "?")
        columns = self.config.get("columns", "?")
        mines = self.config.get("mines", "?")
        episodes = self.config.get("episodes", "?")
        base = f"{self.model_type} {rows}x{columns}/{mines} ({episodes} episodes)"
        tags = self.tags
        return f"{base} · {' · '.join(tags)}" if tags else base

    @property
    def tags(self) -> list[str]:
        """Short badges for training choices the base label does not show."""

        tags: list[str] = []
        if self.config.get("exploration") == "boltzmann":
            tags.append("boltzmann")
        if self.config.get("backup") == "soft":
            tags.append("soft")
        if tags and "temperature" in self.config:
            tags.append(f"τ={_format_number(self.config['temperature'])}")
        encoder = self.config.get("encoder", "identity")
        if encoder != "identity":
            tags.append(str(encoder))
        return tags

    @property
    def model_card(self) -> list[list[str]]:
        """Short human-readable facts about the model, rendered in the UI."""

        config = self.config
        if config.get("exploration") == "boltzmann":
            exploration = f"boltzmann (τ={_format_number(config.get('temperature', 0.05))})"
        else:
            epsilon_start = _format_number(config.get("epsilon_start", 1.0))
            epsilon_end = _format_number(config.get("epsilon_end", 0.05))
            exploration = f"ε-greedy {epsilon_start}→{epsilon_end}"
        backup = "soft (τ·logsumexp)" if config.get("backup") == "soft" else "hard max"
        card = [
            ["Run", self.id],
            ["Value head", _value_head_label(config)],
            ["Exploration", exploration],
            ["Backup", backup],
        ]
        hidden_sizes = config.get("hidden_sizes") or []
        if hidden_sizes:
            card.append(["Network", "MLP " + " × ".join(str(size) for size in hidden_sizes)])
        if "gamma" in config:
            card.append(["Discount", f"γ = {_format_number(config['gamma'])}"])
        if "seed" in config:
            card.append(["Seed", str(config["seed"])])
        return card

    @property
    def training_summary(self) -> dict[str, Any]:
        keys = [
            "episodes",
            "seed",
            "rows",
            "columns",
            "mines",
            "encoder",
            "gamma",
            "learning_rate",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay_episodes",
            "exploration",
            "backup",
            "temperature",
            "value_head",
            "replay_capacity",
            "replay_warmup",
            "batch_size",
            "target_sync_interval",
        ]
        return {key: self.config[key] for key in keys if key in self.config}

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "model_type": self.model_type,
            "label": self.label,
            "tags": self.tags,
            "model_card": self.model_card,
            "loadable": self.loadable,
            "status": self.status,
            "config": self.config,
            "summary": self.summary,
            "latest_evaluation": self.latest_evaluation,
            "training": self.training_summary,
        }


AgentLoader = Callable[[MinesweeperRun], MinesweeperPolicy]


# Attribution methods offered in the browser, with reduced live-compute budgets:
# the CLI keeps higher-quality defaults (nsamples "auto", 128 permutations) for
# offline use; the UI trades some accuracy for click-latency.
UI_EXPLAIN_METHODS: tuple[tuple[str, str], ...] = (
    ("kernel", "KernelSHAP"),
    ("cohort-kernel", "Cohort KernelSHAP"),
    ("tree", "Tree surrogate"),
    ("masked-q", "Masked Q"),
    ("fastshap", "FastSHAP"),
)
EXPLAIN_BACKGROUND_STATES = 128
EXPLAIN_BACKGROUND_SEED = 1234
EXPLAIN_MAX_BACKGROUND = 64
EXPLAIN_KERNEL_NSAMPLES = 512
EXPLAIN_NUM_PERMUTATIONS = 64
EXPLAIN_HISTORY_LIMIT = 50


class MinesweeperSession:
    """In-memory Minesweeper play session for one browser process."""

    def __init__(
        self,
        *,
        runs_dir: str | Path = Path("runs"),
        run_id: str | None = None,
        seed: int = 0,
        runs: list[MinesweeperRun] | None = None,
        agent_loader: AgentLoader | None = None,
    ) -> None:
        self.seed = int(seed)
        self.runs_dir = Path(runs_dir)
        self.runs = list(runs) if runs is not None else discover_minesweeper_runs(self.runs_dir)
        self.selected_run: MinesweeperRun | None = self._initial_run(run_id)
        self.agent_loader = agent_loader or load_minesweeper_policy
        self.agent: MinesweeperPolicy | None = self._load_selected_agent()
        self.env = None
        self.observation: dict[str, np.ndarray] = {}
        self.info: dict[str, Any] = {}
        self.total_reward = 0.0
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        self._explain_background: tuple[BackgroundDataset, str] | None = None
        self._explain_artifacts: dict[str, list[str]] | None = None
        self._explain_explainers: dict[str, Any] = {}
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        if self.env is not None:
            self.env.close()
        self.env = self._make_env()
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.total_reward = 0.0
        self.step_count = 0
        self.history = [self._history_frame(actor="initial", action=None, reward=0.0)]
        return self.payload()

    def new_board(self) -> dict[str, Any]:
        """Start the next deterministic board in the seed sequence."""

        return self.reset(seed=self.seed + 1)

    def refresh_runs(self) -> dict[str, Any]:
        previous_id = None if self.selected_run is None else self.selected_run.id
        self._clear_explain_cache()
        self.runs = discover_minesweeper_runs(self.runs_dir)
        if previous_id is not None:
            refreshed = next((run for run in self.runs if run.id == previous_id), None)
            if refreshed is not None:
                self.selected_run = refreshed
                if not refreshed.loadable:
                    self.agent = None
                elif self.agent is None:
                    self.agent = self._load_selected_agent()
                return self.payload()

        self.selected_run = self._initial_run(None)
        self.agent = self._load_selected_agent()
        return self.reset()

    def set_run(self, run_id: str) -> dict[str, Any]:
        self.selected_run = self._find_run(run_id)
        self.agent = self._load_selected_agent()
        self._clear_explain_cache()
        return self.reset()

    def reveal(self, action: int) -> dict[str, Any]:
        return self._apply(action, actor="human")

    def step_agent(self) -> dict[str, Any]:
        if self.agent is None:
            raise ValueError("select a loadable Minesweeper DQN run before stepping the agent")
        action = self._choose_agent_action()
        return self._apply(action, actor="agent")

    def payload(self) -> dict[str, Any]:
        config = self._board_config()
        observation = np.asarray(self.observation.get("observation", []), dtype=np.int8)
        action_mask = np.asarray(self.observation.get("action_mask", []), dtype=np.int8)
        safe = int(self.info.get("safe_cells_revealed", 0))
        total_safe = int(self.info.get("total_safe_cells", config.total_safe_cells))
        payload: dict[str, Any] = {
            "rows": config.rows,
            "columns": config.columns,
            "mines": config.mines,
            "cells": observation.astype(int).tolist(),
            "action_mask": action_mask.astype(int).tolist(),
            "terminal": bool(self.env.board.is_terminal),
            "outcome": self.info.get("outcome"),
            "status": self._status(),
            "safe_cells_revealed": safe,
            "total_safe_cells": total_safe,
            "completion_ratio": float(self.info.get("completion_ratio", 0.0)),
            "total_reward": self.total_reward,
            "step_count": self.step_count,
            "seed": self.seed,
            "runs": [run.to_json_dict() for run in self.runs],
            "selected_run": None if self.selected_run is None else self.selected_run.to_json_dict(),
            "history": list(self.history),
            "mine_positions": self._visible_mines(),
            "exploded_action": self.env.board.exploded_action,
            "hidden_value": HIDDEN,
            "exploded_value": EXPLODED_MINE,
        }
        payload.update(self._q_values_payload(observation, action_mask))
        payload["explain"] = self.explain_stub()
        return payload

    def _q_policy_label(self) -> str:
        config = {} if self.selected_run is None else self.selected_run.config
        if config.get("exploration") == "boltzmann":
            return f"softmax(τ={config.get('temperature', 0.05):g})"
        return "greedy"

    def _q_values_payload(self, observation: np.ndarray, action_mask: np.ndarray) -> dict[str, Any]:
        if self.agent is None:
            return {
                "q_values_available": False,
                "q_action": None,
                "q_values": None,
                "legal_q_values": None,
                "q_error": None,
                "q_distribution_available": False,
                "q_support": None,
                "legal_q_distributions": None,
                "q_distribution_error": None,
            }
        if self.env.board.is_terminal:
            return {
                "q_values_available": True,
                "q_policy_label": self._q_policy_label(),
                "q_action": None,
                "q_values": None,
                "legal_q_values": None,
                "q_error": None,
                "q_distribution_available": False,
                "q_support": None,
                "legal_q_distributions": None,
                "q_distribution_error": None,
            }
        try:
            values = np.asarray(self.agent.q_values(observation), dtype=np.float32)
        except Exception as exc:
            return {
                "q_values_available": False,
                "q_action": None,
                "q_values": None,
                "legal_q_values": None,
                "q_error": str(exc),
                "q_distribution_available": False,
                "q_support": None,
                "legal_q_distributions": None,
                "q_distribution_error": None,
            }
        legal_actions = [index for index, value in enumerate(action_mask) if int(value) == 1]
        q_action = max(legal_actions, key=lambda action: (float(values[action]), -action)) if legal_actions else None
        payload = {
            "q_values_available": True,
            "q_policy_label": self._q_policy_label(),
            "q_action": q_action,
            "q_values": [float(value) for value in values],
            "legal_q_values": [
                float(values[index]) if int(action_mask[index]) == 1 else None for index in range(len(values))
            ],
            "q_error": None,
            "q_distribution_available": False,
            "q_support": None,
            "legal_q_distributions": None,
            "q_distribution_error": None,
        }
        distribution_reader = getattr(self.agent, "q_distributions", None)
        if distribution_reader is None:
            return payload

        try:
            result = distribution_reader(observation)
            if result is None:
                return payload
            support, distributions = result
            support = np.asarray(support, dtype=np.float32)
            distributions = np.asarray(distributions, dtype=np.float32)
            expected_shape = (len(values), len(support))
            if support.ndim != 1 or distributions.shape != expected_shape:
                raise ValueError(
                    f"invalid categorical Q shape: support {support.shape}, "
                    f"distributions {distributions.shape}, expected {expected_shape}"
                )
            payload.update(
                {
                    "q_distribution_available": True,
                    "q_support": [float(value) for value in support],
                    "legal_q_distributions": [
                        [float(probability) for probability in distributions[index]]
                        if int(action_mask[index]) == 1
                        else None
                        for index in range(len(values))
                    ],
                }
            )
        except Exception as exc:
            payload["q_distribution_error"] = str(exc)
        return payload

    def explain_stub(self) -> dict[str, Any]:
        """Return method availability without computing (embedded in /state)."""

        if self.selected_run is None or self.agent is None:
            return {
                "available": False,
                "reason": "Select a loadable Minesweeper DQN run to enable explanations.",
                "methods": [],
            }
        methods = self._method_entries()
        if self.env.board.is_terminal:
            return {"available": False, "reason": "Explanations require a non-terminal board.", "methods": methods}
        if not any(entry["available"] for entry in methods):
            return {"available": False, "reason": "No explanation method is currently available.", "methods": methods}
        return {"available": True, "reason": None, "methods": methods}

    def explain(self, method: str) -> dict[str, Any]:
        """Compute, persist, and return one attribution for the current board."""

        known = {entry_id for entry_id, _label in UI_EXPLAIN_METHODS}
        if method not in known:
            raise ValueError(f"unknown explanation method {method!r}; expected one of {sorted(known)}")
        run, policy = self.selected_run, self.agent
        if run is None or policy is None:
            return self._explain_unavailable(
                "Select a loadable Minesweeper DQN run to enable explanations.", method=method
            )
        if self.env.board.is_terminal:
            return self._explain_unavailable("Explanations require a non-terminal board.", method=method)
        entry = next(item for item in self._method_entries() if item["id"] == method)
        if not entry["available"]:
            return self._explain_unavailable(str(entry["reason"]), method=method)

        observation = np.asarray(self.observation["observation"], dtype=np.float32).reshape(-1)
        action_mask = np.asarray(self.observation["action_mask"], dtype=np.int8).reshape(-1)
        try:
            started = time.perf_counter()
            result, extras = self._compute_explanation(method, observation, action_mask, run=run, policy=policy)
            duration = time.perf_counter() - started
        except Exception as exc:
            return self._explain_unavailable(f"Explanation failed: {exc}", method=method, error=str(exc))

        config = self._board_config()
        document = explanation_document(
            method=method,
            env_id=MINESWEEPER_ENV_ID,
            run_dir=run.path,
            model_type=run.model_type,
            state_source=f"ui:seed{self.seed}:step{self.step_count}",
            state_seed=self.seed,
            observation=observation,
            action_mask=action_mask,
            result=result,
            extras={**extras, "layout": {"rows": config.rows, "columns": config.columns}},
            duration_seconds=duration,
        )
        save_error = None
        try:
            document["id"] = save_explanation(run.path, document).stem
        except OSError as exc:
            # Display must not depend on disk state; surface the failure instead.
            save_error = str(exc)
        return json_safe(
            {
                "available": True,
                "error": None,
                "reason": None,
                "method": method,
                "id": document.get("id"),
                "save_error": save_error,
                "delta": result.full_value - result.baseline_value,
                "document": document,
            }
        )

    def saved_explanations(self, limit: int | None = EXPLAIN_HISTORY_LIMIT) -> dict[str, Any]:
        """List persisted explanation documents for the selected run."""

        if self.selected_run is None:
            return {"run_id": None, "explanations": []}
        return {
            "run_id": self.selected_run.id,
            "explanations": list_explanations(self.selected_run.path, limit=limit),
        }

    def _explain_unavailable(self, reason: str, *, method: str | None = None, error: str | None = None):
        return {"available": False, "error": error, "reason": reason, "method": method}

    def _method_entries(self) -> list[dict[str, Any]]:
        artifacts = self._discover_explain_artifacts()
        explain_ok = explain_dependencies_available()
        neural_ok = neural_dependencies_available()
        train_hint = {
            "masked-q": (
                "This run has no trained masked-input path and no masked-Q surrogate "
                "(xrl train-q-surrogate --run <run>)."
            ),
            "fastshap": "No FastSHAP explainer trained for this run (xrl train-fastshap --run <run>).",
        }
        entries: list[dict[str, Any]] = []
        declares_direct_masked_q = bool(
            self.selected_run
            and self.selected_run.config.get("feature_mask_input", False)
            and float(self.selected_run.config.get("masked_batch_ratio", 0.0)) > 0
        )
        has_direct_masked_q = declares_direct_masked_q and callable(getattr(self.agent, "masked_q_values", None))
        for method, label in UI_EXPLAIN_METHODS:
            reason: str | None = None
            if method in {"kernel", "cohort-kernel", "tree"}:
                if not explain_ok:
                    reason = "Install the 'explain' extra (uv sync --extra explain) to enable SHAP methods."
            elif (
                method == "masked-q" and declares_direct_masked_q and not has_direct_masked_q and not artifacts[method]
            ):
                reason = "Run config declares positive masked training, but the policy has no masked_q_values API."
            elif not artifacts[method] and (method != "masked-q" or not has_direct_masked_q):
                reason = train_hint[method]
            elif not neural_ok:
                reason = f"{label} explanations require the 'neural' extra."
            entries.append({"id": method, "label": label, "available": reason is None, "reason": reason})
        return entries

    def _compute_explanation(
        self,
        method: str,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        run: MinesweeperRun,
        policy: MinesweeperPolicy,
    ):
        from xrl.explainability.shap_baselines import greedy_action

        adapter = get_explanation_adapter(MINESWEEPER_ENV_ID)
        feature_spec = adapter.feature_spec(run.config)
        action = greedy_action(policy, observation, action_mask)

        if method in {"kernel", "cohort-kernel", "tree"}:
            from xrl.explainability.shap_baselines import (
                MarginalShapExplainer,
                ShapBaselineConfig,
                TreeSurrogateShapExplainer,
            )

            background, background_source = self._resolve_explain_background(adapter, run=run, policy=policy)
            extras: dict[str, Any] = {"background": {"states": background.size, "source": background_source}}
            explainer = self._explain_explainers.get(method)
            if explainer is None:
                config = ShapBaselineConfig(
                    max_background=EXPLAIN_MAX_BACKGROUND,
                    kernel_nsamples=EXPLAIN_KERNEL_NSAMPLES,
                    seed=EXPLAIN_BACKGROUND_SEED,
                )
                if method == "tree":
                    explainer = TreeSurrogateShapExplainer(policy, feature_spec, background, config=config)
                else:
                    explainer = MarginalShapExplainer(
                        policy,
                        feature_spec,
                        background,
                        method=method,
                        cohort_key=adapter.cohort_key,
                        config=config,
                    )
                self._explain_explainers[method] = explainer
            result = explainer.explain_action(observation, action_mask, action=action)
            if method == "tree":
                extras["surrogate_fidelity_r2"] = explainer.surrogate_fit(action).fidelity_r2
            return result, extras

        artifacts = self._discover_explain_artifacts()
        if method == "masked-q":
            if (
                bool(run.config.get("feature_mask_input", False))
                and float(run.config.get("masked_batch_ratio", 0.0)) > 0
                and callable(getattr(policy, "masked_q_values", None))
            ):
                from xrl.explainability.distributional_q_explainer import direct_q_explanation

                result = direct_q_explanation(
                    policy,
                    observation,
                    action,
                    action_mask=action_mask,
                    feature_names=feature_spec.names,
                    num_permutations=EXPLAIN_NUM_PERMUTATIONS,
                    seed=EXPLAIN_BACKGROUND_SEED,
                )
                return result, {}
            from xrl.explainability.distributional_q_explainer import surrogate_explanation

            artifact = artifacts["masked-q"][0]
            result = surrogate_explanation(
                artifact,
                observation,
                action,
                action_mask=action_mask,
                base_run_dir=run.path,
                feature_names=feature_spec.names,
                num_permutations=EXPLAIN_NUM_PERMUTATIONS,
                seed=EXPLAIN_BACKGROUND_SEED,
            )
            return result, {"artifact": artifact}

        artifact = artifacts["fastshap"][0]
        explainer = self._explain_explainers.get(method)
        if explainer is None:
            from xrl.explainability.fastshap import load_fastshap_explainer, validate_fastshap_base_run

            validate_fastshap_base_run(artifact, run.path)
            explainer = load_fastshap_explainer(Path(artifact), base_run_dir=run.path)
            self._explain_explainers[method] = explainer
        return explainer.explain_action(observation, action=action), {"artifact": artifact}

    def _resolve_explain_background(
        self,
        adapter: Any,
        *,
        run: MinesweeperRun,
        policy: MinesweeperPolicy,
    ) -> tuple[BackgroundDataset, str]:
        if self._explain_background is None:
            # Rollout collection needs action_distribution; fall back to a
            # uniform-random walk (and matching cache name) when it is absent.
            rollout_policy = policy if hasattr(policy, "action_distribution") else None
            self._explain_background = resolve_background(
                adapter,
                rollout_policy,
                run.config,
                run.path,
                states=EXPLAIN_BACKGROUND_STATES,
                seed=EXPLAIN_BACKGROUND_SEED,
                policy_name="greedy" if rollout_policy is not None else "random",
                save=True,
            )
        return self._explain_background

    def _discover_explain_artifacts(self) -> dict[str, list[str]]:
        if self._explain_artifacts is None:
            self._explain_artifacts = {
                "masked-q": self._matching_artifacts("surrogate_config.json"),
                "fastshap": self._matching_artifacts("fastshap_config.json"),
            }
        return self._explain_artifacts

    def _matching_artifacts(self, config_name: str) -> list[str]:
        if self.selected_run is None:
            return []
        # Discovery shares one decision with the validators that run at explain
        # time. They used to disagree, so the UI would offer an artifact that
        # explaining then refused. See ADR 0023.
        ref = RunInstanceRef.of(self.selected_run.path)
        matches: list[tuple[float, str]] = []
        for config_path in self.runs_dir.rglob(config_name):
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                matched = payload_matches_run(config, artifact_dir=config_path.parent, ref=ref)
            except (OSError, json.JSONDecodeError, ValueError):
                # An artifact whose provenance cannot be decided is simply not
                # offered. Skipping errs the safe way round: the method is then
                # reported unavailable, rather than listed and refused later.
                continue
            if matched:
                matches.append((config_path.stat().st_mtime, str(config_path.parent)))
        return [path for _mtime, path in sorted(matches, reverse=True)]

    def _clear_explain_cache(self) -> None:
        self._explain_background = None
        self._explain_artifacts = None
        self._explain_explainers = {}

    def _apply(self, action: int, *, actor: str) -> dict[str, Any]:
        action = int(action)
        if self.env.board.is_terminal:
            raise ValueError("game is already over")
        if action < 0 or action >= self._board_config().cells:
            raise ValueError(f"action must be in [0, {self._board_config().cells - 1}], got {action}")
        if int(self.observation["action_mask"][action]) != 1:
            raise ValueError(f"illegal action {action}: cell is already revealed")
        self.observation, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += float(reward)
        self.step_count += 1
        self.history.append(self._history_frame(actor=actor, action=action, reward=float(reward)))
        if terminated or truncated:
            self.info["outcome"] = self.env.board.outcome
        return self.payload()

    def _history_frame(self, *, actor: str, action: int | None, reward: float) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "actor": actor,
            "action": action,
            "reward": float(reward),
            "safe_cells_revealed": int(self.info.get("safe_cells_revealed", 0)),
            "completion_ratio": float(self.info.get("completion_ratio", 0.0)),
            "outcome": self.info.get("outcome"),
            "cells": np.asarray(self.observation.get("observation", []), dtype=np.int8).astype(int).tolist(),
        }

    def _choose_agent_action(self) -> int:
        observation = np.asarray(self.observation["observation"], dtype=np.float32)
        action_mask = np.asarray(self.observation["action_mask"], dtype=np.int8)
        return int(self.agent.select_action(observation, action_mask, epsilon=0.0))  # type: ignore[union-attr]

    def _make_env(self):
        config = self._board_config()
        return minesweeper_v0.env(rows=config.rows, columns=config.columns, mines=config.mines)

    def _board_config(self) -> MinesweeperConfig:
        if self.selected_run is None:
            return MinesweeperConfig()
        return _config_geometry(self.selected_run.config)

    def _status(self) -> str:
        outcome = self.info.get("outcome")
        if outcome == "win":
            return "Win"
        if outcome == "value_head":
            return "Loss"
        return "In progress"

    def _visible_mines(self) -> list[int]:
        if not self.env.board.is_terminal:
            return []
        mines = self.env.board.mine_positions or frozenset()
        return sorted(int(action) for action in mines)

    def _initial_run(self, run_id: str | None) -> MinesweeperRun | None:
        if run_id is not None:
            return self._find_run(run_id)
        return next((run for run in self.runs if run.loadable), None)

    def _find_run(self, run_id: str) -> MinesweeperRun:
        for run in self.runs:
            if run.id == run_id:
                return run
        raise ValueError(f"unknown Minesweeper run {run_id!r}")

    def _load_selected_agent(self) -> MinesweeperPolicy | None:
        if self.selected_run is None:
            return None
        if not self.selected_run.loadable:
            return None
        return self.agent_loader(self.selected_run)


def discover_minesweeper_runs(runs_dir: str | Path) -> list[MinesweeperRun]:
    """Return loadable-looking Minesweeper DQN runs found below ``runs_dir``."""

    root = Path(runs_dir)
    if not root.exists():
        return []

    runs: list[MinesweeperRun] = []
    for config_path in sorted(root.rglob("config.json")):
        run_dir = config_path.parent
        run = _run_from_dir(root, run_dir)
        if run is not None:
            runs.append(run)
    runs.sort(key=_run_sort_key, reverse=True)
    return runs


def load_minesweeper_policy(run: MinesweeperRun | str, runs: list[MinesweeperRun] | None = None) -> MinesweeperPolicy:
    """Load a greedy Minesweeper DQN policy from a discovered run."""

    selected = run if isinstance(run, MinesweeperRun) else _find_run(run, runs or [])
    if not selected.loadable:
        raise RuntimeError(selected.status)
    config = MinesweeperDQNConfig.from_json_dict(selected.config)
    return MinesweeperDQNTrainer(config).load_agent(selected.path)


def make_handler(session: MinesweeperSession, *, html: str | None = None):
    """Build a request handler bound to one Minesweeper session."""

    html = HTML if html is None else html
    session_lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                if self.path == "/":
                    self._send_html(html)
                elif self.path == "/state":
                    with session_lock:
                        self._send_json(session.payload())
                elif self.path == "/runs":
                    with session_lock:
                        session.refresh_runs()
                        self._send_json({"runs": [run.to_json_dict() for run in session.runs]})
                elif self.path == "/explanations":
                    with session_lock:
                        self._send_json(session.saved_explanations())
                else:
                    self._send_problem(404, "Not found")
            except Exception as exc:
                self._send_problem(400, str(exc))

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                with session_lock:
                    if self.path == "/reveal":
                        self._send_json(session.reveal(int(body["action"])))
                    elif self.path == "/agent-step":
                        self._send_json(session.step_agent())
                    elif self.path == "/reset":
                        seed = body.get("seed")
                        self._send_json(session.reset(seed=None if seed in (None, "") else int(seed)))
                    elif self.path == "/new":
                        self._send_json(session.new_board())
                    elif self.path == "/refresh-runs":
                        self._send_json(session.refresh_runs())
                    elif self.path == "/run":
                        self._send_json(session.set_run(str(body.get("run_id", ""))))
                    elif self.path == "/explain":
                        self._send_json(session.explain(str(body.get("method", ""))))
                    else:
                        self._send_problem(404, "Not found")
            except Exception as exc:
                self._send_problem(400, str(exc))

        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_problem(self, status: int, message: str) -> None:
            data = json.dumps({"error": message}).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, content: str) -> None:
            data = content.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def play_minesweeper(
    host: str = "127.0.0.1",
    port: int = 8767,
    runs_dir: Path = Path("runs"),
    run_id: str | None = None,
    seed: int = 0,
) -> None:
    """Run a local Minesweeper browser UI."""

    resolved_runs_dir = resolve_runs_dir(runs_dir)
    session = MinesweeperSession(runs_dir=resolved_runs_dir, run_id=run_id, seed=seed)
    server = ThreadingHTTPServer((host, port), make_handler(session))
    print(f"Serving xrl Minesweeper at http://{host}:{port}")
    print(f"Using runs dir: {resolved_runs_dir} ({len(session.runs)} Minesweeper runs)")
    server.serve_forever()


def _run_from_dir(root: Path, run_dir: Path) -> MinesweeperRun | None:
    try:
        config = _read_json(run_dir / "config.json")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict) or not _is_minesweeper_config(config):
        return None

    model_type = f"minesweeper_dqn_{config.get('value_head', 'regression')}"
    loadable = True
    status = "ready"
    if not (run_dir / MinesweeperDQNTrainer.model_filename).exists():
        loadable = False
        status = f"Missing {MinesweeperDQNTrainer.model_filename}."
    elif not neural_dependencies_available():
        loadable = False
        status = "Neural dependencies are unavailable; install xrl[neural]."

    return MinesweeperRun(
        id=_run_id(root, run_dir),
        path=run_dir,
        model_type=model_type,
        loadable=loadable,
        status=status,
        config=config,
        summary=_read_optional_json(run_dir / "summary.json"),
        latest_evaluation=_latest_row(run_dir / "eval_metrics.csv"),
    )


def _is_minesweeper_config(config: dict[str, Any]) -> bool:
    if config.get("env_id") == "minesweeper_v0":
        return True
    env_options = config.get("env_options")
    if isinstance(env_options, dict) and {"rows", "columns", "mines"}.issubset(env_options):
        return True
    return {"rows", "columns", "mines"}.issubset(config) and "opponent" not in config


def _config_geometry(config: dict[str, Any]) -> MinesweeperConfig:
    env_options = dict(config.get("env_options", {}) or {})
    return MinesweeperConfig(
        rows=int(config.get("rows", env_options.get("rows", 5))),
        columns=int(config.get("columns", env_options.get("columns", 5))),
        mines=int(config.get("mines", env_options.get("mines", 3))),
    )


def _find_run(run_id: str, runs: list[MinesweeperRun]) -> MinesweeperRun:
    for run in runs:
        if run.id == run_id:
            return run
    raise ValueError(f"unknown Minesweeper run {run_id!r}")


def _run_sort_key(run: MinesweeperRun) -> tuple[str, float, str]:
    return (_timestamp_token(run.id), _path_mtime(run.path), run.id)


def _timestamp_token(value: str) -> str:
    match = re.search(r"(20\d{6})[-T]?(\d{6})", value)
    return "" if match is None else "".join(match.groups())


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _value_head_label(config: dict[str, Any]) -> str:
    head = normalize_value_head(str(config.get("value_head", SCALAR)))
    if not is_categorical(head):
        return head
    bins = config.get("num_bins", "?")
    v_min = _format_number(config.get("v_min", "?"))
    v_max = _format_number(config.get("v_max", "?"))
    return f"{head}, {bins} bins on [{v_min}, {v_max}]"


def _run_id(root: Path, run_dir: Path) -> str:
    try:
        return run_dir.relative_to(root).as_posix()
    except ValueError:
        return run_dir.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _latest_row(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    return rows[-1] if rows else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    except OSError:
        return []


def _coerce_value(value: str | None) -> Any:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped in {"True", "False"}:
        return stripped == "True"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Minesweeper browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    play_minesweeper(host=args.host, port=args.port, runs_dir=args.runs_dir, run_id=args.run_id, seed=args.seed)


if __name__ == "__main__":
    main()
