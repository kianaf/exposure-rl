"""In-memory session state for the local Tic Tac Toe UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xrl.explainability import ExplanationResult, TicTacToeSVERLExplainer
from xrl.tictactoe.core import GameState, O, TicTacToeRules, X
from xrl.tictactoe.replay import ReplayRecorder, status_text
from xrl.training.policies import frozen_random_move_probability
from xrl.ui.trained_runs import TrainedPolicy, TrainedRun, discover_trained_runs, load_trained_policy

SVERL_TARGET_METADATA: dict[str, dict[str, Any]] = {
    "behavior": {
        "description": "Behavior explains why the agent is about to take the pending action.",
        "positive_meaning": "increases the pending action probability",
        "negative_meaning": "decreases the pending action probability",
        "value_unit": "probability",
    },
    "prediction": {
        "description": "Prediction explains the agent's current state-value estimate.",
        "positive_meaning": "raises the predicted value for O",
        "negative_meaning": "lowers the predicted value for O",
        "value_unit": "value",
    },
    "outcome": {
        "description": "Outcome explains the expected game result if play continues under the selected policies.",
        "positive_meaning": "improves the expected result for O",
        "negative_meaning": "worsens the expected result for O",
        "value_unit": "expected return",
    },
}


class TicTacToeSession:
    """In-memory play session for one browser process."""

    def __init__(
        self,
        *,
        runs_dir: str | Path = Path("runs"),
        run_id: str | None = None,
        seed: int | None = None,
        runs: list[TrainedRun] | None = None,
    ) -> None:
        self.rules = TicTacToeRules()
        self.seed = seed
        self.runs_dir = Path(runs_dir)
        self.runs = list(runs) if runs is not None else discover_trained_runs(self.runs_dir)
        self.selected_run: TrainedRun | None = self._initial_run(run_id)
        self.agent: TrainedPolicy | None = self._load_selected_agent()
        self._sverl_explainer: TicTacToeSVERLExplainer | None = None
        self._sverl_run_id: str | None = None
        self.pending_agent_action: int | None = None
        self.state = self.rules.initial_state()
        self.recorder = ReplayRecorder(seed=seed, options=self._recorder_options())

    def reset(self) -> dict[str, Any]:
        self.state = self.rules.initial_state()
        self.pending_agent_action = None
        self.recorder = ReplayRecorder(seed=self.seed, options=self._recorder_options())
        return self.payload()

    def set_run(self, run_id: str) -> dict[str, Any]:
        self.selected_run = self._find_run(run_id)
        self.agent = self._load_selected_agent()
        self._clear_sverl_explainer()
        self.pending_agent_action = None
        return self.reset()

    def refresh_runs(self) -> dict[str, Any]:
        """Refresh run metadata from disk while preserving the selected run when possible."""

        previous_id = None if self.selected_run is None else self.selected_run.id
        self.runs = discover_trained_runs(self.runs_dir)
        self._clear_sverl_explainer()
        if previous_id is not None:
            refreshed = next((run for run in self.runs if run.id == previous_id), None)
            if refreshed is not None:
                self.selected_run = refreshed
                if self.agent is None and refreshed.loadable:
                    self.agent = self._load_selected_agent()
                return self.payload()

        self.selected_run = self._initial_run(None)
        self.agent = self._load_selected_agent()
        if previous_id is None:
            return self.payload()
        self.pending_agent_action = None
        return self.reset()

    def human_move(self, action: int) -> dict[str, Any]:
        if self.agent is None:
            raise ValueError("select a trained run before playing")
        if self.pending_agent_action is not None:
            raise ValueError("step the pending agent move before playing another human move")
        if self.state.is_terminal:
            raise ValueError("game is already over")
        if self.state.current_player != X:
            raise ValueError("it is not the human player's turn")
        self._apply(action)
        if not self.state.is_terminal:
            self.pending_agent_action = self._choose_agent_action()
        return self.payload()

    def agent_move(self) -> None:
        if self.agent is None:
            raise ValueError("select a trained run before playing")
        if self.pending_agent_action is not None:
            self.step_agent()
            return
        self._apply(self._choose_agent_action())

    def step_agent(self) -> dict[str, Any]:
        if self.pending_agent_action is None:
            raise ValueError("no pending agent move")
        action = self.pending_agent_action
        self.pending_agent_action = None
        self._apply(action)
        return self.payload()

    def replay_payload(self) -> dict[str, Any]:
        replay = self.recorder.build()
        return {
            "result": replay.result,
            "actions": list(replay.actions),
            "frames": replay.frames(self.rules),
        }

    def sverl_payload(self) -> dict[str, Any]:
        return self._sverl_payload(compute=True)

    def payload(self) -> dict[str, Any]:
        observation = self.rules.observation(self.state, self.state.current_player, "flat")
        payload: dict[str, Any] = {
            "board": list(self.state.board),
            "current_player": self.state.current_player,
            "status": status_text(self.state),
            "terminal": self.state.is_terminal,
            "winner": self.state.winner,
            "draw": self.state.is_draw,
            "action_mask": observation["action_mask"].astype(int).tolist(),
            "agent": None if self.selected_run is None else self.selected_run.id,
            "agent_pending": self.pending_agent_action is not None,
            "pending_agent_action": self.pending_agent_action,
            "runs": [run.to_json_dict() for run in self.runs],
            "selected_run": None if self.selected_run is None else self.selected_run.to_json_dict(),
            "history": [
                {"agent": move.agent, "action": move.action, "board": list(move.board)}
                for move in self.recorder.build().moves
            ],
        }
        payload.update(self._q_values_payload())
        payload["sverl"] = self._sverl_payload(compute=False)
        return payload

    def _q_values_payload(self) -> dict[str, Any]:
        if self.agent is None or self.state.is_terminal:
            return {
                "q_values_available": self.agent is not None,
                "q_action": None,
                "q_values": None,
                "legal_q_values": None,
                "q_value_support": None,
                "q_value_distributions": None,
            }
        observation = self.rules.observation(
            GameState(
                board=self.state.board,
                current_player=O,
                move_count=self.state.move_count,
                winner=self.state.winner,
                is_draw=self.state.is_draw,
            ),
            O,
            "flat",
        )
        q_values = self.agent.q_values(observation["observation"])
        legal_q_values = [
            q_values[index] if int(observation["action_mask"][index]) == 1 else None for index in range(len(q_values))
        ]
        distributions = self.agent.q_value_distributions(observation["observation"])
        legal_distributions = (
            [
                distributions[index] if int(observation["action_mask"][index]) == 1 else None
                for index in range(len(distributions))
            ]
            if distributions is not None
            else None
        )
        legal_actions = [index for index, value in enumerate(observation["action_mask"]) if int(value) == 1]
        q_action = (
            self.pending_agent_action
            if self.pending_agent_action is not None
            else max(legal_actions, key=lambda action: (q_values[action], -action))
            if legal_actions
            else None
        )
        return {
            "q_values_available": True,
            "q_action": q_action,
            "q_values": q_values,
            "legal_q_values": legal_q_values,
            "q_value_support": self.agent.q_value_support(),
            "q_value_distributions": legal_distributions,
        }

    def _sverl_payload(self, *, compute: bool) -> dict[str, Any]:
        if self.agent is None or self.selected_run is None:
            return self._empty_sverl_payload("Select a trained run to enable SVERL explanations.")
        if self.pending_agent_action is None:
            return self._empty_sverl_payload("Play a move to explain the agent decision.")
        if self.state.is_terminal:
            return self._empty_sverl_payload("SVERL explanations require a non-terminal state.")
        if not compute:
            return self._empty_sverl_payload(
                "Computing exact SVERL by evaluating board-cell coalitions. This can take a moment."
            )

        try:
            explainer = self._get_sverl_explainer()
            targets = {
                "behavior": self._sverl_result_payload(
                    explainer.explain_behavior(self.state, action=self.pending_agent_action, learner_mark=O)
                ),
                "prediction": self._sverl_result_payload(explainer.explain_prediction(self.state, learner_mark=O)),
                "outcome": self._sverl_result_payload(explainer.explain_outcome(self.state, learner_mark=O)),
            }
        except Exception as exc:
            return self._empty_sverl_payload(f"SVERL unavailable: {exc}", error=str(exc))

        return {
            "available": True,
            "error": None,
            "board": list(self.state.board),
            "pending_action": self.pending_agent_action,
            "targets": targets,
        }

    def _get_sverl_explainer(self) -> TicTacToeSVERLExplainer:
        if self.agent is None or self.selected_run is None:
            raise ValueError("select a trained run before explaining")
        if self._sverl_explainer is not None and self._sverl_run_id == self.selected_run.id:
            return self._sverl_explainer

        config = self.selected_run.config
        # SVERL explains the post-training (frozen) opponent: the random-move rate the
        # annealing schedule reached by the final episode. Fall back to a legacy constant
        # rate when viewing runs saved before the discount switch.
        legacy_probability = config.get("perturbed_minimax_probability")
        if legacy_probability is None:
            probability = frozen_random_move_probability(
                float(config.get("perturbed_minimax_discount", 1.0)),
                int(config.get("episodes", 1)),
            )
        else:
            probability = legacy_probability
        self._sverl_explainer = TicTacToeSVERLExplainer(
            self.agent,
            opponent=str(config.get("opponent", "random")),
            perturbed_minimax_probability=probability,
            rules=self.rules,
        )
        self._sverl_run_id = self.selected_run.id
        return self._sverl_explainer

    def _clear_sverl_explainer(self) -> None:
        self._sverl_explainer = None
        self._sverl_run_id = None

    def _sverl_result_payload(self, result: ExplanationResult) -> dict[str, Any]:
        metadata = SVERL_TARGET_METADATA.get(result.target, {})
        return {
            "target": result.target,
            "action": result.action,
            "baseline_value": float(result.baseline_value),
            "full_value": float(result.full_value),
            "delta": float(result.full_value - result.baseline_value),
            "attributions": {key: float(value) for key, value in result.attributions.items()},
            "assumption": result.characteristic_assumption,
            "description": metadata.get("description", ""),
            "positive_meaning": metadata.get("positive_meaning", ""),
            "negative_meaning": metadata.get("negative_meaning", ""),
            "value_unit": metadata.get("value_unit", ""),
            "coalition_count": 2 ** len(result.feature_names),
            "exact": True,
        }

    def _empty_sverl_payload(self, reason: str, *, error: str | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "error": error,
            "reason": reason,
            "pending_action": self.pending_agent_action,
            "targets": {},
        }

    def _choose_agent_action(self) -> int:
        if self.agent is None:
            raise ValueError("select a trained run before playing")
        if self.state.current_player != O:
            raise ValueError("it is not the agent player's turn")
        observation = self.rules.observation(self.state, O, "flat")
        return self.agent(observation, observation["action_mask"], state=self.state, player=O)

    def _initial_run(self, run_id: str | None) -> TrainedRun | None:
        if run_id is not None:
            return self._find_run(run_id)
        return next((run for run in self.runs if run.loadable), None)

    def _find_run(self, run_id: str) -> TrainedRun:
        for run in self.runs:
            if run.id == run_id:
                return run
        raise ValueError(f"unknown trained run {run_id!r}")

    def _load_selected_agent(self) -> TrainedPolicy | None:
        if self.selected_run is None:
            return None
        return load_trained_policy(self.selected_run)

    def _recorder_options(self) -> dict[str, str]:
        if self.selected_run is None:
            return {}
        return {"run_id": self.selected_run.id, "model_type": self.selected_run.model_type}

    def _apply(self, action: int) -> None:
        if action not in self.rules.legal_actions(self.state):
            raise ValueError(f"illegal action {action}")
        self.state = self.recorder.apply(action)
