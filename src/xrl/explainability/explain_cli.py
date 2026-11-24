"""CLI entrypoints for env-agnostic explanations over saved runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from xrl.explainability.adapters import (
    TICTACTOE_ENV_ID,
    LoadedExplanationRun,
    load_explanation_run,
    run_env_id,
)
from xrl.explainability.backgrounds import resolve_background
from xrl.explainability.comparison import COMPARISON_METHODS, GAMES, ComparisonConfig, run_comparison
from xrl.explainability.core import ExplanationResult, validate_action
from xrl.explainability.results import explanation_document, save_explanation
from xrl.explainability.shap_baselines import (
    MarginalShapExplainer,
    ShapBaselineConfig,
    TreeSurrogateShapExplainer,
    greedy_action,
)

EXPLAIN_METHODS = ("exact", "kernel", "cohort-kernel", "tree", "fastshap", "masked-q", "sverl")

# SVERL's four target quantities. Each is a different scalar to explain, so they
# are the *target quantity* axis and not four settings of one method: only
# ``q_value`` is comparable with the rest of the CLI's methods, which all
# explain a fixed action's Q-value.
SVERL_TARGETS = ("behavior", "prediction", "outcome", "q_value")

_BACKGROUND_POLICIES = ("greedy", "epsilon", "random")


def explain(
    run: Annotated[Path, typer.Option("--run", help="Saved training run directory.")],
    method: Annotated[str, typer.Option(help=f"Attribution method: {', '.join(EXPLAIN_METHODS)}.")] = "kernel",
    action: Annotated[int | None, typer.Option(help="Action to explain (default: greedy legal action).")] = None,
    state: Annotated[str, typer.Option(help="State to explain: 'reset' or 'rollout:K' greedy steps.")] = "rollout:0",
    state_seed: Annotated[int, typer.Option(help="Environment seed for the explained state.")] = 0,
    board: Annotated[
        str | None,
        typer.Option(help="TicTacToe-only literal board, 9 chars of x/o/. row-major (overrides --state)."),
    ] = None,
    background: Annotated[Path | None, typer.Option(help="Saved background dataset (.npz).")] = None,
    background_states: int = 256,
    background_seed: int = 1234,
    background_policy: Annotated[
        str, typer.Option(help=f"Rollout policy for auto-collected backgrounds: {', '.join(_BACKGROUND_POLICIES)}.")
    ] = "greedy",
    background_epsilon: float = 0.1,
    save_background: Annotated[
        bool, typer.Option("--save-background", help="Cache the auto-collected background under <run>/explain/.")
    ] = False,
    q_surrogate: Annotated[
        Path | None, typer.Option(help="Masked distributional Q surrogate run (--method masked-q).")
    ] = None,
    fastshap_run: Annotated[Path | None, typer.Option(help="FastSHAP run directory (--method fastshap).")] = None,
    nsamples: Annotated[str, typer.Option(help="KernelSHAP coalition budget: integer or 'auto'.")] = "auto",
    max_background: int = 128,
    cohort_min_size: int = 20,
    num_permutations: Annotated[
        int, typer.Option(help="Permutation budget for sampled masked-q Shapley on large boards.")
    ] = 128,
    sverl_target: Annotated[
        str,
        typer.Option(
            help=(
                "Quantity exact SVERL explains: "
                f"{', '.join(SVERL_TARGETS)}. Only q_value matches the other methods' fixed-action Q."
            )
        ),
    ] = "behavior",
    out: Annotated[Path | None, typer.Option(help="Also write the JSON document to this file.")] = None,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Persist the JSON document under <run>/explain/results/.")
    ] = True,
    render: Annotated[bool, typer.Option("--render/--no-render", help="Print an ANSI attribution grid.")] = False,
) -> None:
    """Explain a saved run's Q-values with SHAP-style feature attributions."""

    if method not in EXPLAIN_METHODS:
        raise typer.BadParameter(f"method must be one of {EXPLAIN_METHODS}, got {method!r}")
    if background_policy not in _BACKGROUND_POLICIES:
        raise typer.BadParameter(f"background-policy must be one of {_BACKGROUND_POLICIES}")
    if sverl_target not in SVERL_TARGETS:
        raise typer.BadParameter(f"sverl-target must be one of {SVERL_TARGETS}, got {sverl_target!r}")
    if method == "sverl":
        _require_tictactoe_for_sverl(run_env_id(run))
    loaded = load_explanation_run(run)
    observation, action_mask, game_state = _resolve_state(loaded, state_spec=state, state_seed=state_seed, board=board)
    target_action = greedy_action(loaded.policy, observation, action_mask) if action is None else int(action)
    try:
        # The shared guard raises ValueError; the option label keeps the message
        # in the caller's vocabulary without rewriting the string here.
        target_action = validate_action(target_action, action_mask, num_actions=loaded.num_actions, label="--action")
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    started = time.perf_counter()
    result, extras = _build_result(
        loaded,
        method,
        observation,
        action_mask,
        target_action,
        game_state=game_state,
        background_path=background,
        background_states=background_states,
        background_seed=background_seed,
        background_policy=background_policy,
        background_epsilon=background_epsilon,
        save_background=save_background,
        q_surrogate=q_surrogate,
        fastshap_run=fastshap_run,
        nsamples=nsamples,
        max_background=max_background,
        cohort_min_size=cohort_min_size,
        num_permutations=num_permutations,
        sverl_target=sverl_target,
    )

    document = explanation_document(
        method=method,
        env_id=loaded.env_id,
        run_dir=run,
        model_type=loaded.model_type,
        state_source="board" if board is not None else state,
        state_seed=state_seed,
        observation=observation,
        action_mask=action_mask,
        result=result,
        extras=extras,
        duration_seconds=time.perf_counter() - started,
    )
    if save:
        document["id"] = save_explanation(run, document).stem
    text = json.dumps(document, indent=2)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if render:
        typer.echo(loaded.adapter.render_attributions(observation, result, loaded.config))


def train_fastshap(
    base_run: Annotated[Path, typer.Option("--base-run", "--run", help="Saved base training run directory.")],
    out: Path = Path("runs/fastshap"),
    q_surrogate: Annotated[
        Path | None,
        typer.Option(help="Masked Q surrogate run used as the value function (default: marginal imputation)."),
    ] = None,
    value_function: Annotated[
        str,
        typer.Option(
            help=(
                "Coalition game to train against: auto (surrogate when --q-surrogate is given, else marginal), "
                "surrogate, marginal, or direct (the run's own jointly trained masked Q-function)."
            )
        ),
    ] = "auto",
    background: Annotated[Path | None, typer.Option(help="Saved background dataset (.npz).")] = None,
    background_states: int = 1_024,
    seed: int = 0,
    updates: int = 2_000,
    batch_size: int = 32,
    coalitions_per_state: int = 8,
    learning_rate: float = 1e-3,
    marginal_samples: int = 16,
    log_interval: int = 50,
) -> None:
    """Train an amortized FastSHAP explainer for a saved run."""

    from xrl.explainability.fastshap import FastShapConfig, FastShapTrainer

    try:
        config = FastShapConfig(
            base_run_dir=base_run,
            out_dir=out,
            q_surrogate_dir=q_surrogate,
            value_function=value_function,
            background_path=background,
            background_states=background_states,
            seed=seed,
            updates=updates,
            batch_size=batch_size,
            coalitions_per_state=coalitions_per_state,
            learning_rate=learning_rate,
            marginal_samples=marginal_samples,
            log_interval=log_interval,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    _explainer, summary = FastShapTrainer(config).train()
    print(json.dumps({"out_dir": str(out), "summary": summary}, indent=2))


def _build_result(
    loaded: LoadedExplanationRun,
    method: str,
    observation: np.ndarray,
    action_mask: np.ndarray,
    target_action: int,
    *,
    game_state: Any,
    background_path: Path | None,
    background_states: int,
    background_seed: int,
    background_policy: str,
    background_epsilon: float,
    save_background: bool,
    q_surrogate: Path | None,
    fastshap_run: Path | None,
    nsamples: str,
    max_background: int,
    cohort_min_size: int,
    num_permutations: int,
    sverl_target: str = "behavior",
) -> tuple[ExplanationResult, dict[str, Any]]:
    if method in {"exact", "kernel", "cohort-kernel", "tree"}:
        dataset, dataset_source = resolve_background(
            loaded.adapter,
            loaded.policy,
            loaded.config,
            loaded.run_dir,
            background_path=background_path,
            states=background_states,
            seed=background_seed,
            policy_name=background_policy,
            epsilon=background_epsilon,
            save=save_background,
        )
        config = ShapBaselineConfig(
            max_background=max_background,
            kernel_nsamples=_parse_nsamples(nsamples),
            cohort_min_size=cohort_min_size,
            seed=background_seed,
        )
        extras: dict[str, Any] = {"background": {"states": dataset.size, "source": dataset_source}}
        if method == "tree":
            tree_explainer = TreeSurrogateShapExplainer(loaded.policy, loaded.feature_spec, dataset, config=config)
            result = tree_explainer.explain_action(observation, action_mask, action=target_action)
            extras["surrogate_fidelity_r2"] = tree_explainer.surrogate_fit(target_action).fidelity_r2
            return result, extras
        explainer = MarginalShapExplainer(
            loaded.policy,
            loaded.feature_spec,
            dataset,
            method=method,
            cohort_key=loaded.adapter.cohort_key,
            config=config,
        )
        return explainer.explain_action(observation, action_mask, action=target_action), extras

    if method == "masked-q":
        result = _masked_q_result(
            loaded,
            observation,
            action_mask,
            target_action,
            q_surrogate,
            num_permutations,
        )
        extras = {"artifact": str(q_surrogate)} if q_surrogate is not None else {}
        return result, extras

    if method == "fastshap":
        if fastshap_run is None:
            raise typer.BadParameter("--fastshap-run is required for --method fastshap")
        from xrl.explainability.fastshap import load_fastshap_explainer, validate_fastshap_base_run

        try:
            validate_fastshap_base_run(fastshap_run, loaded.run_dir)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        fastshap_explainer = load_fastshap_explainer(fastshap_run, base_run_dir=loaded.run_dir)
        result = fastshap_explainer.explain_action(observation, action=target_action)
        return result, {"artifact": str(fastshap_run)}

    return _sverl_result(loaded, game_state, target_action, sverl_target)


def _masked_q_result(
    loaded: LoadedExplanationRun,
    observation: np.ndarray,
    action_mask: np.ndarray,
    target_action: int,
    q_surrogate: Path | None,
    num_permutations: int,
) -> ExplanationResult:
    from xrl.explainability.distributional_q_explainer import direct_q_explanation, surrogate_explanation

    if q_surrogate is not None:
        try:
            return surrogate_explanation(
                q_surrogate,
                observation,
                target_action,
                action_mask=action_mask,
                base_run_dir=loaded.run_dir,
                feature_names=loaded.feature_spec.names,
                num_permutations=num_permutations,
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    if bool(loaded.config.get("feature_mask_input", False)) and float(loaded.config.get("masked_batch_ratio", 0.0)) > 0:
        if not callable(getattr(loaded.policy, "masked_q_values", None)):
            raise typer.BadParameter(
                "run config declares positive masked training but the loaded policy has no masked_q_values API"
            )
        return direct_q_explanation(
            loaded.policy,
            observation,
            target_action,
            action_mask=action_mask,
            feature_names=loaded.feature_spec.names,
            num_permutations=num_permutations,
        )
    raise typer.BadParameter(
        "--method masked-q needs feature_mask_input=true and masked_batch_ratio>0 or an explicit --q-surrogate"
    )


def _require_tictactoe_for_sverl(env_id: str) -> None:
    if env_id != TICTACTOE_ENV_ID:
        raise typer.BadParameter(
            "exact SVERL is tictactoe-only: its conditional-occupancy state enumeration is intractable "
            f"for {env_id}; use --method kernel or --method masked-q instead"
        )


def _sverl_result(
    loaded: LoadedExplanationRun,
    game_state: Any,
    target_action: int,
    sverl_target: str,
) -> tuple[ExplanationResult, dict[str, Any]]:
    _require_tictactoe_for_sverl(loaded.env_id)
    if game_state is None:
        raise typer.BadParameter("--method sverl needs a rollout state (--state rollout:K), not --board")
    from xrl.explainability.tictactoe import TicTacToeSVERLExplainer, run_opponent_law

    opponent, randomness = run_opponent_law(loaded.config)
    explainer = TicTacToeSVERLExplainer(
        loaded.policy,
        opponent=opponent,
        perturbed_minimax_probability=randomness,
    )
    # Which occupancy the explanation marginalized over is part of the answer,
    # not an implementation detail: the same policy explained against a
    # different opponent law is a different game.
    extras = {"opponent": opponent, "opponent_randomness": randomness}
    if sverl_target == "prediction":
        return explainer.explain_prediction(game_state), extras
    if sverl_target == "outcome":
        return explainer.explain_outcome(game_state), extras
    if sverl_target == "q_value":
        return explainer.explain_q_value(game_state, action=target_action), extras
    return explainer.explain_behavior(game_state, action=target_action), extras


def _resolve_state(
    loaded: LoadedExplanationRun,
    *,
    state_spec: str,
    state_seed: int,
    board: str | None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if board is not None:
        if loaded.env_id != TICTACTOE_ENV_ID:
            raise typer.BadParameter("--board is only supported for tictactoe runs")
        observation, action_mask = _tictactoe_board_state(board)
        return observation, action_mask, None
    steps = _parse_state_spec(state_spec)
    env = loaded.make_env()
    try:
        if hasattr(env, "agent_iter"):
            return _aec_rollout_state(env, loaded, steps=steps, seed=state_seed)
        return _gym_rollout_state(env, loaded, steps=steps, seed=state_seed)
    finally:
        env.close()


def _parse_state_spec(state_spec: str) -> int:
    if state_spec == "reset":
        return 0
    if state_spec.startswith("rollout:"):
        suffix = state_spec.removeprefix("rollout:")
        if suffix.isdigit():
            return int(suffix)
    raise typer.BadParameter(f"state must be 'reset' or 'rollout:K', got {state_spec!r}")


def _gym_rollout_state(env, loaded: LoadedExplanationRun, *, steps: int, seed: int):
    observation, _info = env.reset(seed=seed)
    for step in range(steps):
        obs_vector, mask = _split_observation(observation)
        chosen = greedy_action(loaded.policy, obs_vector, mask)
        observation, _reward, terminated, truncated, _info = env.step(chosen)
        if terminated or truncated:
            raise typer.BadParameter(
                f"episode ended after {step + 1} greedy steps; requested rollout:{steps} (try another --state-seed)"
            )
    obs_vector, mask = _split_observation(observation)
    return obs_vector, mask, None


def _aec_rollout_state(env, loaded: LoadedExplanationRun, *, steps: int, seed: int):
    env.reset(seed=seed)
    for plies, _agent_id in enumerate(env.agent_iter()):
        observation, _reward, terminated, truncated, _info = env.last()
        if terminated or truncated:
            raise typer.BadParameter(
                f"game ended after {plies} plies; requested rollout:{steps} (try another --state-seed)"
            )
        obs_vector, mask = _split_observation(observation)
        if plies == steps:
            return obs_vector, mask, env.unwrapped.state
        env.step(greedy_action(loaded.policy, obs_vector, mask))
    raise typer.BadParameter(f"could not reach rollout:{steps} before the episode ended")


def _split_observation(observation) -> tuple[np.ndarray, np.ndarray]:
    obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
    mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
    return obs_vector, mask


def _tictactoe_board_state(board: str) -> tuple[np.ndarray, np.ndarray]:
    cleaned = board.strip().lower()
    if len(cleaned) != 9 or any(char not in "xo." for char in cleaned):
        raise typer.BadParameter("--board must be 9 characters of x/o/. in row-major order")
    marks = np.asarray([1 if char == "x" else -1 if char == "o" else 0 for char in cleaned], dtype=np.int8)
    x_count = int(np.sum(marks == 1))
    o_count = int(np.sum(marks == -1))
    if x_count - o_count not in {0, 1}:
        raise typer.BadParameter(f"--board has an impossible move count (x={x_count}, o={o_count})")
    mover = 1 if x_count == o_count else -1
    observation = (marks * mover).astype(np.float32)
    action_mask = (marks == 0).astype(np.int8)
    return observation, action_mask


def _parse_nsamples(nsamples: str) -> int | str:
    if nsamples == "auto":
        return "auto"
    if nsamples.isdigit():
        return int(nsamples)
    raise typer.BadParameter(f"nsamples must be an integer or 'auto', got {nsamples!r}")


def compare_explanations(
    run: Annotated[Path, typer.Option("--run", help="Saved training run directory.")],
    out: Annotated[Path, typer.Option("--out", help="Directory for the study's CSV/JSON artifacts.")],
    methods: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated methods, each '<game>:<estimator>' or a shorthand "
                f"({', '.join(COMPARISON_METHODS)}). Games: {', '.join(GAMES)}."
            )
        ),
    ] = ",".join(COMPARISON_METHODS),
    states: Annotated[int, typer.Option(help="Number of held-out probe states to explain.")] = 64,
    state_seed: Annotated[
        int, typer.Option(help="Seed stream for probe states (keep distinct from the background).")
    ] = 987654,
    background_states: int = 256,
    background_seed: int = 1234,
    background_policy: Annotated[
        str, typer.Option(help=f"Rollout policy for the imputation background: {', '.join(_BACKGROUND_POLICIES)}.")
    ] = "greedy",
    max_background: int = 64,
    nsamples: Annotated[str, typer.Option(help="Marginal KernelSHAP coalition budget: integer or 'auto'.")] = "auto",
    kernel_coalitions: Annotated[int, typer.Option(help="Coalition budget for KernelSHAP over a masked-Q game.")] = 512,
    cohort_min_size: int = 20,
    num_permutations: Annotated[int, typer.Option(help="Permutation budget for sampled masked-q Shapley.")] = 64,
    top_k: Annotated[int, typer.Option(help="Top-k size for the Jaccard agreement statistic.")] = 5,
    q_surrogate: Annotated[Path | None, typer.Option(help="Masked distributional Q surrogate run.")] = None,
    fastshap_run: Annotated[
        str | None,
        typer.Option(
            help=(
                "FastSHAP run directories, comma-separated. Each artifact is matched to the game it "
                "records, so one study can carry an amortized arm per game."
            )
        ),
    ] = None,
    faithfulness: Annotated[
        bool, typer.Option("--faithfulness/--no-faithfulness", help="Score deletion/insertion curves.")
    ] = True,
    faithfulness_stride: Annotated[
        int, typer.Option(help="Evaluate every k-th curve step; raise this on large boards.")
    ] = 1,
    references: Annotated[
        bool, typer.Option("--references/--no-references", help="Score approximations against their exact reference.")
    ] = True,
    coalition_fidelity: Annotated[
        bool,
        typer.Option(
            "--coalition-fidelity/--no-coalition-fidelity",
            help=(
                "Compare each learned coalition game with the exact conditional one over every "
                "coalition. Separates model error from a difference in semantics; tictactoe-only."
            ),
        ),
    ] = True,
    plots: Annotated[
        bool, typer.Option("--plots/--no-plots", help="Render figures (needs the monitoring extra).")
    ] = True,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress per-state progress output.")] = False,
) -> None:
    """Compare explanation methods over many states and write a study directory."""

    if background_policy not in _BACKGROUND_POLICIES:
        raise typer.BadParameter(f"background-policy must be one of {_BACKGROUND_POLICIES}")
    selected = tuple(name.strip() for name in methods.split(",") if name.strip())
    try:
        config = ComparisonConfig(
            methods=selected,
            states=states,
            state_seed=state_seed,
            background_states=background_states,
            background_seed=background_seed,
            background_policy=background_policy,
            max_background=max_background,
            kernel_nsamples=_parse_nsamples(nsamples),
            kernel_coalitions=kernel_coalitions,
            cohort_min_size=cohort_min_size,
            num_permutations=num_permutations,
            top_k=top_k,
            faithfulness=faithfulness,
            faithfulness_stride=faithfulness_stride,
            references=references,
            coalition_fidelity=coalition_fidelity,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    report = run_comparison(
        run,
        config=config,
        q_surrogate=q_surrogate,
        fastshap_run=fastshap_run,
        progress=None if quiet else lambda message: typer.echo(f"  {message}", err=True),
    )
    written = report.write(out)
    if plots:
        from xrl.explainability.comparison_plots import plot_comparison

        written.extend(plot_comparison(out))

    for method, reason in sorted(report.skipped.items()):
        typer.echo(f"skipped {method}: {reason}", err=True)
    if report.failures:
        typer.echo(f"{len(report.failures)} per-state failures recorded in summary.json", err=True)
    typer.echo(
        json.dumps(
            {
                "out": str(out),
                "states": report.state_count,
                "methods": list(report.methods),
                "skipped": report.skipped,
                "files": [path.name for path in written],
            },
            indent=2,
        )
    )
