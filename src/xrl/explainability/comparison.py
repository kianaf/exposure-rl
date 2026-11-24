"""Multi-state comparison harness for the explanation baselines.

A comparison between attribution methods is only meaningful when the run, the
state, and the imputation background are held fixed and only the method
varies. A comparison on *one* state is an anecdote regardless: attributions
vary enormously between an opening position and a nearly-solved board, so this
harness evaluates every method over a population of states and reports
bootstrap intervals.

Four separate questions are answered, and conflating them is the usual
mistake. The first three compare attributions; the last compares the coalition
games those attributions are computed from, and is the only one that can say
whether a disagreement is about semantics or about model error:

* **Agreement** (:mod:`xrl.explainability.agreement`) — do two methods rank
  the same features as important? Valid across families, which is the only
  comparison that is, because families do not share a value scale.
* **Reference error** — how far is an approximation from the exact answer to
  *its own* game? Valid only inside a family, and the reference is selected by
  rule rather than declared: the outcome playing the same game whose
  ``shapley_estimator`` is ``exact``. Where no such outcome exists — every
  masked-Q estimate on a board past
  :data:`~xrl.explainability.distributional_q_explainer.EXACT_SHAPLEY_MAX_FEATURES`
  — no reference rows are written at all, because the alternative is a column
  that silently means error-against-truth in one environment and
  estimator-against-estimator in another.
* **Faithfulness** (:mod:`xrl.explainability.faithfulness`) — does the model
  actually behave as the attribution claims? Scored under the marginal value
  function, under the masked-surrogate one when a surrogate is supplied, under
  the direct masked-Q one when the base run was trained with a mask input and a
  positive masked batch ratio, and under exact conditional occupancy when a method plays
  that game — because scoring only under the marginal game favours the marginal
  family, and scoring only under learned games judges every method by an
  approximation. A method that wins under every evaluator is the only robust
  conclusion.
* **Coalition fidelity** — how far is a learned coalition game from the exact
  conditional one it targets, measured over ``v`` itself before any Shapley
  transform? Without it, a low cross-family agreement is unattributable: the
  masked-Q games are *trained* to reproduce the conditional game, so their
  distance from it is model error, and reading that distance as a difference in
  coalition semantics credits the model's failure to the theory. Runs only where
  the exact game exists and every coalition can be enumerated.

Probe states are drawn with a different seed stream than the imputation
background, so methods are never evaluated on the states used to impute them.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xrl._optional import explain_dependencies_available, neural_dependencies_available
from xrl.explainability.adapters import TICTACTOE_ENV_ID, LoadedExplanationRun, load_explanation_run
from xrl.explainability.agreement import (
    attribution_regression,
    bootstrap_mean_ci,
    coalition_error,
    compare_attributions,
    efficiency_residual,
    per_feature_agreement,
    reference_error,
)
from xrl.explainability.backgrounds import BackgroundDataset, resolve_background
from xrl.explainability.cohorts import cohort_background
from xrl.explainability.core import (
    CachedCharacteristicFunction,
    ExplanationResult,
    ExplanationSemantics,
    all_coalitions,
)
from xrl.explainability.faithfulness import (
    DELETION,
    INSERTION,
    control_curves,
    perturbation_curves,
    summarize_curves,
)
from xrl.explainability.results import json_safe
from xrl.explainability.shap_baselines import (
    EXACT_MARGINAL_MAX_FEATURES,
    MarginalShapExplainer,
    ShapBaselineConfig,
    TreeSurrogateShapExplainer,
    greedy_action,
    marginal_characteristic,
)

__all__ = [
    "COMPARISON_METHODS",
    "EXACT_ESTIMATOR",
    "FAMILY_LABELS",
    "GAMES",
    "GAME_ESTIMATORS",
    "METHOD_ALIASES",
    "ComparisonConfig",
    "ComparisonReport",
    "MethodOutcome",
    "MethodSpec",
    "family_key",
    "family_label",
    "fastshap_paths",
    "parse_method",
    "run_comparison",
]

COMPARISON_METHODS = ("exact", "kernel", "cohort-kernel", "tree", "masked-q", "masked-q-direct", "fastshap")

# The estimator that answers a game exactly, and therefore the only one whose
# output is truth rather than an estimate of it.
EXACT_ESTIMATOR = "exact"

# A family *is* the coalition game: two attributions are comparable in value
# units exactly when they estimate the same ``v``, and ``v`` is determined by
# the coalition semantics and the value model it is read from — never by the
# estimator, which is a numerical method applied to a fixed ``v``. Family is
# therefore derived from what the explainer reported (:func:`family_key`), not
# declared per method name; a table keyed on method names restates the
# explainer's own answer and can only drift from it.
#
# These labels are cosmetic: they keep the study CSVs readable for the games
# that have a conventional short name. An unlabelled game still gets a stable
# key, so a new game needs no entry here to be handled correctly.
FAMILY_LABELS = {
    ("marginal_background", "base_q_function"): "marginal",
    ("cohort_background", "base_q_function"): "marginal-cohort",
    ("marginal_background", "tree_surrogate"): "tree-surrogate",
    ("surrogate_masked_q", "masked_q_surrogate"): "masked-surrogate",
    ("direct_masked_q", "base_q_function"): "masked-direct",
    ("conditional_occupancy", "base_q_function"): "conditional-occupancy",
}

# A method names a coalition game and the estimator applied to it. The pair is
# what a study varies, and separating them is what lets one study hold the game
# fixed and compare estimators, or hold the estimator fixed and compare games.
GAMES = (
    "marginal",
    "marginal-cohort",
    "tree-surrogate",
    "conditional-occupancy",
    "surrogate-masked-q",
    "direct-masked-q",
)

# Estimators each game can actually be answered with. Not a full cross product:
# TreeSHAP only reads a fitted tree, the cohort game exists only in KernelSHAP
# form, and the conditional-occupancy game is enumerated exactly — sampling it
# would trade away the one property that makes it worth having, that it is the
# on-manifold answer rather than an estimate of one.
GAME_ESTIMATORS = {
    "marginal": ("exact", "kernel", "fastshap"),
    "marginal-cohort": ("kernel",),
    "tree-surrogate": ("tree",),
    "conditional-occupancy": ("exact",),
    "surrogate-masked-q": ("auto", "exact", "permutation", "kernel", "fastshap"),
    "direct-masked-q": ("auto", "exact", "permutation", "kernel", "fastshap"),
}

# Historical flat names, kept so saved commands, docs and studies keep working.
# ``fastshap`` alone carries no game: the artifact records which one it was
# trained against, and asserting a different one is the caller's choice to make
# explicitly with ``<game>:fastshap``.
METHOD_ALIASES: dict[str, tuple[str | None, str]] = {
    "exact": ("marginal", "exact"),
    "kernel": ("marginal", "kernel"),
    "cohort-kernel": ("marginal-cohort", "kernel"),
    "tree": ("tree-surrogate", "tree"),
    "sverl": ("conditional-occupancy", "exact"),
    "masked-q": ("surrogate-masked-q", "auto"),
    "masked-q-direct": ("direct-masked-q", "auto"),
    "fastshap": (None, "fastshap"),
}

# Game <-> the FastSHAP value mode that plays it. Spelled out rather than
# imported from :mod:`xrl.explainability.fastshap`, which pulls in the neural
# stack; every use of that module here is a local import for the same reason.
# ``test_fastshap_modes_match_the_harness`` pins the two against each other.
_FASTSHAP_GAME_MODES = {
    "marginal": "marginal",
    "surrogate-masked-q": "masked-surrogate",
    "direct-masked-q": "direct",
}
_FASTSHAP_MODE_GAMES = {mode: game for game, mode in _FASTSHAP_GAME_MODES.items()}

_MASKED_GAMES = ("surrogate-masked-q", "direct-masked-q")

_COHORT_FAMILY = ("cohort_background", "base_q_function")

# The exact conditional-occupancy game enumerates the reachable learner-decision
# distribution, which only Tic Tac Toe is small enough to traverse.
_CONDITIONAL_GAME = "conditional-occupancy"


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One selected method: a coalition game and the estimator applied to it."""

    name: str
    game: str | None
    estimator: str


def parse_method(text: str) -> MethodSpec:
    """Parse ``game:estimator``, or a historical flat name, into a spec."""

    label = text.strip()
    if ":" not in label:
        alias = METHOD_ALIASES.get(label)
        if alias is None:
            raise ValueError(
                f"unknown method {label!r}; expected one of {sorted(METHOD_ALIASES)} or '<game>:<estimator>' "
                f"with a game from {list(GAMES)}"
            )
        return MethodSpec(label, alias[0], alias[1])
    game, _, estimator = label.partition(":")
    if game not in GAMES:
        raise ValueError(f"unknown game {game!r} in method {label!r}; expected one of {list(GAMES)}")
    allowed = GAME_ESTIMATORS[game]
    if estimator not in allowed:
        raise ValueError(
            f"game {game!r} cannot be answered by estimator {estimator!r}; expected one of {list(allowed)}"
        )
    return MethodSpec(label, game, estimator)


_EVALUATOR_MARGINAL = "marginal"
_EVALUATOR_SURROGATE = "surrogate"
_EVALUATOR_DIRECT = "direct"
_EVALUATOR_CONDITIONAL = "conditional"

# How each evaluator's game stands to the exact conditional-occupancy game, used
# to label coalition-fidelity rows. The masked-Q games are trained to reproduce
# it — under a proper loss the population minimiser of a regression of the
# full-observation value onto the masked observation *is* the conditional mean —
# so their distance from it is model error and smaller is strictly better. The
# marginal game answers a different question by construction, so its distance is
# the semantics gap and has no target value. Calling the second an error would
# repeat, one level down, the mistake ADR 0020's family rule exists to prevent.
_TARGETS_REFERENCE = "targets_reference"
_DISTINCT_GAME = "distinct_game"
_FIDELITY_RELATIONS = {
    _EVALUATOR_SURROGATE: _TARGETS_REFERENCE,
    _EVALUATOR_DIRECT: _TARGETS_REFERENCE,
    _EVALUATOR_MARGINAL: _DISTINCT_GAME,
}


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Budgets and sampling knobs for one comparison study."""

    methods: tuple[str, ...] = COMPARISON_METHODS
    states: int = 64
    # Deliberately distinct from background_seed: probe states must not be the
    # states the marginal methods impute from.
    state_seed: int = 987654
    state_policy: str = "epsilon"
    background_states: int = 256
    background_seed: int = 1234
    background_policy: str = "greedy"
    max_background: int = 64
    kernel_nsamples: int | str = "auto"
    # Budget for the in-repo KernelSHAP estimator over a masked-Q game, which is
    # a different mechanism from ``kernel_nsamples`` (shap's, for the marginal
    # game) and from ``num_permutations`` (the permutation sampler's).
    kernel_coalitions: int = 512
    cohort_min_size: int = 20
    num_permutations: int = 64
    top_k: int = 5
    faithfulness: bool = True
    # Evaluate every k-th deletion/insertion step. Faithfulness dominates the
    # cost of a study on larger boards (each step is a full coalition
    # evaluation over the background), so coarsening the ladder is the first
    # knob to reach for before dropping faithfulness entirely.
    faithfulness_stride: int = 1
    references: bool = True
    # Compare each learned coalition game with the exact conditional one over
    # every coalition, before any Shapley transform. Costs 2^d evaluations per
    # state per game, so it runs only where that enumeration is already
    # affordable and where the exact game exists to compare against.
    coalition_fidelity: bool = True
    bootstrap_resamples: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("at least one method is required")
        for method in self.methods:
            parse_method(method)
        if self.kernel_coalitions < 1:
            raise ValueError(f"kernel_coalitions must be positive, got {self.kernel_coalitions}")
        if self.states < 1:
            raise ValueError(f"states must be positive, got {self.states}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.faithfulness_stride < 1:
            raise ValueError(f"faithfulness_stride must be positive, got {self.faithfulness_stride}")

    def faithfulness_steps(self, feature_count: int) -> tuple[int, ...] | None:
        """Return the deletion/insertion ladder, always including both endpoints."""

        if self.faithfulness_stride <= 1:
            return None
        ladder = list(range(0, feature_count + 1, self.faithfulness_stride))
        if ladder[-1] != feature_count:
            ladder.append(feature_count)
        return tuple(ladder)


def family_key(semantics: ExplanationSemantics) -> tuple[str, str]:
    """Return the coalition game two attributions must share to be comparable.

    The game is ``(coalition_semantics, value_model_source)``. Two attributions
    estimate the same ``v`` — and so may be compared in value units rather than
    only by rank — exactly when these agree.
    """

    return (semantics.coalition_semantics, semantics.value_model_source)


def family_label(semantics: ExplanationSemantics) -> str:
    """Return the display name of a game, falling back to its raw key."""

    key = family_key(semantics)
    return FAMILY_LABELS.get(key, "/".join(key))


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    """One method's attribution for one probe state."""

    state_index: int
    method: str
    action: int
    baseline_value: float
    full_value: float
    attributions: dict[str, float]
    duration_seconds: float
    # What the explainer reported about the question it answered. Carried rather
    # than re-derived: family and reference selection both read it, and a table
    # keyed on the method name would restate it and could contradict it.
    semantics: ExplanationSemantics = field(default_factory=ExplanationSemantics)

    @property
    def efficiency_residual(self) -> float:
        return efficiency_residual(
            self.attributions,
            baseline_value=self.baseline_value,
            full_value=self.full_value,
        )

    @property
    def family(self) -> str:
        return family_label(self.semantics)

    @property
    def is_exact(self) -> bool:
        return self.semantics.shapley_estimator == EXACT_ESTIMATOR


@dataclass
class ComparisonReport:
    """Everything one comparison produced, plus CSV/JSON serialisation."""

    run_dir: Path
    env_id: str
    feature_names: tuple[str, ...]
    config: ComparisonConfig
    outcomes: list[MethodOutcome] = field(default_factory=list)
    agreement_rows: list[dict[str, Any]] = field(default_factory=list)
    reference_rows: list[dict[str, Any]] = field(default_factory=list)
    fidelity_rows: list[dict[str, Any]] = field(default_factory=list)
    faithfulness_rows: list[dict[str, Any]] = field(default_factory=list)
    curve_rows: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(outcome.method for outcome in self.outcomes))

    @property
    def state_count(self) -> int:
        return len({outcome.state_index for outcome in self.outcomes})

    def summary(self) -> dict[str, Any]:
        """Aggregate every per-state statistic with bootstrap intervals."""

        return {
            "run_dir": str(self.run_dir),
            "env_id": self.env_id,
            "feature_count": len(self.feature_names),
            "states": self.state_count,
            "methods": list(self.methods),
            "skipped": dict(self.skipped),
            "failures": self.failures,
            "config": _config_dict(self.config),
            "cost_seconds": self._cost_summary(),
            "agreement": _aggregate(self.agreement_rows, ("method_a", "method_b"), self.config.bootstrap_resamples),
            # Pooled over cells rather than averaged over states, and reported
            # separately because it answers a different question: the slope
            # says whether two methods are on the same scale, which no
            # correlation does. Do not read its r_squared as agreement.
            "pooled_regression": self.pooled_rows(),
            "reference_error": _aggregate(
                self.reference_rows, ("method", "reference"), self.config.bootstrap_resamples
            ),
            # Distance between characteristic functions, not between
            # attributions. Read it before the agreement table: a learned game
            # far from the one it targets makes its family's cross-family
            # disagreement model error rather than a difference in semantics,
            # and no attribution-level statistic can tell those apart.
            "coalition_fidelity": _aggregate(
                self.fidelity_rows, ("game", "reference"), self.config.bootstrap_resamples
            ),
            "faithfulness": _aggregate(
                self.faithfulness_rows, ("method", "evaluator"), self.config.bootstrap_resamples
            ),
        }

    def write(self, out_dir: str | Path) -> list[Path]:
        """Write long-format CSVs plus ``summary.json``; return the paths written.

        Long format keeps the study re-analysable in pandas without paying the
        (dominant) cost of recomputing the explanations.
        """

        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        written = [
            _write_csv(destination / "attributions.csv", self._attribution_rows()),
            _write_csv(destination / "agreement.csv", self.agreement_rows),
            _write_csv(destination / "agreement_pooled.csv", self.pooled_rows()),
            _write_csv(destination / "agreement_per_feature.csv", self.per_feature_rows()),
            _write_csv(destination / "reference_error.csv", self.reference_rows),
            _write_csv(destination / "coalition_fidelity.csv", self.fidelity_rows),
            _write_csv(destination / "faithfulness.csv", self.faithfulness_rows),
            _write_csv(destination / "curves.csv", self.curve_rows),
        ]
        summary_path = destination / "summary.json"
        summary_path.write_text(json.dumps(json_safe(self.summary()), indent=2) + "\n", encoding="utf-8")
        written.append(summary_path)
        return [path for path in written if path is not None]

    def pooled_rows(self) -> list[dict[str, Any]]:
        """One regression per method pair over every (state, feature) cell.

        The third of three readings of "do these methods agree", and the one
        the per-state statistics cannot give: the slope. Two estimators of one
        game should land on ``slope ~ 1``; a systematically shrunk or inflated
        explainer shows up here and nowhere else.
        """

        rows: list[dict[str, Any]] = []
        for first, second, left, right in self._pairs():
            flat_left = [values[name] for values in left for name in self.feature_names]
            flat_right = [values[name] for values in right for name in self.feature_names]
            fit = attribution_regression(flat_left, flat_right)
            rows.append(
                {
                    "method_a": first,
                    "method_b": second,
                    "states": len(left),
                    "slope": fit.slope,
                    "intercept": fit.intercept,
                    "r_squared": fit.r_squared,
                    "n": fit.n,
                }
            )
        return rows

    def per_feature_rows(self) -> list[dict[str, Any]]:
        """One correlation per method pair per feature, across states."""

        rows: list[dict[str, Any]] = []
        for first, second, left, right in self._pairs():
            for score in per_feature_agreement(left, right, self.feature_names):
                rows.append(
                    {
                        "method_a": first,
                        "method_b": second,
                        "feature": score.feature,
                        "pearson": score.pearson,
                        "spearman": score.spearman,
                        "n_effective": score.n_effective,
                        "states": len(left),
                    }
                )
        return rows

    def _pairs(self) -> list[tuple[str, str, list[dict[str, float]], list[dict[str, float]]]]:
        """Method pairs with their attributions on the states both explained."""

        indexed: dict[str, dict[int, MethodOutcome]] = {}
        for outcome in self.outcomes:
            indexed.setdefault(outcome.method, {})[outcome.state_index] = outcome
        pairs = []
        for first, second in _unordered_pairs(sorted(indexed)):
            shared = sorted(set(indexed[first]) & set(indexed[second]))
            if not shared:
                continue
            pairs.append(
                (
                    first,
                    second,
                    [indexed[first][index].attributions for index in shared],
                    [indexed[second][index].attributions for index in shared],
                )
            )
        return pairs

    def _attribution_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "state_index": outcome.state_index,
                "method": outcome.method,
                "family": outcome.family,
                "target_quantity": outcome.semantics.target_quantity,
                "coalition_semantics": outcome.semantics.coalition_semantics,
                "value_model_source": outcome.semantics.value_model_source,
                "shapley_estimator": outcome.semantics.shapley_estimator,
                "action": outcome.action,
                "feature": name,
                "attribution": outcome.attributions[name],
                "baseline_value": outcome.baseline_value,
                "full_value": outcome.full_value,
                "efficiency_residual": outcome.efficiency_residual,
                "duration_seconds": outcome.duration_seconds,
            }
            for outcome in self.outcomes
            for name in self.feature_names
        ]

    def _cost_summary(self) -> dict[str, dict[str, float]]:
        costs: dict[str, list[float]] = {}
        for outcome in self.outcomes:
            costs.setdefault(outcome.method, []).append(outcome.duration_seconds)
        return {
            method: {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "total": float(np.sum(values)),
                "count": len(values),
            }
            for method, values in costs.items()
        }


def run_comparison(
    run_dir: str | Path,
    *,
    config: ComparisonConfig | None = None,
    q_surrogate: str | Path | None = None,
    fastshap_run: str | Path | Sequence[str | Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ComparisonReport:
    """Run every available method over a population of probe states.

    Methods whose optional extra or trained artifact is missing are recorded
    in ``report.skipped`` with a reason rather than aborting the study.
    """

    settings = config or ComparisonConfig()
    announce = progress or (lambda _message: None)
    loaded = load_explanation_run(run_dir)
    feature_names = tuple(loaded.feature_spec.names)
    if q_surrogate is not None:
        from xrl.explainability.distributional_q_explainer import validate_surrogate_base_run

        validate_surrogate_base_run(q_surrogate, loaded.run_dir)
    if fastshap_run is not None:
        from xrl.explainability.fastshap import validate_fastshap_base_run

        # Every supplied artifact, not only the one _fastshap_artifact selects:
        # selection is by the game an artifact records, so an artifact belonging
        # to another run can be chosen without ever being checked.
        for artifact in fastshap_paths(fastshap_run):
            validate_fastshap_base_run(artifact, loaded.run_dir)

    direct_mask_available = (
        bool(loaded.config.get("feature_mask_input", False))
        and float(loaded.config.get("masked_batch_ratio", 0.0)) > 0
        and callable(getattr(loaded.policy, "masked_q_values", None))
    )
    methods, skipped = _resolve_methods(
        settings.methods,
        len(feature_names),
        q_surrogate,
        fastshap_run,
        direct_mask_available=direct_mask_available,
        env_id=loaded.env_id,
    )
    report = ComparisonReport(
        run_dir=Path(run_dir),
        env_id=loaded.env_id,
        feature_names=feature_names,
        config=settings,
        skipped=skipped,
    )
    if not methods:
        return report

    announce(f"collecting imputation background ({settings.background_states} states)")
    background, _ = resolve_background(
        loaded.adapter,
        loaded.policy,
        loaded.config,
        loaded.run_dir,
        states=settings.background_states,
        seed=settings.background_seed,
        policy_name=settings.background_policy,
        save=True,
    )
    announce(f"sampling {settings.states} held-out probe states")
    probes = _probe_states(loaded, settings)
    announce(f"{len(probes)} distinct probe states; running {len(methods)} methods")

    # Built once: traversing the reachable occupancy is setup, and both the
    # method arm and the faithfulness evaluator read the same oracle.
    oracle = _conditional_occupancy_oracle(loaded, methods)
    # Also built once, and shared: the masked-q arm and the faithfulness
    # evaluator previously loaded the same surrogate by two separate routes, the
    # first of them once per probe state.
    surrogate = (
        _load_surrogate(q_surrogate, loaded.run_dir)
        if q_surrogate is not None and _needs_surrogate(methods, settings)
        else None
    )
    explainers = _build_explainers(loaded, background, methods, settings, surrogate, fastshap_run, oracle)

    for state_index, (observation, action_mask) in enumerate(probes):
        action = greedy_action(loaded.policy, observation, action_mask)
        outcomes = _explain_state(explainers, state_index, observation, action_mask, action, report)
        if not outcomes:
            continue
        _score_state(
            report,
            loaded=loaded,
            background=background,
            settings=settings,
            state_index=state_index,
            observation=observation,
            action=action,
            outcomes=outcomes,
            surrogate=surrogate,
            oracle=oracle,
        )
        announce(f"state {state_index + 1}/{len(probes)} scored ({len(outcomes)} methods)")

    return report


def _explain_state(
    explainers: Mapping[str, Callable[[np.ndarray, np.ndarray, int], ExplanationResult]],
    state_index: int,
    observation: np.ndarray,
    action_mask: np.ndarray,
    action: int,
    report: ComparisonReport,
) -> dict[str, MethodOutcome]:
    outcomes: dict[str, MethodOutcome] = {}
    for method, explain in explainers.items():
        started = time.perf_counter()
        try:
            result = explain(observation, action_mask, action)
        except Exception as error:  # noqa: BLE001 - one method failing must not lose the study
            report.failures.append({"state_index": state_index, "method": method, "error": str(error)})
            continue
        outcome = MethodOutcome(
            state_index=state_index,
            method=method,
            action=action,
            baseline_value=float(result.baseline_value),
            full_value=float(result.full_value),
            attributions={name: float(value) for name, value in result.attributions.items()},
            duration_seconds=time.perf_counter() - started,
            semantics=result.semantics,
        )
        outcomes[method] = outcome
        report.outcomes.append(outcome)
    return outcomes


def _score_state(
    report: ComparisonReport,
    *,
    loaded: LoadedExplanationRun,
    background: BackgroundDataset,
    settings: ComparisonConfig,
    state_index: int,
    observation: np.ndarray,
    action: int,
    outcomes: Mapping[str, MethodOutcome],
    surrogate: Any,
    oracle: Any = None,
) -> None:
    names = report.feature_names
    vectors = {method: outcome.attributions for method, outcome in outcomes.items()}

    for first, second in _unordered_pairs(sorted(vectors)):
        scores = compare_attributions(
            [vectors[first][name] for name in names],
            [vectors[second][name] for name in names],
            top_k=settings.top_k,
        )
        report.agreement_rows.append(
            {
                "state_index": state_index,
                "method_a": first,
                "method_b": second,
                "same_family": family_key(outcomes[first].semantics) == family_key(outcomes[second].semantics),
                "spearman": scores.spearman,
                "pearson": scores.pearson,
                "kendall_tau_b": scores.kendall_tau_b,
                "top_k_jaccard": scores.top_k_jaccard,
                **{f"top_{k}_jaccard": value for k, value in sorted(scores.top_k_jaccards.items())},
                "top_1_agreement": scores.top_1_agreement,
                "sign_agreement": scores.sign_agreement,
                "cosine": scores.cosine,
            }
        )

    if settings.references:
        _score_references(report, loaded, background, settings, state_index, observation, action, outcomes)

    if settings.faithfulness:
        evaluators = {
            _EVALUATOR_MARGINAL: marginal_characteristic(
                loaded.policy,
                observation,
                action,
                background,
                max_background=settings.max_background,
                seed=settings.seed,
            )
        }
        if surrogate is not None:
            evaluators[_EVALUATOR_SURROGATE] = _surrogate_characteristic(surrogate, observation, action, len(names))
        # Keyed on the game an outcome reported, not on a method name: several
        # estimators of the direct game may be present under different names,
        # and all of them want scoring under it.
        if any(outcome.semantics.coalition_semantics == "direct_masked_q" for outcome in outcomes.values()):
            evaluators[_EVALUATOR_DIRECT] = _direct_masked_q_characteristic(
                loaded.policy, observation, action, len(names)
            )
        # The on-manifold evaluator. Without it the marginal families are scored
        # only under games that splice unreachable boards, and the masked-Q
        # families only under their own learned approximations — every method in
        # the study would be judged by a game it or a sibling defines. This is
        # the one evaluator whose coalition values are exact.
        if oracle is not None and any(
            outcome.semantics.coalition_semantics == "conditional_occupancy" for outcome in outcomes.values()
        ):
            evaluators[_EVALUATOR_CONDITIONAL] = _conditional_occupancy_characteristic(oracle, observation, action)
        # Before faithfulness, so the enumeration warms the shared coalition
        # caches the deletion ladders then read rather than duplicating them.
        if settings.coalition_fidelity:
            _score_coalition_fidelity(report, evaluators, state_index, len(names))
        for label, characteristic in evaluators.items():
            _score_faithfulness(report, label, characteristic, state_index, outcomes, settings)


def _score_references(
    report: ComparisonReport,
    loaded: LoadedExplanationRun,
    background: BackgroundDataset,
    settings: ComparisonConfig,
    state_index: int,
    observation: np.ndarray,
    action: int,
    outcomes: Mapping[str, MethodOutcome],
) -> None:
    names = report.feature_names
    candidates = _reference_candidates(loaded, background, settings, observation, action, outcomes)
    for method, outcome in outcomes.items():
        # An exact answer is truth, not an approximation of one; scoring it
        # against itself would report a row of zeros.
        if outcome.is_exact:
            continue
        candidate = candidates.get(family_key(outcome.semantics))
        if candidate is None or candidate.name == method:
            continue
        error = reference_error(
            [outcome.attributions[name] for name in names],
            [candidate.attributions[name] for name in names],
        )
        report.reference_rows.append(
            {
                "state_index": state_index,
                "method": method,
                "reference": candidate.name,
                # The reference's own estimator, so a reader never has to infer
                # whether this row measures error against truth.
                "reference_estimator": candidate.estimator,
                "mean_absolute_error": error.mean_absolute_error,
                "max_absolute_error": error.max_absolute_error,
                "root_mean_squared_error": error.root_mean_squared_error,
                "relative_error": error.relative_error,
                "spearman": error.spearman,
            }
        )


@dataclass(frozen=True, slots=True)
class _ReferenceCandidate:
    """An exact answer to one game, usable as ground truth for that game."""

    name: str
    estimator: str
    attributions: dict[str, float]


def _reference_candidates(
    loaded: LoadedExplanationRun,
    background: BackgroundDataset,
    settings: ComparisonConfig,
    observation: np.ndarray,
    action: int,
    outcomes: Mapping[str, MethodOutcome],
) -> dict[tuple[str, str], _ReferenceCandidate]:
    """Map each game to the exact answer available for it, if any.

    Selection is a rule rather than a table: an approximation's reference is the
    outcome that plays *the same game* and answers it *exactly*. That makes the
    Minesweeper case right by construction — above
    :data:`EXACT_SHAPLEY_MAX_FEATURES` no masked-Q outcome is exact, so no
    candidate exists and no reference rows are written, rather than a number
    that quietly means estimator-vs-estimator instead of error-against-truth.
    """

    candidates: dict[tuple[str, str], _ReferenceCandidate] = {}
    for method, outcome in outcomes.items():
        if outcome.is_exact:
            candidates.setdefault(
                family_key(outcome.semantics),
                _ReferenceCandidate(method, outcome.semantics.shapley_estimator, outcome.attributions),
            )
    # The cohort game has no method of its own producing an exact answer, so its
    # candidate is synthesised — but only when an approximation actually needs
    # it, because enumerating 2^d coalitions costs more than the rest combined.
    if _COHORT_FAMILY not in candidates and any(
        family_key(outcome.semantics) == _COHORT_FAMILY and not outcome.is_exact for outcome in outcomes.values()
    ):
        truth = _exact_cohort_attributions(loaded, background, settings, observation, action)
        if truth is not None:
            candidates[_COHORT_FAMILY] = _ReferenceCandidate("exact-cohort", EXACT_ESTIMATOR, truth)
    return candidates


def _score_coalition_fidelity(
    report: ComparisonReport,
    evaluators: Mapping[str, Any],
    state_index: int,
    feature_count: int,
) -> None:
    """Compare every other coalition game with the exact conditional one.

    Answers the question the attribution tables cannot: when two methods
    disagree, is it because their games ask different things, or because a game
    that *targets* another fails to reproduce it? The first is a finding about
    semantics, the second is model error, and at the attribution level both
    appear only as "the vectors differ".

    Needs the exact game to compare against, which exists on Tic Tac Toe alone,
    and enumerates every coalition, so it is bounded by the same feature count
    that bounds exact marginal enumeration. Both conditions are silent no-ops
    rather than skips: nothing was requested by name, so there is nothing to
    report as unfulfilled.
    """

    reference_game = evaluators.get(_EVALUATOR_CONDITIONAL)
    if reference_game is None or feature_count > EXACT_MARGINAL_MAX_FEATURES:
        return

    coalitions = all_coalitions(feature_count)
    reference_values = [reference_game.value(coalition) for coalition in coalitions]
    for label, characteristic in evaluators.items():
        relation = _FIDELITY_RELATIONS.get(label)
        if relation is None:
            continue
        gap = coalition_error([characteristic.value(coalition) for coalition in coalitions], reference_values)
        report.fidelity_rows.append(
            {
                "state_index": state_index,
                "game": label,
                "reference": _EVALUATOR_CONDITIONAL,
                # Whether this row is an error or a semantic distance. Dropped
                # from the aggregate as a non-numeric key, exactly as
                # ``same_family`` is, and read per state from the CSV.
                "relation": relation,
                "coalitions": gap.coalitions,
                "mean_absolute_error": gap.mean_absolute_error,
                "max_absolute_error": gap.max_absolute_error,
                "root_mean_squared_error": gap.root_mean_squared_error,
                "normalized_mae": gap.normalized_mae,
                "reference_spread": gap.reference_spread,
                "empty_endpoint_error": gap.empty_endpoint_error,
                "full_endpoint_error": gap.full_endpoint_error,
            }
        )


def _score_faithfulness(
    report: ComparisonReport,
    evaluator: str,
    characteristic: Any,
    state_index: int,
    outcomes: Mapping[str, MethodOutcome],
    settings: ComparisonConfig,
) -> None:
    """Score every method for one state under one evaluator.

    The random control depends only on the state, so it is computed once here
    and shared; each method's own curves are computed once and reduced to the
    scalar summary rather than recomputed for it.
    """

    names = report.feature_names
    steps = settings.faithfulness_steps(len(names))
    control = control_curves(characteristic, steps=steps, random_seed=settings.seed)
    for method, outcome in outcomes.items():
        curves = perturbation_curves(characteristic, outcome.attributions, feature_names=names, steps=steps)
        report.faithfulness_rows.append(
            {
                "state_index": state_index,
                "method": method,
                "evaluator": evaluator,
                **summarize_curves(curves, control),
            }
        )
        for mode in (DELETION, INSERTION):
            curve = curves[mode]
            for step, value in zip(curve.steps, curve.values, strict=True):
                report.curve_rows.append(
                    {
                        "state_index": state_index,
                        "method": method,
                        "evaluator": evaluator,
                        "mode": mode,
                        "step": step,
                        "value": value,
                        "reference_span": curve.reference_span,
                    }
                )


def _resolve_methods(
    requested: Sequence[str],
    feature_count: int,
    q_surrogate: str | Path | None,
    fastshap_run: str | Path | Sequence[str | Path] | None,
    *,
    direct_mask_available: bool = False,
    env_id: str = TICTACTOE_ENV_ID,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Split requested methods into runnable ones and skipped ones with reasons."""

    has_explain = explain_dependencies_available()
    has_neural = neural_dependencies_available()
    artifacts = fastshap_paths(fastshap_run)
    available: list[str] = []
    skipped: dict[str, str] = {}
    for method in requested:
        spec = parse_method(method)
        reason = _skip_reason(
            spec,
            feature_count,
            q_surrogate,
            artifacts,
            direct_mask_available=direct_mask_available,
            env_id=env_id,
            has_explain=has_explain,
            has_neural=has_neural,
        )
        if reason is None:
            available.append(method)
        else:
            skipped[method] = reason
    return tuple(available), skipped


def _skip_reason(
    spec: MethodSpec,
    feature_count: int,
    q_surrogate: str | Path | None,
    artifacts: Sequence[Path],
    *,
    direct_mask_available: bool,
    env_id: str,
    has_explain: bool,
    has_neural: bool,
) -> str | None:
    """Return why a method cannot run in this study, or ``None`` if it can."""

    # shap backs the marginal kernel and tree paths; the in-repo KernelSHAP over
    # a masked-Q game needs nothing beyond NumPy.
    needs_shap = spec.estimator in {"kernel", "tree"} and spec.game in {"marginal", "marginal-cohort", "tree-surrogate"}
    if needs_shap and not has_explain:
        return "requires the 'explain' extra (shap, scikit-learn)"
    if spec.game == _CONDITIONAL_GAME and env_id != TICTACTOE_ENV_ID:
        return (
            "exact conditional occupancy is tictactoe-only: enumerating the reachable "
            f"learner-decision distribution is intractable for {env_id}"
        )
    if spec.game == "marginal" and spec.estimator == "exact" and feature_count > EXACT_MARGINAL_MAX_FEATURES:
        return f"exact enumeration needs at most {EXACT_MARGINAL_MAX_FEATURES} features, run has {feature_count}"
    if (spec.game in _MASKED_GAMES or spec.estimator == "fastshap") and not has_neural:
        return "requires the 'neural' extra (jax, flax, optax)"
    if spec.game == "surrogate-masked-q" and spec.estimator != "fastshap" and q_surrogate is None:
        return "requires a masked-Q surrogate (xrl train-q-surrogate)"
    if spec.game == "direct-masked-q" and not direct_mask_available:
        return "requires feature_mask_input=true and masked_batch_ratio>0 on the base run"
    if spec.estimator == "fastshap":
        return _fastshap_skip_reason(spec, artifacts)
    return None


def fastshap_paths(fastshap_run: str | Path | Sequence[str | Path] | None) -> tuple[Path, ...]:
    """Normalise the artifact argument to a tuple of directories.

    Accepts one path, a comma-separated string, or a sequence, so a saved
    single-artifact command keeps its meaning while a study can carry one
    amortized arm per game.
    """

    if fastshap_run is None:
        return ()
    if isinstance(fastshap_run, (str, Path)):
        text = str(fastshap_run)
        return tuple(Path(part.strip()) for part in text.split(",") if part.strip())
    return tuple(Path(str(entry)) for entry in fastshap_run)


def _fastshap_artifact(spec: MethodSpec, paths: Sequence[Path]) -> tuple[Path | None, str | None]:
    """Select the artifact playing the spec's game, or say why none does.

    A FastSHAP network estimates the Shapley values of whatever ``v`` it was
    trained against, so the artifact's *recorded* mode decides which game it can
    answer. Selecting by that rather than by argument order is what lets a
    surrogate-trained and a direct-trained explainer sit in one study without
    either being scored against the wrong game.
    """

    if not paths:
        return None, "requires a FastSHAP run (xrl train-fastshap)"
    from xrl.explainability.fastshap import fastshap_value_mode

    modes: list[str] = []
    for path in paths:
        try:
            modes.append(fastshap_value_mode(path))
        except ValueError as error:
            return None, str(error)

    if spec.game is None:
        # The bare alias adopts whatever game the artifact records, which is
        # only well defined when there is one artifact to adopt from.
        if len(paths) == 1:
            return paths[0], None
        games = sorted(_FASTSHAP_MODE_GAMES.get(mode, mode) for mode in modes)
        return None, f"'fastshap' is ambiguous across {len(paths)} artifacts playing {games}; name the game explicitly"

    expected = _FASTSHAP_GAME_MODES[spec.game]
    matches = [path for path, mode in zip(paths, modes, strict=True) if mode == expected]
    if not matches:
        actual = sorted(_FASTSHAP_MODE_GAMES.get(mode, mode) for mode in modes)
        return None, f"no FastSHAP artifact plays the {spec.game!r} game; the supplied ones play {actual}"
    if len(matches) > 1:
        listed = [str(path) for path in matches]
        return None, f"{len(matches)} FastSHAP artifacts play the {spec.game!r} game: {listed}"
    return matches[0], None


def _fastshap_skip_reason(spec: MethodSpec, paths: Sequence[Path]) -> str | None:
    """Check some supplied FastSHAP artifact plays the game the method claims."""

    return _fastshap_artifact(spec, paths)[1]


def _conditional_occupancy_oracle(loaded: LoadedExplanationRun, methods: Sequence[str]) -> Any:
    """Build the exact occupancy oracle once, when a method actually plays it.

    Traversing the reachable learner-decision distribution costs about a second
    and is pure setup, so it happens here rather than per state. It is skipped
    entirely when nothing in the study plays the game, which keeps every
    existing study's cost unchanged.

    The opponent law comes from the run rather than a default: the occupancy a
    policy faces — and therefore the game being played — depends on which
    opponent generated it.
    """

    if not any(parse_method(method).game == _CONDITIONAL_GAME for method in methods):
        return None
    from xrl.explainability.tictactoe import TicTacToeSVERLExplainer, run_opponent_law

    opponent, randomness = run_opponent_law(loaded.config)
    return TicTacToeSVERLExplainer(
        loaded.policy,
        opponent=opponent,
        perturbed_minimax_probability=randomness,
    )


def _conditional_occupancy_explainer(oracle: Any) -> Callable[[np.ndarray, np.ndarray, int], ExplanationResult]:
    """Explain one fixed-action Q-value under exact conditional occupancy."""

    if oracle is None:
        raise ValueError("the conditional-occupancy game needs an occupancy oracle; none was built")
    from xrl.explainability.tictactoe import state_from_observation

    def explain(observation: np.ndarray, mask: np.ndarray, action: int) -> ExplanationResult:
        del mask  # legality is re-derived from the position the observation encodes
        return oracle.explain_q_value(state_from_observation(observation), action=action)

    return explain


def _conditional_occupancy_characteristic(oracle: Any, observation: np.ndarray, action: int) -> Any:
    from xrl.explainability.tictactoe import state_from_observation

    return oracle.q_value_characteristic(state_from_observation(observation), action=action)


def _build_explainers(
    loaded: LoadedExplanationRun,
    background: BackgroundDataset,
    methods: Sequence[str],
    settings: ComparisonConfig,
    surrogate: Any,
    fastshap_run: str | Path | Sequence[str | Path] | None,
    oracle: Any = None,
) -> dict[str, Callable[[np.ndarray, np.ndarray, int], ExplanationResult]]:
    """Construct each explainer once so setup cost is not charged per state."""

    shap_config = ShapBaselineConfig(
        max_background=settings.max_background,
        kernel_nsamples=settings.kernel_nsamples,
        cohort_min_size=settings.cohort_min_size,
        seed=settings.background_seed,
    )
    artifacts = fastshap_paths(fastshap_run)
    explainers: dict[str, Callable[[np.ndarray, np.ndarray, int], ExplanationResult]] = {}
    for method in methods:
        spec = parse_method(method)
        if spec.estimator == "fastshap":
            explainers[method] = _fastshap_explainer(spec, artifacts, loaded.run_dir)
        elif spec.game == "tree-surrogate":
            explainers[method] = _tree_explainer(loaded, background, shap_config)
        elif spec.game == "marginal-cohort":
            explainers[method] = _marginal_explainer(loaded, background, "cohort-kernel", shap_config)
        elif spec.game == "marginal":
            explainers[method] = _marginal_explainer(loaded, background, spec.estimator, shap_config)
        elif spec.game == _CONDITIONAL_GAME:
            explainers[method] = _conditional_occupancy_explainer(oracle)
        elif spec.game == "surrogate-masked-q":
            explainers[method] = _masked_q_explainer(loaded, surrogate, settings, spec.estimator)
        elif spec.game == "direct-masked-q":
            explainers[method] = _direct_masked_q_explainer(loaded, settings, spec.estimator)
    return explainers


def _marginal_explainer(
    loaded: LoadedExplanationRun,
    background: BackgroundDataset,
    method: str,
    shap_config: ShapBaselineConfig,
) -> Callable[[np.ndarray, np.ndarray, int], ExplanationResult]:
    explainer = MarginalShapExplainer(
        loaded.policy,
        loaded.feature_spec,
        background,
        method=method,
        cohort_key=loaded.adapter.cohort_key,
        config=shap_config,
    )
    return lambda observation, mask, action: explainer.explain_action(observation, mask, action=action)


def _tree_explainer(
    loaded: LoadedExplanationRun,
    background: BackgroundDataset,
    shap_config: ShapBaselineConfig,
) -> Callable[[np.ndarray, np.ndarray, int], ExplanationResult]:
    explainer = TreeSurrogateShapExplainer(loaded.policy, loaded.feature_spec, background, config=shap_config)
    return lambda observation, mask, action: explainer.explain_action(observation, mask, action=action)


def _masked_q_explainer(
    loaded: LoadedExplanationRun,
    surrogate: Any,
    settings: ComparisonConfig,
    estimator: str = "auto",
) -> Callable[[np.ndarray, np.ndarray, int], ExplanationResult]:
    from xrl.explainability.distributional_q_explainer import surrogate_masked_q_explanation

    if surrogate is None:
        raise ValueError("cannot build 'surrogate-masked-q': requires a masked-Q surrogate run")
    names = tuple(loaded.feature_spec.names)

    def explain(observation: np.ndarray, mask: np.ndarray, action: int) -> ExplanationResult:
        return surrogate_masked_q_explanation(
            surrogate,
            observation,
            action,
            action_mask=mask,
            feature_names=names,
            num_permutations=settings.num_permutations,
            seed=settings.seed,
            estimator=estimator,
            kernel_coalitions=settings.kernel_coalitions,
        )

    return explain


def _fastshap_explainer(
    spec: MethodSpec,
    paths: Sequence[Path],
    base_run_dir: Path | None = None,
) -> Callable[[np.ndarray, np.ndarray, int], ExplanationResult]:
    from xrl.explainability.fastshap import load_fastshap_explainer

    artifact, reason = _fastshap_artifact(spec, paths)
    if artifact is None:
        raise ValueError(f"cannot build {spec.name!r}: {reason}")
    explainer = load_fastshap_explainer(artifact, base_run_dir=base_run_dir)
    return lambda observation, mask, action: explainer.explain_action(observation, action=action, action_mask=mask)


def _exact_cohort_attributions(
    loaded: LoadedExplanationRun,
    background: BackgroundDataset,
    settings: ComparisonConfig,
    observation: np.ndarray,
    action: int,
) -> dict[str, float] | None:
    """Exact Shapley over the cohort background — ground truth for cohort-kernel.

    ``cohort-kernel`` plays a different game than ``kernel`` (a narrowed
    reference distribution), so ``exact`` over the full background is *not*
    its ground truth; the exact answer over its own cohort is.

    Returns ``None`` unless the caller also requested ``exact``. Enumerating
    2^d coalitions costs more than every other method combined, and deriving
    that cost from ``--references`` alone would make a study of cheap methods
    silently unbounded — opting into ``exact`` is the opt-in.
    """

    if "exact" not in settings.methods:
        return None
    if len(loaded.feature_spec.names) > EXACT_MARGINAL_MAX_FEATURES:
        return None
    cohort = cohort_background(
        background,
        observation,
        loaded.adapter.cohort_key,
        min_size=settings.cohort_min_size,
    )
    explainer = MarginalShapExplainer(
        loaded.policy,
        loaded.feature_spec,
        cohort,
        method="exact",
        config=ShapBaselineConfig(
            max_background=settings.max_background,
            cohort_min_size=settings.cohort_min_size,
            seed=settings.background_seed,
        ),
    )
    mask: np.ndarray = np.ones(len(loaded.feature_spec.names), dtype=np.int8)
    result = explainer.explain_action(observation, mask, action=action)
    return {name: float(value) for name, value in result.attributions.items()}


def _direct_masked_q_explainer(
    loaded: LoadedExplanationRun,
    settings: ComparisonConfig,
    estimator: str = "auto",
) -> Callable[[np.ndarray, np.ndarray, int], ExplanationResult]:
    """Explain by querying the jointly trained network's own masked-Q API.

    No distilled surrogate stands between the explanation and the deployed
    policy, which is the whole point of the direct arm.
    """

    from xrl.explainability.distributional_q_explainer import direct_q_explanation

    names = tuple(loaded.feature_spec.names)

    def explain(observation: np.ndarray, mask: np.ndarray, action: int) -> ExplanationResult:
        return direct_q_explanation(
            loaded.policy,
            observation,
            action,
            action_mask=mask,
            feature_names=names,
            num_permutations=settings.num_permutations,
            seed=settings.seed,
            estimator=estimator,
            kernel_coalitions=settings.kernel_coalitions,
        )

    return explain


def _direct_masked_q_characteristic(
    policy: Any,
    observation: np.ndarray,
    action: int,
    feature_count: int,
) -> CachedCharacteristicFunction:
    """Wrap a jointly trained network's masked-Q API as a characteristic function.

    The direct arm's own game, so its faithfulness is not judged only under the
    marginal one — which the marginal family optimises and it does not.
    """

    states = np.asarray(observation, dtype=np.float32).reshape(-1)

    def evaluate(coalition) -> float:
        known: np.ndarray = np.zeros(feature_count, dtype=np.int8)
        indices = [int(index) for index in coalition]
        if indices:
            known[indices] = 1
        return float(np.asarray(policy.masked_q_values(states, known), dtype=np.float64)[action])

    return CachedCharacteristicFunction(feature_count, evaluate)


def _surrogate_characteristic(
    surrogate: Any,
    observation: np.ndarray,
    action: int,
    feature_count: int,
) -> CachedCharacteristicFunction:
    """Wrap a masked distributional Q surrogate as a characteristic function.

    Gives faithfulness a second evaluator, so a method that only wins under
    the marginal game — the one the marginal family optimises — is visible as
    such rather than crowned.
    """

    from xrl.explainability.fastshap import surrogate_masked_value_function

    value_fn = surrogate_masked_value_function(surrogate)
    states = np.asarray(observation, dtype=np.float32).reshape(1, -1)

    def evaluate(coalition) -> float:
        known: np.ndarray = np.zeros((1, feature_count), dtype=np.float32)
        indices = [int(index) for index in coalition]
        if indices:
            known[0, indices] = 1.0
        return float(value_fn(states, known)[0, action])

    return CachedCharacteristicFunction(feature_count, evaluate)


def _load_surrogate(q_surrogate: str | Path, base_run_dir: str | Path) -> Any:
    from xrl.explainability.distributional_q_explainer import load_validated_surrogate

    return load_validated_surrogate(q_surrogate, base_run_dir)


def _needs_surrogate(methods: Sequence[str], settings: ComparisonConfig) -> bool:
    """Whether any part of this study will consult the masked-Q surrogate."""

    if settings.faithfulness:
        return True
    return any(parse_method(method).game == "surrogate-masked-q" for method in methods)


def _probe_states(
    loaded: LoadedExplanationRun,
    settings: ComparisonConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample distinct evaluation states from a seed stream of their own.

    Duplicates are dropped here (unlike in the imputation background, where
    revisits carry occupancy weighting) — re-explaining the same board wastes
    the study's dominant cost and over-weights common positions.
    """

    dataset, _ = resolve_background(
        loaded.adapter,
        loaded.policy,
        loaded.config,
        loaded.run_dir,
        states=max(settings.states * 4, settings.states),
        seed=settings.state_seed,
        policy_name=settings.state_policy,
        save=False,
    )
    seen: set[bytes] = set()
    probes: list[tuple[np.ndarray, np.ndarray]] = []
    for observation, mask in zip(dataset.observations, dataset.action_masks, strict=True):
        if len(probes) >= settings.states:
            break
        row = np.asarray(observation, dtype=np.float32).reshape(-1)
        key = row.tobytes()
        if key in seen or not np.any(np.asarray(mask, dtype=np.int8)):
            continue
        seen.add(key)
        probes.append((row, np.asarray(mask, dtype=np.int8)))
    return probes


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    resamples: int,
) -> list[dict[str, Any]]:
    """Bootstrap the mean of every numeric column, grouped by ``group_keys``."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items(), key=lambda item: [str(part) for part in item[0]]):
        entry: dict[str, Any] = dict(zip(group_keys, group, strict=True))
        entry["states"] = len(members)
        for column in _numeric_columns(members[0]):
            mean, low, high = bootstrap_mean_ci(
                [_as_float(member[column]) for member in members],
                resamples=resamples,
            )
            entry[column] = {"mean": mean, "ci_low": low, "ci_high": high}
        aggregated.append(entry)
    return aggregated


def _numeric_columns(row: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key, value in row.items()
        if key not in {"state_index", "step"} and isinstance(value, int | float) and not isinstance(value, bool)
    ]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _unordered_pairs(items: Sequence[str]) -> list[tuple[str, str]]:
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


def _config_dict(config: ComparisonConfig) -> dict[str, Any]:
    return {
        "methods": list(config.methods),
        "states": config.states,
        "state_seed": config.state_seed,
        "state_policy": config.state_policy,
        "background_states": config.background_states,
        "background_seed": config.background_seed,
        "background_policy": config.background_policy,
        "max_background": config.max_background,
        "kernel_nsamples": config.kernel_nsamples,
        "kernel_coalitions": config.kernel_coalitions,
        "cohort_min_size": config.cohort_min_size,
        "num_permutations": config.num_permutations,
        "top_k": config.top_k,
        "faithfulness_stride": config.faithfulness_stride,
        "faithfulness": config.faithfulness,
        "references": config.references,
        "seed": config.seed,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    if not rows:
        return None
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
