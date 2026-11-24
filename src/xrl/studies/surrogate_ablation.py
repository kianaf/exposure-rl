"""Ablate surrogate construction against a frozen mask-ratio agent.

The completed mask-ratio matrix showed the post-hoc surrogate is the more
faithful explainer, yet still leaves roughly half the exact game's spread as
error. This study holds every trained agent fixed and varies only how the
surrogate is built, so any change in fidelity is attributable to construction
rather than to the policy, the reference law, or the exact game.

``training_points`` is deliberately not an axis: the surrogate dataset is the
complete enumerated support of the reference law, not a sampling budget. The
support arm varies it the only way it can vary -- by changing the reference.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Annotated, Any

import typer

from xrl.studies.analysis import (
    StudyAnalysisError,
    aggregate_within_seed,
    plot_metric_curves,
    read_table,
    write_table,
)
from xrl.studies.cell_worker import SURROGATE_ABLATION_HANDLER, SubprocessCellWork
from xrl.studies.core import (
    StudyCellContext,
    StudyCondition,
    StudyManifestMismatchError,
    StudyProfile,
    StudyResult,
    StudyRunner,
    StudySpec,
    digest_manifest_payload,
    load_study_spec,
)
from xrl.studies.fidelity import ControlledQPosthocResult, run_controlled_q_posthoc
from xrl.training.neural_q import NeuralQConfig, NeuralQTrainer
from xrl.training.value_heads import normalize_value_head

SURROGATE_ABLATION_SCHEMA_VERSION = 1
ABLATION_MANIFEST_FILENAME = "surrogate_ablation_manifest.json"

CONSTRUCTION_ARM = "construction"
SUPPORT_ARM = "support"

# Every dose-response comparison is read against the agent-independent shared
# law, so that the two curves are scored on one fixed estimand.
_DOSE_RESPONSE_REFERENCE = "shared"

_ABLATION_METRICS = (
    "attribution_mae",
    "attribution_rmse",
    "relative_attribution_error",
    "attribution_spearman",
    "attribution_sign_agreement",
    "attribution_top_3_overlap",
    "coalition_mae",
    "coalition_normalized_mae",
    "coalition_reference_spread",
)


@dataclass(frozen=True, slots=True)
class SurrogateAblationProfile:
    """Immutable budget for one surrogate-construction sweep."""

    name: str
    base_condition: str
    base_value_head: str
    base_masked_batch_ratio: float
    seeds: tuple[int, ...]
    hidden_sizes: tuple[tuple[int, ...], ...]
    updates: tuple[int, ...]
    mask_mixtures: tuple[tuple[float, float], ...]
    support_softening: tuple[float, ...]
    shared_explanation_states: int
    policy_explanation_states: int
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be distinct, got {self.seeds}")
        if not self.hidden_sizes or not self.updates or not self.mask_mixtures:
            raise ValueError("hidden_sizes, updates, and mask_mixtures must be non-empty")
        if any(value < 1 for value in self.updates):
            raise ValueError("updates must be positive")
        for shapley, uniform in self.mask_mixtures:
            if not 0.0 <= shapley <= 1.0 or not 0.0 <= uniform <= 1.0:
                raise ValueError(f"mask mixture probabilities must lie in [0, 1], got {(shapley, uniform)}")
            if abs(shapley + uniform - 1.0) > 1e-9:
                raise ValueError(f"mask mixture must sum to one, got {(shapley, uniform)}")
        for value in self.support_softening:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"support softening must lie in [0, 1], got {value}")
        # The post-hoc comparison splits the policy sample evenly across the two
        # roles and rejects an odd total -- but only after the first cell has
        # trained both surrogates, so the check has to happen here instead.
        for field_name, count in (
            ("shared_explanation_states", self.shared_explanation_states),
            ("policy_explanation_states", self.policy_explanation_states),
        ):
            if count < 2 or count % 2:
                raise ValueError(f"{field_name} must be a positive even integer, got {count}")

    @property
    def baseline_hidden(self) -> tuple[int, ...]:
        return self.hidden_sizes[0]

    @property
    def baseline_updates(self) -> int:
        return self.updates[0]

    @property
    def baseline_mixture(self) -> tuple[float, float]:
        return self.mask_mixtures[0]

    def study_profile(self) -> StudyProfile:
        """Resolve the knobs a condition does not override into the manifest."""

        return StudyProfile(
            self.name,
            {
                "base_condition": self.base_condition,
                "surrogate_updates": self.baseline_updates,
                "surrogate_batch_size": 64,
                "surrogate_hidden_sizes": list(self.baseline_hidden),
                "surrogate_full_mask_example_interval": 512,
                "surrogate_empty_mask_example_interval": 512,
                "mask_shapley_probability": self.baseline_mixture[0],
                "mask_uniform_probability": self.baseline_mixture[1],
                "shared_explanation_states": self.shared_explanation_states,
                "policy_explanation_states": self.policy_explanation_states,
                "explanation_seeds": list(self.seeds),
                "bootstrap_resamples": self.bootstrap_resamples,
            },
        )


FAST_ABLATION_PROFILE = SurrogateAblationProfile(
    name="fast",
    base_condition="primary-two-hot-r1",
    base_value_head="two-hot",
    base_masked_batch_ratio=1.0,
    seeds=(0, 1),
    hidden_sizes=((64, 64), (128, 128)),
    updates=(500, 1_000),
    mask_mixtures=((0.7, 0.3), (1.0, 0.0)),
    support_softening=(0.0, 0.5),
    shared_explanation_states=32,
    policy_explanation_states=128,
    bootstrap_resamples=500,
)

RESEARCH_ABLATION_PROFILE = SurrogateAblationProfile(
    name="research",
    base_condition="primary-two-hot-r1",
    base_value_head="two-hot",
    base_masked_batch_ratio=1.0,
    seeds=(0, 1, 2, 3, 4),
    hidden_sizes=((64, 64), (128, 128), (256, 256)),
    updates=(4_000, 16_000, 32_000),
    mask_mixtures=((0.7, 0.3), (1.0, 0.0), (0.0, 1.0)),
    support_softening=(0.0, 0.25, 0.5, 1.0),
    shared_explanation_states=32,
    policy_explanation_states=128,
    bootstrap_resamples=10_000,
)

SURROGATE_ABLATION_PROFILES: dict[str, SurrogateAblationProfile] = {
    FAST_ABLATION_PROFILE.name: FAST_ABLATION_PROFILE,
    RESEARCH_ABLATION_PROFILE.name: RESEARCH_ABLATION_PROFILE,
}


def surrogate_ablation_profile(name: str | SurrogateAblationProfile) -> SurrogateAblationProfile:
    if isinstance(name, SurrogateAblationProfile):
        return name
    try:
        return SURROGATE_ABLATION_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"profile must be one of {tuple(SURROGATE_ABLATION_PROFILES)}, got {name!r}") from exc


def _hidden_key(hidden: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in hidden)


def _mixture_key(mixture: Sequence[float]) -> str:
    return f"{float(mixture[0]):g}".replace(".", "p")


def _base_parameters(profile: SurrogateAblationProfile) -> dict[str, Any]:
    # ``run_controlled_q_posthoc`` gates on the primary arm and reads the head
    # and ratio for its row fields and its ratio-zero OOD override. The agent is
    # frozen, so these describe the loaded cell rather than any new training.
    return {
        "arm": "primary",
        "value_head": profile.base_value_head,
        "feature_mask_input": True,
        "masked_batch_ratio": profile.base_masked_batch_ratio,
    }


def construction_conditions(profile: SurrogateAblationProfile) -> tuple[StudyCondition, ...]:
    """Cross capacity with the update budget, then vary the mixture alone."""

    base = _base_parameters(profile)
    conditions: list[StudyCondition] = []
    for hidden in profile.hidden_sizes:
        for updates in profile.updates:
            conditions.append(
                StudyCondition(
                    f"cap{_hidden_key(hidden)}-upd{updates}",
                    {
                        **base,
                        "ablation_axis": "capacity_updates",
                        "surrogate_hidden_sizes": list(hidden),
                        "surrogate_updates": int(updates),
                        "mask_shapley_probability": profile.baseline_mixture[0],
                        "mask_uniform_probability": profile.baseline_mixture[1],
                    },
                )
            )
    for mixture in profile.mask_mixtures[1:]:
        conditions.append(
            StudyCondition(
                f"mix{_mixture_key(mixture)}",
                {
                    **base,
                    "ablation_axis": "mask_mixture",
                    "surrogate_hidden_sizes": list(profile.baseline_hidden),
                    "surrogate_updates": int(profile.baseline_updates),
                    "mask_shapley_probability": float(mixture[0]),
                    "mask_uniform_probability": float(mixture[1]),
                },
            )
        )
    return tuple(conditions)


def support_conditions(profile: SurrogateAblationProfile) -> tuple[StudyCondition, ...]:
    """Vary the reference law's support, which changes the estimand."""

    base = _base_parameters(profile)
    return tuple(
        StudyCondition(
            f"support-eps{_mixture_key((softening, 1.0 - softening))}",
            {
                **base,
                "ablation_axis": "reference_support",
                "policy_reference_softening": float(softening),
                "surrogate_hidden_sizes": list(profile.baseline_hidden),
                "surrogate_updates": int(profile.baseline_updates),
                "mask_shapley_probability": profile.baseline_mixture[0],
                "mask_uniform_probability": profile.baseline_mixture[1],
            },
        )
        for softening in profile.support_softening
    )


@dataclass(frozen=True, slots=True)
class SurrogateAblationPlan:
    """Construction sweep plus the separately reported reference-support arm."""

    root: Path
    base_matrix: Path
    profile: SurrogateAblationProfile
    construction: StudySpec
    support: StudySpec

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SURROGATE_ABLATION_SCHEMA_VERSION,
            "name": "surrogate-fidelity-ablation",
            "base_matrix": str(self.base_matrix),
            "profile": json.loads(json.dumps(asdict(self.profile), allow_nan=False)),
            "construction": self.construction.manifest_document(),
            "support": self.support.manifest_document(),
        }


def build_surrogate_ablation_plan(
    root: Path,
    base_matrix: Path,
    *,
    profile: str | SurrogateAblationProfile = "fast",
) -> SurrogateAblationPlan:
    resolved = surrogate_ablation_profile(profile)
    study_profile = resolved.study_profile()
    root = Path(root)
    # Resolved, not stored as given: the manifest is the only channel that tells
    # a worker which matrix to freeze, and a relative string there is re-resolved
    # against whatever directory the worker happens to run in.
    return SurrogateAblationPlan(
        root=root,
        base_matrix=Path(base_matrix).resolve(),
        profile=resolved,
        construction=StudySpec(
            name="surrogate-ablation-construction",
            root=root / CONSTRUCTION_ARM,
            profile=study_profile,
            conditions=construction_conditions(resolved),
            seeds=resolved.seeds,
        ),
        support=StudySpec(
            name="surrogate-ablation-support",
            root=root / SUPPORT_ARM,
            profile=study_profile,
            conditions=support_conditions(resolved),
            seeds=resolved.seeds,
        ),
    )


@cache
def _validated_base_cell(base_matrix: Path, condition_key: str, seed: int) -> Path:
    """Content-address one published mask-ratio cell before anything reads it.

    ``StudyRunner`` holds its own cells to a completion record, an identity
    against the study manifest, and an artifact-hash index. A borrowed cell has
    to clear the same bar: the ablation's whole claim is that the agent is the
    one the base matrix declares, and a cell modified or copied without its
    record would otherwise be accepted in silence. Cached because every cell in
    the sweep reloads the same handful of base cells.
    """

    primary = Path(base_matrix) / "primary"
    try:
        spec = load_study_spec(primary)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StudyManifestMismatchError(f"no published mask-ratio study at {primary}: {exc}") from exc
    return StudyRunner(spec).load_published_cell(condition_key, seed).path


def load_frozen_agent(
    base_matrix: Path,
    condition_key: str,
    seed: int,
    *,
    base_value_head: str,
    base_masked_batch_ratio: float,
) -> Any:
    """Load one validated mask-ratio agent read-only.

    The declared head and ratio are checked against the cell's own config rather
    than trusted: they label every row this cell produces, so a profile pointing
    at a condition that disagrees with them would mislabel the results instead
    of failing.
    """

    cell = _validated_base_cell(Path(base_matrix), condition_key, seed)
    config = NeuralQConfig.from_json_dict(json.loads((cell / "config.json").read_text(encoding="utf-8")))
    declared_head = normalize_value_head(base_value_head)
    if config.value_head != declared_head:
        raise ValueError(
            f"frozen cell {cell} trained value head {config.value_head!r}, profile declares {declared_head!r}"
        )
    if config.masked_batch_ratio != float(base_masked_batch_ratio):
        raise ValueError(
            f"frozen cell {cell} trained masked_batch_ratio {config.masked_batch_ratio}, "
            f"profile declares {float(base_masked_batch_ratio)}"
        )
    return NeuralQTrainer(config).load_agent(cell)


def ablate_surrogate_cell(
    context: StudyCellContext,
    *,
    base_matrix: Path,
    base_condition: str,
    base_value_head: str,
    base_masked_batch_ratio: float,
) -> Mapping[str, Any]:
    """Score one surrogate construction against a frozen agent's exact game."""

    agent = load_frozen_agent(
        base_matrix,
        base_condition,
        context.seed,
        base_value_head=base_value_head,
        base_masked_batch_ratio=base_masked_batch_ratio,
    )
    posthoc: ControlledQPosthocResult = run_controlled_q_posthoc(agent, context)
    if not posthoc.executed:
        raise ValueError(f"ablation cell {context.cell_id} did not execute its post-hoc comparison")
    return {
        "base_condition": base_condition,
        "ablation_axis": context.condition.parameters["ablation_axis"],
        "posthoc": posthoc.to_json_dict(),
    }


@dataclass(frozen=True, slots=True)
class SurrogateAblationResult:
    root: Path
    manifest_fingerprint: str
    construction: StudyResult
    support: StudyResult
    tables: tuple[Path, ...] = ()
    plots: tuple[Path, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_fingerprint": self.manifest_fingerprint,
            "construction_cells": len(self.construction.cells),
            "support_cells": len(self.support.cells),
            "reused_cells": self.construction.reused_cells + self.support.reused_cells,
            "tables": [str(path) for path in self.tables],
            "plots": [str(path) for path in self.plots],
        }


def _axis_columns(plan: SurrogateAblationPlan) -> dict[str, dict[str, Any]]:
    """Map each condition key to the axis values its rows should carry.

    Fidelity rows record the condition key but not what the condition varied,
    so the report joins the ablated knobs back on before plotting.
    """

    columns: dict[str, dict[str, Any]] = {}
    for spec in (plan.construction, plan.support):
        for condition in spec.conditions:
            parameters = condition.parameters
            columns[condition.key] = {
                "ablation_axis": parameters["ablation_axis"],
                "surrogate_hidden": _hidden_key(parameters["surrogate_hidden_sizes"]),
                "surrogate_updates": int(parameters["surrogate_updates"]),
                "mask_shapley_probability": float(parameters["mask_shapley_probability"]),
                "policy_reference_softening": float(parameters.get("policy_reference_softening", 0.0)),
            }
    return columns


def _final_masked_budgets(base_matrix: Path) -> dict[tuple[str, str], float]:
    """Masked donor rows each base-matrix cell trained on, at its final episode.

    The mask ratio is a training knob, so its budget is recorded in the policy
    table rather than the fidelity table; the dose-response panel joins the two
    on ``(condition_key, seed)``.
    """

    latest: dict[tuple[str, str], tuple[int, float]] = {}
    for row in read_table(base_matrix / "tables" / "policy_seed.parquet"):
        key = (str(row["condition_key"]), str(row["seed"]))
        episode = int(row["episode"])
        previous = latest.get(key)
        if previous is None or episode >= previous[0]:
            latest[key] = (episode, float(row["masked_training_example_count"]))
    return {key: budget for key, (_, budget) in latest.items()}


def _direct_dose_response_rows(base_matrix: Path) -> list[dict[str, Any]]:
    """Seed-level direct-arm rows from the base matrix, keyed by donor rows seen.

    Replicates are the base matrix's ``(value_head, seed)`` cells, which is a
    different population from the surrogate's construction seeds; the panel
    keeps the two behind separate series for that reason.
    """

    budgets = _final_masked_budgets(base_matrix)
    selected: list[dict[str, str]] = []
    for row in read_table(base_matrix / "tables" / "fidelity_seed.parquet"):
        if row["comparison_kind"] != "direct":
            continue
        if row["action_subset"] != "all_legal" or row["evaluation_reference"] != _DOSE_RESPONSE_REFERENCE:
            continue
        key = (str(row["condition_key"]), str(row["seed"]))
        if key not in budgets:
            raise StudyAnalysisError(f"no masked-example budget for base-matrix cell {key}")
        if budgets[key] <= 0:
            # r=0 never trains the masked branch. It is the untrained control,
            # not a point on a dose-response curve, and has no place on a log axis.
            continue
        selected.append(row)

    # Episode length varies a little with the seed, so the realised budget does
    # too. Anchoring every seed of a condition at the condition's mean exposure
    # keeps the dose levels the design actually set, rather than splitting each
    # into five one-seed points with no band.
    per_condition: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        per_condition[str(row["condition_key"])].append(budgets[(str(row["condition_key"]), str(row["seed"]))])
    dose = {key: sum(values) / len(values) for key, values in per_condition.items()}

    rows: list[dict[str, Any]] = []
    for row in selected:
        value_head = str(row["value_head"])
        rows.append(
            {
                "method": "direct",
                "series": f"direct ({value_head})",
                "variant": value_head,
                "masked_batch_ratio": row["masked_batch_ratio"],
                "masked_donor_rows": dose[str(row["condition_key"])],
                "seed": row["seed"],
                "condition_key": row["condition_key"],
                "evaluation_reference": row["evaluation_reference"],
                **{metric: row[metric] for metric in _ABLATION_METRICS},
            }
        )
    return rows


def _surrogate_dose_response_rows(rows: Sequence[Mapping[str, Any]], *, batch_size: int) -> list[dict[str, Any]]:
    """Construction-arm rows keyed by the donor rows the surrogate sampled.

    Capacity is kept in the series rather than the x-axis because two capacity
    points share an update budget, and so would collide at one x value.
    """

    out: list[dict[str, Any]] = []
    for row in rows:
        if row["ablation_axis"] != "capacity_updates" or row["evaluation_reference"] != _DOSE_RESPONSE_REFERENCE:
            continue
        hidden = str(row["surrogate_hidden"])
        out.append(
            {
                "method": "surrogate",
                "series": f"surrogate ({hidden})",
                "variant": hidden,
                # The surrogate arm has no mask ratio at all. Under CSV this was an
                # empty string sharing a column with the direct arm's floats; a typed
                # table needs the honest null.
                "masked_batch_ratio": None,
                "masked_donor_rows": float(int(row["surrogate_updates"]) * batch_size),
                "seed": row["seed"],
                "condition_key": row["condition_key"],
                "evaluation_reference": row["evaluation_reference"],
                **{metric: row[metric] for metric in _ABLATION_METRICS},
            }
        )
    return out


def _budget_dose_response_rows(
    plan: SurrogateAblationPlan,
    construction_surrogate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Both dose-response curves in one long table, on a shared x-axis.

    The x-axis counts masked donor rows, which is an exposure count and not a
    matched compute budget: one direct row supervises the single replayed
    action, while one surrogate row supervises every legal action. The two
    curves also draw on different replicate populations, so the panel is a
    contrast of internally paired dose responses rather than a paired
    cross-method effect.
    """

    if not construction_surrogate:
        return []
    if not (plan.base_matrix / "tables" / "fidelity_seed.parquet").exists():
        # The ablation can be pointed at a matrix whose report was never built;
        # there is then nothing to contrast, which is not an error.
        return []
    batch_size = int(plan.construction.profile.settings["surrogate_batch_size"])
    rows = _direct_dose_response_rows(plan.base_matrix)
    rows.extend(_surrogate_dose_response_rows(construction_surrogate, batch_size=batch_size))
    return rows


def build_surrogate_ablation_report(
    plan: SurrogateAblationPlan,
    construction: StudyResult,
    support: StudyResult,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Concatenate cell tables, aggregate within seed, and plot each arm."""

    tables_dir = plan.root / "tables"
    plots_dir = plan.root / "plots"
    axis_columns = _axis_columns(plan)
    tables: list[Path] = []
    plots: list[Path] = []
    construction_surrogate: list[dict[str, Any]] = []

    for arm, result in ((CONSTRUCTION_ARM, construction), (SUPPORT_ARM, support)):
        rows: list[dict[str, Any]] = []
        for cell in result.cells:
            for row in read_table(cell.path / "fidelity.parquet"):
                rows.append({**row, **axis_columns[cell.condition_key]})
        if not rows:
            continue
        tables.append(write_table(tables_dir / f"{arm}_fidelity.parquet", rows))
        # ``action_subset`` is not a raw column: it is produced by aggregating
        # twice, once over every legal action and once over greedy actions only,
        # exactly as the mask-ratio report does.
        groups = (
            "condition_key",
            "ablation_axis",
            "surrogate_hidden",
            "surrogate_updates",
            "mask_shapley_probability",
            "policy_reference_softening",
            "evaluation_reference",
            "model_method",
            "model_reference",
            "comparison_kind",
        )
        seed_rows = [
            {**row, "action_subset": "all_legal"}
            for row in aggregate_within_seed(rows, group_fields=groups, metric_fields=_ABLATION_METRICS)
        ] + [
            {**row, "action_subset": "greedy"}
            for row in aggregate_within_seed(
                [row for row in rows if str(row["greedy_action"]).strip().lower() in {"true", "1"}],
                group_fields=groups,
                metric_fields=_ABLATION_METRICS,
            )
        ]
        seed_path = write_table(tables_dir / f"{arm}_fidelity_seed.parquet", seed_rows)
        tables.append(seed_path)

        surrogate_only = [
            row
            for row in seed_rows
            if row["model_method"] == "surrogate"
            and row["comparison_kind"] == "native_fidelity"
            and row["action_subset"] == "all_legal"
        ]
        if not surrogate_only:
            continue
        surrogate_path = write_table(tables_dir / f"{arm}_surrogate_seed.parquet", surrogate_only)
        tables.append(surrogate_path)
        if arm == CONSTRUCTION_ARM:
            construction_surrogate = surrogate_only

        # One panel set per ablation axis. The mixture conditions deliberately
        # hold the baseline capacity and budget, so projecting every axis onto
        # one plot would collide them with the baseline capacity/updates cell.
        baseline_hidden = _hidden_key(plan.profile.baseline_hidden)
        baseline_updates = plan.profile.baseline_updates
        panels: tuple[tuple[str, list[dict[str, Any]], str, tuple[str, ...]], ...]
        if arm == CONSTRUCTION_ARM:
            panels = (
                (
                    "capacity_updates",
                    [row for row in surrogate_only if row["ablation_axis"] == "capacity_updates"],
                    "surrogate_updates",
                    ("surrogate_hidden", "evaluation_reference"),
                ),
                (
                    "mask_mixture",
                    [
                        row
                        for row in surrogate_only
                        if row["surrogate_hidden"] == baseline_hidden
                        and int(row["surrogate_updates"]) == baseline_updates
                    ],
                    "mask_shapley_probability",
                    ("evaluation_reference",),
                ),
            )
        else:
            panels = (
                (
                    "reference_support",
                    surrogate_only,
                    "policy_reference_softening",
                    ("evaluation_reference",),
                ),
            )

        for panel, panel_rows, x_field, series_fields in panels:
            if len(panel_rows) < 2:
                continue
            panel_path = write_table(tables_dir / f"{arm}_{panel}_seed.parquet", panel_rows)
            tables.append(panel_path)
            for metric in ("attribution_mae", "coalition_normalized_mae"):
                plot = plot_metric_curves(
                    panel_path,
                    plots_dir / f"{panel}_{metric}.png",
                    x_field=x_field,
                    y_field=metric,
                    series_fields=series_fields,
                    facet_field="evaluation_reference",
                    title=f"{panel}: {metric}",
                    bootstrap_resamples=plan.profile.bootstrap_resamples,
                )
                if plot is not None:
                    plots.append(plot)

    dose_rows = _budget_dose_response_rows(plan, construction_surrogate)
    if dose_rows:
        dose_path = write_table(tables_dir / "budget_dose_response_seed.parquet", dose_rows)
        tables.append(dose_path)
        dose_plot = plot_metric_curves(
            dose_path,
            plots_dir / "budget_dose_response.png",
            x_field="masked_donor_rows",
            y_field="coalition_normalized_mae",
            series_fields=("series", "method"),
            facet_field="method",
            x_scale="log",
            x_label="masked donor rows seen (not compute-matched)",
            title="coalition_normalized_mae vs masked donor rows",
        )
        if dose_plot is not None:
            plots.append(dose_plot)
    return tuple(tables), tuple(plots)


def _manifest_matches(existing: Any, payload: Mapping[str, Any]) -> bool:
    """Decide whether a stored ablation manifest is the plan asking to resume it.

    Every field must match exactly except ``base_matrix``. A manifest written
    before the path was resolved holds the relative string it was given, and is
    accepted only when the current working directory makes it the same matrix --
    which keeps a published root resumable while still refusing the case the
    resolution exists to catch: one string naming two different matrices.
    """

    if not isinstance(existing, dict) or existing.keys() != payload.keys():
        return False
    if any(existing[key] != payload[key] for key in payload if key != "base_matrix"):
        return False
    stored = existing["base_matrix"]
    return isinstance(stored, str) and Path(stored).resolve() == Path(payload["base_matrix"])


def run_surrogate_ablation(
    root: Path,
    base_matrix: Path,
    *,
    profile: str | SurrogateAblationProfile = "fast",
    reuse: bool = True,
    workers: int = 1,
) -> SurrogateAblationResult:
    """Run or validate-resume the surrogate construction and support arms."""

    if workers < 1:
        raise ValueError("workers must be positive")
    plan = build_surrogate_ablation_plan(root, base_matrix, profile=profile)
    plan.root.mkdir(parents=True, exist_ok=True)
    manifest = plan.root / ABLATION_MANIFEST_FILENAME
    payload = plan.manifest_payload()
    if manifest.exists():
        if not _manifest_matches(json.loads(manifest.read_text(encoding="utf-8")), payload):
            raise StudyManifestMismatchError(f"{plan.root} belongs to a different surrogate ablation")
    else:
        if any(plan.root.iterdir()):
            raise StudyManifestMismatchError(
                f"{plan.root} is non-empty but has no {ABLATION_MANIFEST_FILENAME}; choose a fresh ablation root"
            )
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    work = SubprocessCellWork(root=plan.construction.root, handler=SURROGATE_ABLATION_HANDLER)
    support_work = SubprocessCellWork(root=plan.support.root, handler=SURROGATE_ABLATION_HANDLER)
    construction = StudyRunner(plan.construction).run(work, reuse=reuse, max_workers=workers)
    support = StudyRunner(plan.support).run(support_work, reuse=reuse, max_workers=workers)
    tables, plots = build_surrogate_ablation_report(plan, construction, support)
    return SurrogateAblationResult(
        root=plan.root,
        # The construction arm alone is blind to the support arm: two profiles
        # differing only in ``support_softening`` build identical construction
        # specs, so that fingerprint cannot tell the two studies apart.
        manifest_fingerprint=digest_manifest_payload(payload),
        construction=construction,
        support=support,
        tables=tables,
        plots=plots,
    )


def make_ablation_cell_work() -> Any:
    """Build the cell callable a worker process runs.

    The arm root sits beside the ablation manifest, which is where the frozen
    base matrix and base condition are recorded; the child rediscovers them from
    disk rather than receiving them across the process boundary.
    """

    def work(context: StudyCellContext) -> Mapping[str, Any]:
        # publish_dir is <arm root>/cells/<condition>/seed_N, so parents[2] is
        # the arm root and parents[3] is the ablation root holding the manifest.
        manifest_path = context.publish_dir.parents[3] / ABLATION_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        profile = manifest["profile"]
        return ablate_surrogate_cell(
            context,
            base_matrix=Path(manifest["base_matrix"]),
            base_condition=str(profile["base_condition"]),
            base_value_head=str(profile["base_value_head"]),
            base_masked_batch_ratio=float(profile["base_masked_batch_ratio"]),
        )

    return work


def surrogate_ablation_study(
    out: Annotated[
        Path,
        typer.Option("--out", help="Fresh or manifest-matched ablation output root."),
    ] = Path("runs/surrogate-fidelity-ablation"),
    base_matrix: Annotated[
        Path,
        typer.Option("--base-matrix", help="Completed mask-ratio study whose agents are frozen and reused."),
    ] = Path("runs/controlled-mask-ratio-research"),
    profile: Annotated[
        str,
        typer.Option("--profile", help="Budget profile: fast or research."),
    ] = "fast",
    workers: Annotated[
        int,
        typer.Option("--workers", help="Cells run concurrently, each in its own thread-pinned process."),
    ] = 1,
    no_reuse: Annotated[
        bool,
        typer.Option("--no-reuse", help="Reject existing completed cells instead of validating and reusing them."),
    ] = False,
) -> None:
    """Ablate surrogate construction against frozen mask-ratio agents."""

    result = run_surrogate_ablation(
        out,
        base_matrix,
        profile=profile,
        reuse=not no_reuse,
        workers=workers,
    )
    print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))


__all__ = [
    "ABLATION_MANIFEST_FILENAME",
    "CONSTRUCTION_ARM",
    "FAST_ABLATION_PROFILE",
    "RESEARCH_ABLATION_PROFILE",
    "SUPPORT_ARM",
    "SURROGATE_ABLATION_PROFILES",
    "SurrogateAblationPlan",
    "SurrogateAblationProfile",
    "SurrogateAblationResult",
    "ablate_surrogate_cell",
    "build_surrogate_ablation_plan",
    "build_surrogate_ablation_report",
    "run_surrogate_ablation",
    "construction_conditions",
    "load_frozen_agent",
    "make_ablation_cell_work",
    "support_conditions",
    "surrogate_ablation_profile",
    "surrogate_ablation_study",
]
