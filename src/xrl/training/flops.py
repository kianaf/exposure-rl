"""Training-compute accounting in FLOPs.

Wall-clock seconds cannot match two training arms against each other: they
measure the machine and whatever else it was doing, not the work. FLOPs measure
the work, which is what makes "same compute, which arm is better?" and "how much
compute until they are equal?" answerable questions.

The number comes from XLA, by lowering and compiling the jitted step and reading
the ``flops`` entry of its cost analysis. That is the work the backend actually
performs, including the fusions an analytic count would miss.

Cost analysis is backend-dependent — it is absent or empty on some backends and
its shape has changed across JAX versions — so every measurement carries the
:class:`StepCost` ``source`` that produced it, and an analytic MLP count stands
in when XLA declines. The two agree to within a few percent on the small
networks here (the analytic count omits biases and activations), but they are
not interchangeable: a study that mixed them would be comparing two different
definitions, so ``source`` is recorded alongside every total and the benchmark
refuses to compare across sources.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FLOPS_SOURCE_ANALYTIC",
    "FLOPS_SOURCE_XLA",
    "FLOPS_ROW_FIELDS",
    "StepCost",
    "compiled_step_flops",
    "dqn_update_flops",
    "learner_flops",
    "learner_training_accounting",
    "measure_step_cost",
    "mlp_forward_flops",
    "surrogate_update_flops",
]

# Columns stamped onto every evaluation row by a learner that counts FLOPs.
# ``training_flops`` is cumulative, so the curve — not just the endpoint — is on
# disk, which is what "FLOPs until equal performance" needs.
FLOPS_ROW_FIELDS: tuple[str, ...] = ("training_flops", "flops_source")

FLOPS_SOURCE_XLA = "xla_cost_analysis"
FLOPS_SOURCE_ANALYTIC = "analytic_mlp"

# A backward pass costs about twice its forward pass: one product for the
# gradient with respect to the activations, one for the weights.
_BACKWARD_MULTIPLIER = 3.0


@dataclass(frozen=True, slots=True)
class StepCost:
    """FLOPs for one training update, and which method produced the number."""

    flops_per_update: float
    source: str

    def __post_init__(self) -> None:
        if not self.flops_per_update > 0:
            raise ValueError(f"flops_per_update must be positive, got {self.flops_per_update}")
        if self.source not in (FLOPS_SOURCE_XLA, FLOPS_SOURCE_ANALYTIC):
            raise ValueError(f"unknown FLOPs source {self.source!r}")

    def total(self, updates: int) -> float:
        """Cumulative FLOPs after ``updates`` applications of this step."""

        if updates < 0:
            raise ValueError(f"updates must be non-negative, got {updates}")
        return self.flops_per_update * updates


def compiled_step_flops(step: Callable[..., Any], args: Sequence[Any]) -> float | None:
    """FLOPs for one call of ``step`` on ``args``, or ``None`` if XLA won't say.

    ``step`` is expected to be the jitted function itself, so the measurement
    describes the compiled program the trainer actually runs.
    """

    lower = getattr(step, "lower", None)
    if lower is None:
        return None
    try:
        analysis = lower(*args).compile().cost_analysis()
    except Exception:  # noqa: BLE001 - see module docstring: the fallback is recorded, never silent
        return None
    if isinstance(analysis, Sequence) and not isinstance(analysis, dict | str):
        # Older JAX returned one mapping per compiled computation.
        analysis = analysis[0] if analysis else None
    if not isinstance(analysis, dict):
        return None
    flops = analysis.get("flops")
    if flops is None or not float(flops) > 0:
        return None
    return float(flops)


def mlp_forward_flops(
    *,
    input_width: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    batch_size: int,
) -> float:
    """Multiply-accumulate FLOPs for one forward pass of an MLP over a batch.

    Counts ``2 * fan_in * fan_out`` per layer per row — one multiply and one add
    — and ignores biases and activations, which are lower order.
    """

    for name, value in (("input_width", input_width), ("output_size", output_size), ("batch_size", batch_size)):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    widths = [int(input_width), *(int(size) for size in hidden_sizes), int(output_size)]
    if any(width < 1 for width in widths):
        raise ValueError(f"hidden sizes must be positive, got {tuple(hidden_sizes)}")
    per_row = sum(2.0 * fan_in * fan_out for fan_in, fan_out in zip(widths, widths[1:], strict=False))
    return per_row * int(batch_size)


def dqn_update_flops(
    *,
    input_width: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    full_batch_size: int,
    masked_batch_size: int,
) -> float:
    """Analytic FLOPs for one DQN replay update.

    One target-network forward pass on the next observations, which is not
    differentiated, plus a forward and backward pass over the current
    observations. A mask-aware update adds one differentiated pass over its
    appended masked examples. The target and full batches stay at size ``B``;
    the masked batch has independently declared size ``M``.
    """

    if masked_batch_size < 0:
        raise ValueError(f"masked_batch_size must be non-negative, got {masked_batch_size}")
    full_forward = mlp_forward_flops(
        input_width=input_width,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        batch_size=full_batch_size,
    )
    masked_forward = 0.0
    if masked_batch_size:
        masked_forward = mlp_forward_flops(
            input_width=input_width,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            batch_size=masked_batch_size,
        )
    return full_forward + _BACKWARD_MULTIPLIER * (full_forward + masked_forward)


def surrogate_update_flops(
    *,
    input_width: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    batch_size: int,
) -> float:
    """Analytic FLOPs for one masked-surrogate distillation update.

    A single differentiated forward pass: the targets come from the frozen base
    model's precomputed distributions, so there is no target network to evaluate
    inside the step.
    """

    return _BACKWARD_MULTIPLIER * mlp_forward_flops(
        input_width=input_width,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        batch_size=batch_size,
    )


def learner_flops(learner: Any) -> dict[str, Any]:
    """Cumulative-FLOPs columns for ``learner``, empty if it does not count them.

    Tabular and online-Q learners have no compiled step to cost, so they simply
    contribute no columns rather than a misleading zero.
    """

    total = getattr(learner, "training_flops", None)
    source = getattr(learner, "flops_source", None)
    if total is None or source is None:
        return {}
    return {"training_flops": float(total), "flops_source": str(source)}


def learner_training_accounting(learner: Any) -> dict[str, Any]:
    """Return work counters exposed by a neural replay learner.

    Unsupported counters are omitted rather than inferred, which keeps the
    shared artifact writer valid for tabular and online-Q learners. DQN agents
    expose every field below, allowing one saved run to be compared at episode,
    transition, predictor-example, or FLOP budgets.
    """

    fields = {
        "replay_transition_count": "transition_count",
        "gradient_update_count": "update_count",
        "full_training_example_count": "full_example_count",
        "masked_training_example_count": "masked_example_count",
        "masked_legal_target_count": "masked_legal_target_count",
        "predictor_training_example_count": "predictor_example_count",
    }
    accounting: dict[str, Any] = {}
    for output_name, attribute_name in fields.items():
        value = getattr(learner, attribute_name, None)
        if value is not None:
            accounting[output_name] = int(value)
    accounting.update(learner_flops(learner))
    return accounting


def measure_step_cost(
    step: Callable[..., Any],
    args: Sequence[Any],
    *,
    fallback: Callable[[], float],
) -> StepCost:
    """Cost one update from XLA, falling back to an analytic count.

    The returned :class:`StepCost` records which of the two produced the number,
    so a study can refuse to compare totals measured different ways.
    """

    measured = compiled_step_flops(step, args)
    if measured is not None:
        return StepCost(flops_per_update=measured, source=FLOPS_SOURCE_XLA)
    return StepCost(flops_per_update=float(fallback()), source=FLOPS_SOURCE_ANALYTIC)
