"""Live training progress for the terminal, in one format for every environment.

A long sweep is otherwise silent for minutes at a time, which makes it
impossible to tell a slow run from a stuck one, or to notice that a
configuration is learning nothing until it has finished.

Everything here writes to **stderr**. Each command's machine-readable result
goes to stdout, so ``xrl train ... > result.json`` and ``| jq`` keep working
while progress still reaches the terminal.

Fields are *discovered*, not configured. :data:`EPISODE_FIELDS` and
:data:`EVALUATION_FIELDS` list every metric worth watching along with the
attribute names it may go by; a reporter prints the ones its metrics actually
carry. Tic Tac Toe has a draw rate and no board completion, Minesweeper the
reverse, and neither needed to say so. A new environment gets a sensible line
by exposing any of these names — and a genuinely new metric is one entry here,
not a new reporter.

Reporters are deliberately dumb: they format numbers the trainers already
compute. Nothing here influences training.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TextIO

__all__ = [
    "EPISODE_FIELDS",
    "EVALUATION_FIELDS",
    "ProgressField",
    "ProgressReporter",
    "TrainingProgress",
    "announce",
    "format_duration",
]


def announce(message: str) -> None:
    """Write one milestone line to stderr, keeping stdout parseable JSON.

    For the coarse "what is happening now" lines a long command emits between
    its per-episode updates: which variant started, where output is going.
    """

    print(f"=== {message}", file=sys.stderr, flush=True)


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _epsilon(value: float) -> str:
    return f"{value:.3f}"


def _small(value: float) -> str:
    return f"{value:.4f}"


def _whole(value: float) -> str:
    return f"{int(value)}"


@dataclass(frozen=True, slots=True)
class ProgressField:
    """One metric to display, and the attribute names it may go by.

    ``sources`` is ordered: the first name present on the metric wins. That is
    what lets a single field span environments that compute the same quantity
    under different names.
    """

    label: str
    sources: tuple[str, ...]
    render: Callable[[float], str]

    def read(self, metric: Any) -> str | None:
        for source in self.sources:
            value = metric.get(source) if isinstance(metric, Mapping) else getattr(metric, source, None)
            if value is not None:
                return f"{self.label}={self.render(float(value))}"
        return None


# Per-episode fields, in display order. Rolling means are the learning curve;
# the raw per-episode value would be far too noisy to read.
EPISODE_FIELDS: tuple[ProgressField, ...] = (
    ProgressField("win", ("rolling_win_rate",), _percent),
    ProgressField("draw", ("rolling_draw_rate",), _percent),
    ProgressField("cells", ("rolling_completion_ratio",), _percent),
    ProgressField("return", ("rolling_learner_return",), _signed),
    ProgressField("ε", ("epsilon",), _epsilon),
    ProgressField("opp-random", ("opponent_random_move_probability",), _percent),
    # Both names mean "how large is the training signal": the neural loss for
    # Minesweeper, the TD error for the tabular and AEC learners.
    ProgressField("td", ("loss_mean", "td_error_mean"), _small),
)

EVALUATION_FIELDS: tuple[ProgressField, ...] = (
    ProgressField("win", ("win_rate",), _percent),
    ProgressField("draw", ("draw_rate",), _percent),
    ProgressField("loss", ("loss_rate",), _percent),
    ProgressField("cells", ("mean_completion_ratio",), _percent),
    ProgressField("return", ("mean_return", "mean_learner_return"), _signed),
    ProgressField("over", ("episodes",), _whole),
)


class ProgressReporter(Protocol):
    """Sink for periodic training progress."""

    def episode(self, metric: Any, *, episode: int, total: int) -> None:
        """Report one finished episode; the reporter decides whether to print."""

    def evaluation(self, row: Mapping[str, Any], *, episode: int, total: int, context: str = "") -> None:
        """Report one evaluation checkpoint."""


def format_duration(seconds: float) -> str:
    """Render a duration as ``1h02m``, ``3m04s``, or ``12s``."""

    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


@dataclass
class TrainingProgress:
    """Prints periodic episode lines and every evaluation checkpoint.

    ``interval`` is in episodes; ``0`` disables episode lines. Evaluation
    checkpoints always print when they happen, because they are already rare
    and are the number most worth watching.
    """

    label: str = ""
    interval: int = 0
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    episode_fields: Sequence[ProgressField] = EPISODE_FIELDS
    evaluation_fields: Sequence[ProgressField] = EVALUATION_FIELDS
    _started: float = field(default_factory=time.perf_counter, init=False)

    def episode(self, metric: Any, *, episode: int, total: int) -> None:
        if self.interval <= 0:
            return
        if episode % self.interval != 0 and episode != total:
            return
        self._write(f"ep {episode}/{total}", self.episode_fields, metric, episode=episode, total=total)

    def evaluation(self, row: Mapping[str, Any], *, episode: int, total: int, context: str = "") -> None:
        heading = f"eval @{episode}" + (f" vs {context}" if context else "")
        self._write(heading, self.evaluation_fields, row, episode=episode, total=total, marker="*")

    def _write(
        self,
        heading: str,
        fields: Sequence[ProgressField],
        metric: Any,
        *,
        episode: int,
        total: int,
        marker: str = " ",
    ) -> None:
        rendered = [value for value in (spec.read(metric) for spec in fields) if value is not None]
        elapsed = time.perf_counter() - self._started
        prefix = f"{self.label} " if self.label else ""
        body = "  ".join([heading, *rendered])
        line = f"  {marker} {prefix}{body}  [{format_duration(elapsed)}{_eta(elapsed, episode, total)}]"
        print(line, file=self.stream, flush=True)


def _eta(elapsed: float, episode: int, total: int) -> str:
    if episode <= 0 or episode >= total:
        return ""
    remaining = elapsed / episode * (total - episode)
    return f", eta {format_duration(remaining)}"
