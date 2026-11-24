"""In-repo FastSHAP: amortized Shapley attributions for Q-value functions.

FastSHAP (Jethani et al., 2022) trains an explainer network that outputs the
whole per-action Shapley attribution grid in one forward pass. Coalitions are
sampled from the Shapley kernel via the existing
:class:`~xrl.explainability.distributional_q_masks.CoalitionMaskSampler`, and
coalition values come from a masked value function — by default the trained
:class:`MaskedDistributionalQSurrogate`, with a marginal-imputation fallback
that composes off-coalition features from rollout background states. The
additive efficiency normalization is applied both inside the training loss and
at inference, so ``sum(attributions) == v(full) - v(empty)`` holds exactly.

Requires the optional ``neural`` extra.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from xrl._provenance import provenance_payload, require_artifact_matches_run
from xrl.explainability.adapters import get_explanation_adapter, load_explanation_run
from xrl.explainability.backgrounds import BackgroundDataset
from xrl.explainability.core import ExplanationResult, ExplanationSemantics, PolicyView, validate_action
from xrl.explainability.distributional_q_masks import (
    CoalitionMaskSampler,
    CoalitionMaskSamplerConfig,
    masked_surrogate_inputs,
)
from xrl.training.neural_q import jax, jnp, nn, optax, require_neural_dependencies, serialization

FASTSHAP_ASSUMPTION = (
    "amortized FastSHAP explainer trained against a masked value function (additively normalized to satisfy efficiency)"
)

# Effectively disables the sampler's periodic full/empty masks: both coalitions
# carry zero loss under efficiency normalization, so they only waste batch slots.

VALUE_FUNCTION_CHOICES = ("auto", "surrogate", "marginal", "direct")

# Resolved value-function modes, as persisted in ``fastshap_summary.json`` and
# reported through ``ExplanationSemantics``. Distinct from the config choices
# above: ``auto`` resolves to one of these, and the surrogate mode keeps its
# historical spelling so existing artifacts stay readable.
MODE_SURROGATE = "masked-surrogate"
MODE_MARGINAL = "marginal"
MODE_DIRECT = "direct"


def resolve_value_mode(config: FastShapConfig) -> str:
    """Return the coalition game a config selects.

    ``auto`` reproduces the behaviour that predates ``value_function``: the
    surrogate when one was supplied, marginal imputation otherwise. Never
    ``direct`` — an explainable-Q run would silently change games on reload.
    """

    if config.value_function == "surrogate":
        return MODE_SURROGATE
    if config.value_function == "marginal":
        return MODE_MARGINAL
    if config.value_function == "direct":
        return MODE_DIRECT
    return MODE_SURROGATE if config.q_surrogate_dir is not None else MODE_MARGINAL


MaskedValueFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""``(states (N, d), known_masks (N, d)) -> (N, num_actions)`` coalition values."""

if nn is not None:

    class FastShapNetwork(nn.Module):
        """MLP mapping an observation to a (num_actions, num_features) phi grid."""

        num_actions: int
        num_features: int
        hidden_sizes: tuple[int, ...] = (128, 128)

        @nn.compact
        def __call__(self, x):
            for hidden_size in self.hidden_sizes:
                x = nn.relu(nn.Dense(hidden_size)(x))
            x = nn.Dense(self.num_actions * self.num_features)(x)
            return x.reshape((x.shape[0], self.num_actions, self.num_features))

else:
    FastShapNetwork = None


@dataclass(frozen=True, slots=True)
class FastShapConfig:
    """Configuration for training an amortized FastSHAP explainer."""

    base_run_dir: Path
    out_dir: Path = Path("runs/fastshap")
    q_surrogate_dir: Path | None = None
    background_path: Path | None = None
    background_states: int = 1_024
    seed: int = 0
    updates: int = 2_000
    batch_size: int = 32
    coalitions_per_state: int = 8
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (128, 128)
    marginal_samples: int = 16
    log_interval: int = 50
    feature_count: int = 0
    num_actions: int = 0
    # Which coalition game to train against. FastSHAP estimates the Shapley
    # values of whatever value function it saw, so this is part of the
    # explainer's identity rather than a tuning knob. ``auto`` reproduces the
    # pre-existing resolution exactly, keeping saved configs meaningful.
    value_function: str = "auto"

    def __post_init__(self) -> None:
        if self.value_function not in VALUE_FUNCTION_CHOICES:
            raise ValueError(f"value_function must be one of {VALUE_FUNCTION_CHOICES}, got {self.value_function!r}")
        if self.value_function == "surrogate" and self.q_surrogate_dir is None:
            raise ValueError("value_function 'surrogate' requires q_surrogate_dir")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.coalitions_per_state < 1:
            raise ValueError("coalitions_per_state must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.background_states < 1:
            raise ValueError("background_states must be positive")
        if self.marginal_samples < 1:
            raise ValueError("marginal_samples must be positive")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_run_dir"] = str(self.base_run_dir)
        data["out_dir"] = str(self.out_dir)
        data["q_surrogate_dir"] = None if self.q_surrogate_dir is None else str(self.q_surrogate_dir)
        data["background_path"] = None if self.background_path is None else str(self.background_path)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> FastShapConfig:
        normalized = dict(data)
        # Provenance is a property of the written artifact, not of the requested
        # configuration, so it is recorded beside the config and dropped here.
        normalized.pop("base_run_started_at", None)
        for key in ("base_run_dir", "out_dir", "q_surrogate_dir", "background_path"):
            if normalized.get(key) is not None:
                normalized[key] = Path(normalized[key])
        normalized["hidden_sizes"] = tuple(normalized.get("hidden_sizes", (128, 128)))
        return cls(**normalized)


def surrogate_masked_value_function(surrogate: Any) -> MaskedValueFunction:
    """Batch coalition values from a masked distributional Q surrogate."""

    def evaluate(states: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        inputs = jnp.asarray(masked_surrogate_inputs(states, known_masks), dtype=jnp.float32)
        logits = surrogate.model.apply(surrogate.params, inputs)
        probabilities = jax.nn.softmax(logits, axis=-1)
        support = jnp.asarray(np.asarray(surrogate.support, dtype=np.float32))
        return np.asarray(jnp.sum(probabilities * support[None, None, :], axis=-1), dtype=np.float64)

    return evaluate


def direct_masked_value_function(policy: Any) -> MaskedValueFunction:
    """Batch coalition values from a jointly trained mask-aware Q-function.

    No distilled surrogate stands between the amortized explainer and the
    deployed policy, so the FastSHAP network is trained against exactly the
    game :func:`~xrl.explainability.distributional_q_explainer.direct_q_explanation`
    enumerates.
    """

    if not callable(getattr(policy, "masked_q_values", None)):
        raise TypeError("direct FastSHAP requires feature_mask_input=true and masked_batch_ratio>0")
    batch = getattr(policy, "masked_q_values_batch", None)
    encoder = getattr(getattr(policy, "config", None), "encoder", None)

    def evaluate(states: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        rows = np.asarray(states, dtype=np.float32)
        masks = np.asarray(known_masks, dtype=np.float32)
        # Only the identity encoder leaves table coordinates equal to
        # environment actions; anything else needs the per-row view mapping.
        if batch is not None and encoder == "identity":
            return np.asarray(batch(rows, masks), dtype=np.float64)
        return np.asarray(
            [policy.masked_q_values(row, mask) for row, mask in zip(rows, masks, strict=True)],
            dtype=np.float64,
        )

    return evaluate


def marginal_masked_value_function(
    policy: PolicyView,
    background_observations: np.ndarray,
    *,
    samples: int = 16,
    seed: int = 0,
) -> MaskedValueFunction:
    """Coalition values by imputing off-coalition features from background draws."""

    draws_pool = np.asarray(background_observations, dtype=np.float32)
    rng = np.random.default_rng(seed)

    def evaluate(states: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        states_array = np.asarray(states, dtype=np.float32)
        masks_array = np.asarray(known_masks, dtype=np.float32)
        draws = draws_pool[rng.choice(draws_pool.shape[0], size=samples)]
        composed = np.where(masks_array[:, None, :] > 0.0, states_array[:, None, :], draws[None, :, :])
        flat = composed.reshape(-1, states_array.shape[1])
        q_matrix = _policy_q_matrix(policy, flat)
        return q_matrix.reshape(states_array.shape[0], samples, -1).mean(axis=1)

    return evaluate


def _policy_q_matrix(policy: PolicyView, matrix: np.ndarray) -> np.ndarray:
    batch = getattr(policy, "q_values_batch", None)
    encoder = getattr(getattr(policy, "config", None), "encoder", None)
    if batch is not None and encoder == "identity":
        return np.asarray(batch(matrix), dtype=np.float64)
    return np.asarray([policy.q_values(row) for row in matrix], dtype=np.float64)


class FastShapExplainer:
    """One-forward-pass Shapley attributions from a trained FastSHAP network."""

    def __init__(
        self,
        *,
        network: Any,
        params: Any,
        feature_names: Sequence[str],
        num_actions: int,
        value_fn: MaskedValueFunction,
        coalition_semantics: str = "unspecified",
        value_model_source: str = "unspecified",
    ) -> None:
        self.network = network
        self.params = params
        self.feature_names = tuple(feature_names)
        self.num_actions = int(num_actions)
        self._value_fn = value_fn
        self.coalition_semantics = coalition_semantics
        self.value_model_source = value_model_source

    def explain_action(
        self,
        observation: np.ndarray,
        *,
        action: int,
        action_mask: np.ndarray | Iterable[int] | None = None,
    ) -> ExplanationResult:
        """Explain the masked value function's Q(observation, action).

        ``action_mask`` is optional only because this class is exported and a
        caller may hold no mask. Pass one whenever it is available: without it
        an illegal action can be explained, and the attributions then describe
        an output no TD target ever constrained. See ADR 0024.
        """

        x = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x.shape[1]}")
        # Validate before the forward pass, not after: an out-of-range action
        # should not cost a network evaluation first.
        action = validate_action(action, action_mask, num_actions=self.num_actions)
        phi = np.asarray(self.network.apply(self.params, jnp.asarray(x)), dtype=np.float64)[0]
        full_values = self._value_fn(x, np.ones_like(x))[0]
        empty_values = self._value_fn(x, np.zeros_like(x))[0]
        gap = full_values - empty_values - phi.sum(axis=-1)
        phi = phi + gap[:, None] / phi.shape[1]
        row = phi[int(action)]
        return ExplanationResult(
            target="q_value",
            state=tuple(float(value) for value in x[0]),
            action=int(action),
            baseline_value=float(empty_values[int(action)]),
            full_value=float(full_values[int(action)]),
            attributions={name: float(value) for name, value in zip(self.feature_names, row, strict=True)},
            feature_names=self.feature_names,
            characteristic_assumption=FASTSHAP_ASSUMPTION,
            semantics=ExplanationSemantics(
                target_quantity="fixed_action_q",
                coalition_semantics=self.coalition_semantics,
                value_model_source=self.value_model_source,
                shapley_estimator="fastshap",
            ),
        )


def train_fastshap_network(
    *,
    value_fn: MaskedValueFunction,
    train_states: np.ndarray,
    num_actions: int,
    config: FastShapConfig,
) -> tuple[Any, Any, list[dict[str, float | int]]]:
    """Train a FastShapNetwork against a masked value function.

    Returns ``(network, params, metrics)``. Exposed separately from
    :class:`FastShapTrainer` so tests and callers can train against synthetic
    value functions without run directories.
    """

    require_neural_dependencies("FastSHAP training")
    states = np.asarray(train_states, dtype=np.float32)
    feature_count = states.shape[1]
    network = FastShapNetwork(
        num_actions=int(num_actions),
        num_features=feature_count,
        hidden_sizes=config.hidden_sizes,
    )
    params = network.init(jax.random.PRNGKey(config.seed), jnp.zeros((1, feature_count), dtype=jnp.float32))
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    train_step = _make_train_step(network, optimizer, feature_count)
    sampler = CoalitionMaskSampler(
        CoalitionMaskSamplerConfig(
            feature_count=feature_count,
            full_mask_example_interval=None,
            empty_mask_example_interval=None,
            shapley_probability=1.0,
            uniform_probability=0.0,
            guided_probability=0.0,
        ),
        seed=config.seed,
    )
    rng = np.random.default_rng(config.seed)
    metrics: list[dict[str, float | int]] = []
    repeat = config.coalitions_per_state
    for update in range(config.updates):
        indices = rng.choice(states.shape[0], size=config.batch_size, replace=True)
        batch_states = states[indices]
        repeated = np.repeat(batch_states, repeat, axis=0)
        masks = sampler.sample_batch(repeated.shape[0], example_offset=update * repeated.shape[0])
        coalition_values = value_fn(repeated, masks)
        empty_values = np.repeat(value_fn(batch_states, np.zeros_like(batch_states)), repeat, axis=0)
        full_values = np.repeat(value_fn(batch_states, np.ones_like(batch_states)), repeat, axis=0)
        params, opt_state, loss = train_step(
            params,
            opt_state,
            jnp.asarray(repeated, dtype=jnp.float32),
            jnp.asarray(masks, dtype=jnp.float32),
            jnp.asarray(coalition_values, dtype=jnp.float32),
            jnp.asarray(empty_values, dtype=jnp.float32),
            jnp.asarray(full_values, dtype=jnp.float32),
        )
        if update % config.log_interval == 0 or update == config.updates - 1:
            metrics.append({"update": update + 1, "loss": float(loss)})
    return network, params, metrics


def _make_train_step(network: Any, optimizer: Any, feature_count: int):
    def train_step(params, opt_state, states, masks, coalition_values, empty_values, full_values):
        def loss_fn(current_params):
            phi = network.apply(current_params, states)
            gap = full_values - empty_values - jnp.sum(phi, axis=-1)
            normalized = phi + gap[:, :, None] / feature_count
            predicted = jnp.sum(normalized * masks[:, None, :], axis=-1)
            return jnp.mean((coalition_values - empty_values - predicted) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    return jax.jit(train_step)


class FastShapTrainer:
    """Train, persist, and reload FastSHAP explainers for saved runs."""

    def __init__(self, config: FastShapConfig) -> None:
        require_neural_dependencies("FastSHAP")
        self.config = config

    def train(self) -> tuple[FastShapExplainer, dict[str, Any]]:
        # Provenance is checked here, where the user chose the pairing, and not
        # on the load path: `load` has no selected run to check against, so a
        # failure there would re-litigate a decision already made. See ADR 0023.
        if resolve_value_mode(self.config) == MODE_SURROGATE and self.config.q_surrogate_dir is not None:
            from xrl.explainability.distributional_q_explainer import validate_surrogate_base_run

            validate_surrogate_base_run(self.config.q_surrogate_dir, self.config.base_run_dir)
        run = load_explanation_run(self.config.base_run_dir)
        background, background_path = self._resolve_background(run)
        value_fn, value_mode = self._value_function(run, background)
        config = replace(
            self.config,
            feature_count=int(background.observations.shape[1]),
            num_actions=int(run.num_actions),
            background_path=background_path,
        )
        network, params, metrics = train_fastshap_network(
            value_fn=value_fn,
            train_states=background.observations,
            num_actions=run.num_actions,
            config=config,
        )
        explainer = FastShapExplainer(
            network=network,
            params=params,
            feature_names=run.feature_spec.names,
            num_actions=run.num_actions,
            value_fn=value_fn,
            **_value_mode_semantics(value_mode),
        )
        summary = self._write_artifacts(config, params, metrics, value_mode)
        return explainer, summary

    def load(self, run_dir: Path | None = None, *, base_run_dir: Path | None = None) -> FastShapExplainer:
        run_dir = run_dir or self.config.out_dir
        config = FastShapConfig.from_json_dict(
            json.loads((run_dir / "fastshap_config.json").read_text(encoding="utf-8"))
        )
        if config.feature_count < 1 or config.num_actions < 1:
            raise ValueError(f"fastshap config in {run_dir} is missing trained network dimensions")
        # The recorded base_run_dir is provenance, not a lookup key: it was
        # written under the trainer's working directory and the run may since
        # have moved. Prefer the run the caller actually selected.
        run = load_explanation_run(base_run_dir if base_run_dir is not None else config.base_run_dir)
        background = None
        if resolve_value_mode(config) == MODE_MARGINAL:
            if config.background_path is None:
                raise ValueError(f"marginal fastshap config in {run_dir} has no background_path")
            background = BackgroundDataset.load(config.background_path)
        value_fn, _mode = self._value_function(run, background, config=config)
        network = FastShapNetwork(
            num_actions=config.num_actions,
            num_features=config.feature_count,
            hidden_sizes=config.hidden_sizes,
        )
        template = network.init(
            jax.random.PRNGKey(config.seed), jnp.zeros((1, config.feature_count), dtype=jnp.float32)
        )
        params = serialization.from_bytes(template, (run_dir / "fastshap.msgpack").read_bytes())
        return FastShapExplainer(
            network=network,
            params=params,
            feature_names=run.feature_spec.names,
            num_actions=run.num_actions,
            value_fn=value_fn,
            **_value_mode_semantics(_mode),
        )

    def _resolve_background(self, run) -> tuple[BackgroundDataset, Path | None]:
        if self.config.background_path is not None:
            return BackgroundDataset.load(self.config.background_path), self.config.background_path
        adapter = get_explanation_adapter(run.env_id)
        background = adapter.collect_background(
            run.policy,
            run.config,
            states=self.config.background_states,
            seed=self.config.seed,
        )
        if resolve_value_mode(self.config) != MODE_MARGINAL:
            # Only the marginal value function needs background rows at load
            # time; the model-backed games rebuild ``v`` from a checkpoint.
            return background, None
        path = self.config.out_dir / "fastshap_background.npz"
        background.save(path)
        return background, path

    def _value_function(
        self,
        run,
        background: BackgroundDataset | None,
        config: FastShapConfig | None = None,
    ) -> tuple[MaskedValueFunction, str]:
        config = config or self.config
        mode = resolve_value_mode(config)
        if mode == MODE_SURROGATE:
            from xrl.explainability.distributional_q_trainer import load_distributional_q_surrogate

            if config.q_surrogate_dir is None:
                raise ValueError("surrogate FastSHAP needs a masked-Q surrogate run")
            surrogate = load_distributional_q_surrogate(Path(config.q_surrogate_dir))
            return surrogate_masked_value_function(surrogate), mode
        if mode == MODE_DIRECT:
            return direct_masked_value_function(run.policy), mode
        if background is None:
            raise ValueError("marginal FastSHAP needs a background dataset")
        value_fn = marginal_masked_value_function(
            run.policy,
            background.observations,
            samples=config.marginal_samples,
            seed=config.seed,
        )
        return value_fn, mode

    def _write_artifacts(
        self,
        config: FastShapConfig,
        params: Any,
        metrics: list[dict[str, float | int]],
        value_mode: str,
    ) -> dict[str, Any]:
        out_dir = config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_config = {**config.to_json_dict(), **provenance_payload(config.base_run_dir)}
        (out_dir / "fastshap_config.json").write_text(json.dumps(artifact_config, indent=2), encoding="utf-8")
        (out_dir / "fastshap.msgpack").write_bytes(serialization.to_bytes(params))
        with (out_dir / "fastshap_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["update", "loss"])
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)
        summary = {
            "updates": config.updates,
            "final_loss": float(metrics[-1]["loss"]) if metrics else math.nan,
            "value_function": value_mode,
            **provenance_payload(config.base_run_dir),
            "explainer": "fastshap",
        }
        (out_dir / "fastshap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def load_fastshap_explainer(run_dir: Path, *, base_run_dir: Path | None = None) -> FastShapExplainer:
    """Load a saved FastSHAP explainer run.

    ``base_run_dir`` names the run the caller selected. Pass it whenever one is
    known: the artifact's recorded path may no longer resolve, or may resolve to
    a different copy of the run.
    """

    config = FastShapConfig.from_json_dict(json.loads((run_dir / "fastshap_config.json").read_text(encoding="utf-8")))
    return FastShapTrainer(config).load(run_dir, base_run_dir=base_run_dir)


def validate_fastshap_base_run(fastshap_dir: str | Path, base_run_dir: str | Path) -> None:
    """Reject a FastSHAP artifact that belongs to another selected run."""

    require_artifact_matches_run(
        fastshap_dir,
        base_run_dir,
        config_name="fastshap_config.json",
        label="FastSHAP artifact",
    )


def fastshap_value_mode(fastshap_dir: str | Path) -> str:
    """Return the coalition game a saved FastSHAP explainer was trained against."""

    config_path = Path(fastshap_dir) / "fastshap_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read FastSHAP provenance from {config_path}: {error}") from error
    return resolve_value_mode(FastShapConfig.from_json_dict(payload))


def validate_fastshap_mode(fastshap_dir: str | Path, expected: str) -> None:
    """Reject a FastSHAP artifact trained against a different coalition game.

    FastSHAP estimates the Shapley values of the value function it saw, so a
    marginally-trained network is not an approximation of the masked-Q values —
    it is an answer to a different question. Scoring one as the other is the
    failure this guards.
    """

    actual = fastshap_value_mode(fastshap_dir)
    if actual != expected:
        raise ValueError(
            f"FastSHAP artifact {fastshap_dir} was trained against the {actual!r} game, not {expected!r}; "
            f"retrain with --value-function or select the matching method"
        )


def _value_mode_semantics(value_mode: str) -> dict[str, str]:
    """Map a resolved value mode to the game the explainer answers.

    This is the only place the game is declared; the comparison harness derives
    family and reference selection from what lands in
    :class:`~xrl.explainability.core.ExplanationSemantics`.
    """

    if value_mode == MODE_SURROGATE:
        return {
            "coalition_semantics": "surrogate_masked_q",
            "value_model_source": "masked_q_surrogate",
        }
    if value_mode == MODE_MARGINAL:
        return {
            "coalition_semantics": "marginal_background",
            "value_model_source": "base_q_function",
        }
    if value_mode == MODE_DIRECT:
        return {
            "coalition_semantics": "direct_masked_q",
            "value_model_source": "base_q_function",
        }
    return {"coalition_semantics": "unspecified", "value_model_source": "unspecified"}
