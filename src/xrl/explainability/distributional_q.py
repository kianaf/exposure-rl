"""Compatibility exports for distributional Q surrogate explainability."""

from xrl.explainability.distributional_q_cli import train_q_surrogate
from xrl.explainability.distributional_q_dataset import (
    SurrogateTrainingDataset,
    build_rollout_surrogate_dataset,
    build_surrogate_training_dataset,
    categorical_q_distributions,
    load_categorical_dqn_agent,
    require_categorical_agent,
)
from xrl.explainability.distributional_q_explainer import (
    DIRECT_MASKED_Q_ASSUMPTION,
    EXACT_SHAPLEY_MAX_FEATURES,
    GENERIC_MASKED_Q_ASSUMPTION,
    SURROGATE_MASKED_Q_ASSUMPTION,
    DirectMaskedQAdapter,
    DistributionalQExplainer,
    DistributionalQExplanationResult,
    direct_q_explanation,
    load_validated_surrogate,
    masked_q_explanation,
    surrogate_explanation,
    surrogate_masked_q_explanation,
    validate_surrogate_base_run,
)
from xrl.explainability.distributional_q_masks import (
    CoalitionMaskSampler,
    CoalitionMaskSamplerConfig,
    MaskedQValueModel,
    masked_surrogate_inputs,
)
from xrl.explainability.distributional_q_model import (
    DistributionalQSurrogateConfig,
    MaskedDistributionalQNetwork,
    MaskedDistributionalQSurrogate,
)
from xrl.explainability.distributional_q_trainer import (
    DistributionalQSurrogateTrainer,
    load_distributional_q_surrogate,
)

__all__ = [
    "EXACT_SHAPLEY_MAX_FEATURES",
    "DIRECT_MASKED_Q_ASSUMPTION",
    "GENERIC_MASKED_Q_ASSUMPTION",
    "SURROGATE_MASKED_Q_ASSUMPTION",
    "CoalitionMaskSampler",
    "CoalitionMaskSamplerConfig",
    "DistributionalQExplainer",
    "DistributionalQExplanationResult",
    "DirectMaskedQAdapter",
    "DistributionalQSurrogateConfig",
    "DistributionalQSurrogateTrainer",
    "MaskedDistributionalQNetwork",
    "MaskedDistributionalQSurrogate",
    "MaskedQValueModel",
    "SurrogateTrainingDataset",
    "build_rollout_surrogate_dataset",
    "build_surrogate_training_dataset",
    "categorical_q_distributions",
    "load_categorical_dqn_agent",
    "load_distributional_q_surrogate",
    "masked_surrogate_inputs",
    "masked_q_explanation",
    "direct_q_explanation",
    "require_categorical_agent",
    "load_validated_surrogate",
    "surrogate_explanation",
    "surrogate_masked_q_explanation",
    "validate_surrogate_base_run",
    "train_q_surrogate",
]
