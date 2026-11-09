"""Frozen expert inputs for Stage2.

Reuse the Stage1 ``holdout_train / holdout_val / outer_test`` indices without
creating another cross-validation layer.
"""

from hpsafe_sota.expert_inputs.protocol import (
    CANONICAL_TASKS,
    SPLIT_ROLE_TO_MANIFEST_ROLE,
    ExpertPool,
    load_expert_pool,
)

__all__ = [
    "CANONICAL_TASKS",
    "SPLIT_ROLE_TO_MANIFEST_ROLE",
    "ExpertPool",
    "load_expert_pool",
]
