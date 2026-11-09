from __future__ import annotations

from typing import Any


FP32_PROTOCOL = "jmp_l_fast32_fp32_v1"
FP32_MAX_EPOCHS = 32
FP32_MAX_TIME_DAYS = 3
FP32_TASK_SETTINGS: dict[str, dict[str, Any]] = {
    "mp_gap": {
        "batch_size": 16,
        "precision": "32-true",
        "reduction": "mean",
        "num_workers": 8,
    },
    "mp_e_form": {
        "batch_size": 32,
        "precision": "32-true",
        "reduction": "mean",
        "num_workers": 0,
    },
}


def activate_fp32_protocol() -> None:
    """Activate the single FP32 Fast-32 recipe in this process."""

    from hpsafe_sota.experts import jmp_l_bridge

    jmp_l_bridge.FAST32_PROTOCOL = FP32_PROTOCOL
    jmp_l_bridge.FAST32_MAX_EPOCHS = FP32_MAX_EPOCHS
    jmp_l_bridge.FAST32_MAX_TIME_DAYS = FP32_MAX_TIME_DAYS
    jmp_l_bridge.FAST32_TASK_SETTINGS.clear()
    jmp_l_bridge.FAST32_TASK_SETTINGS.update(
        {task: dict(settings) for task, settings in FP32_TASK_SETTINGS.items()}
    )
