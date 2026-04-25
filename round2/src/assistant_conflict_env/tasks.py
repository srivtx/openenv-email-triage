from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from .conflict_generator import generate_episode
from .eval_set import (
    ADVERSARIAL_SEEDS,
    HOLDOUT_SEEDS,
    TRAIN_SEEDS,
    adversarial_episodes,
    assert_split_disjoint,
    holdout_episodes,
    train_episodes,
)
from .models import TaskDefinition


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conflict_cases.json"
DEFAULT_TASK_ID = "easy_evening_planner"


# Procedural task ids look like ``proc_<difficulty>_<seed>``. Validated on lookup.
_PROC_ID_RE = re.compile(r"^proc_(easy|medium|hard)_(\d+)$")


def _load_static_tasks() -> Dict[str, TaskDefinition]:
    """Load the legacy fixture-based tasks. Kept for the deployed UI demo."""

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    task_map: Dict[str, TaskDefinition] = {}
    for payload in raw.get("tasks", []):
        task = TaskDefinition.model_validate(payload)
        task_map[task.id] = task

    if DEFAULT_TASK_ID not in task_map:
        raise ValueError(f"Missing default task: {DEFAULT_TASK_ID}")
    return task_map


_STATIC_TASKS = _load_static_tasks()
assert_split_disjoint()


def list_tasks() -> List[TaskDefinition]:
    """Return only the static demo tasks (preserves the existing API surface)."""

    return list(_STATIC_TASKS.values())


def get_task(task_id: str) -> TaskDefinition:
    """Resolve either a static fixture id or a procedural ``proc_<diff>_<seed>`` id."""

    if task_id in _STATIC_TASKS:
        return _STATIC_TASKS[task_id]

    match = _PROC_ID_RE.match(task_id)
    if match:
        difficulty = match.group(1)
        seed = int(match.group(2))
        return generate_episode(seed=seed, difficulty=difficulty)

    known = ", ".join(sorted(_STATIC_TASKS.keys()))
    raise KeyError(
        f"Unknown task '{task_id}'. Static tasks: {known}. "
        f"Procedural tasks must match 'proc_<easy|medium|hard>_<seed>'."
    )


__all__ = [
    "DEFAULT_TASK_ID",
    "ADVERSARIAL_SEEDS",
    "HOLDOUT_SEEDS",
    "TRAIN_SEEDS",
    "adversarial_episodes",
    "assert_split_disjoint",
    "get_task",
    "holdout_episodes",
    "list_tasks",
    "train_episodes",
]
