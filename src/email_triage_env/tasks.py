from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import TaskDefinition


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "inbox_cases.json"
DEFAULT_TASK_ID = "easy_priority_routing"


def _load_tasks() -> Dict[str, TaskDefinition]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    task_map: Dict[str, TaskDefinition] = {}
    for task_payload in raw.get("tasks", []):
        task = TaskDefinition.model_validate(task_payload)
        task_map[task.id] = task

    if DEFAULT_TASK_ID not in task_map:
        raise ValueError(f"Missing default task: {DEFAULT_TASK_ID}")

    return task_map


_TASKS = _load_tasks()


def list_tasks() -> List[TaskDefinition]:
    return list(_TASKS.values())


def get_task(task_id: str) -> TaskDefinition:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        known = ", ".join(sorted(_TASKS.keys()))
        raise KeyError(f"Unknown task '{task_id}'. Available: {known}") from exc
