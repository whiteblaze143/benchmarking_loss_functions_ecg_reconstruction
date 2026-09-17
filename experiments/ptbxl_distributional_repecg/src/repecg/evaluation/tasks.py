from __future__ import annotations

import json
from pathlib import Path


def load_dataset_tasks(path: Path) -> dict[str, dict[str, object]]:
    datasets = json.loads(path.read_text())
    required = {"task_id", "task_type", "split_contract"}
    task_ids: set[str] = set()
    for dataset, specification in datasets.items():
        tasks = specification.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"dataset {dataset!r} must declare at least one task")
        for task in tasks:
            missing = required - set(task)
            if missing:
                raise ValueError(f"dataset {dataset!r} task is missing fields: {sorted(missing)}")
            task_id = str(task["task_id"])
            if task_id in task_ids:
                raise ValueError(f"duplicate task_id: {task_id}")
            task_ids.add(task_id)
            if "labels" not in task and "label_source" not in task:
                raise ValueError(f"task {task_id!r} must declare labels or label_source")
            labels = task.get("labels", [])
            if not isinstance(labels, list) or len(labels) != len(set(labels)):
                raise ValueError(f"task {task_id!r} labels must be a unique list")
    return datasets
