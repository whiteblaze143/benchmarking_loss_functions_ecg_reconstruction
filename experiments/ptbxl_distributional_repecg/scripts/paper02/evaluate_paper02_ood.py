"""Retired legacy entry point.

OOD evaluation now requires a dataset-native task head, an explicit checkpoint,
and a reconciled native split. It must not infer a shared target or instantiate
an untrained model when a checkpoint is absent.
"""

raise RuntimeError(
    "retired unsafe OOD evaluator: use checkpoint-backed native-task evaluation"
)
