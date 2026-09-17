"""Retired legacy entry point.

External datasets do not share PTB-XL's five-label ontology. Production
representations must be built by dataset-specific pipelines that preserve the
native task declared in ``configs/dataset_tasks.json``.
"""

raise RuntimeError(
    "retired unsafe OOD builder: use dataset-specific native-task builders; "
    "fabricated PTB-XL labels are forbidden"
)
