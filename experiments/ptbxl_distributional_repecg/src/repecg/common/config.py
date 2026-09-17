from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProgramConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return self.path.parents[3]

    @property
    def data_root(self) -> Path:
        value = Path(self.raw["data"]["root"])
        return value if value.is_absolute() else self.project_root / value

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.raw["labels"]["classes"])


def load_config(path: str | Path) -> ProgramConfig:
    resolved = Path(path).resolve()
    raw = yaml.safe_load(resolved.read_text())
    if raw["firewall"]["require_freeze_manifest_before_fold10"] is not True:
        raise ValueError("fold-10 firewall must be enabled")
    if raw["labels"]["use_clinical_metadata"] is not False:
        raise ValueError("clinical metadata is prohibited")
    return ProgramConfig(path=resolved, raw=raw)
