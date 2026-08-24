"""Shared lead-regime definitions for engineering reconstruction experiments."""

from __future__ import annotations

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAME_TO_INDEX = {name: idx for idx, name in enumerate(LEAD_NAMES)}

REGIMES = {
    "current": ["I", "II", "V2"],
    "wearecg": ["II", "V1", "V5"],
    "historical": ["I", "II", "V3"],
}


def parse_obs_leads(obs_leads_arg: str) -> list[int]:
    try:
        names = [name.strip() for name in obs_leads_arg.split(",") if name.strip()]
        indices = [LEAD_NAME_TO_INDEX[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"Unknown lead name: {exc.args[0]}.") from exc
    if not indices:
        raise ValueError("--obs_leads must specify at least one lead.")
    if len(indices) > 3:
        raise ValueError("Engineering pipelines support at most 3 observed leads.")
    if len(indices) != len(set(indices)):
        raise ValueError("--obs_leads contains duplicates.")
    return indices


def resolve_obs_leads(regime: str | None, obs_leads_arg: str | None) -> list[int]:
    if obs_leads_arg:
        return parse_obs_leads(obs_leads_arg)
    chosen_regime = regime or "current"
    if chosen_regime not in REGIMES:
        raise ValueError(f"Unknown regime: {chosen_regime}. Expected one of {sorted(REGIMES)}.")
    return [LEAD_NAME_TO_INDEX[name] for name in REGIMES[chosen_regime]]


def get_missing_indices(obs_indices: list[int]) -> list[int]:
    return [idx for idx in range(len(LEAD_NAMES)) if idx not in obs_indices]


def format_lead_set(obs_indices: list[int]) -> str:
    return "-".join(LEAD_NAMES[idx] for idx in obs_indices)


def make_lead_indices(obs_indices: list[int], batch_size: int, device) -> object:
    import torch

    return torch.tensor([obs_indices], device=device).expand(batch_size, -1)
