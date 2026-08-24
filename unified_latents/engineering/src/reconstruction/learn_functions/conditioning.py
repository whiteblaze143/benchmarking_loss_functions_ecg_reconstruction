"""Conditioning embedding module for metadata-driven FiLM modulation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


class ConditioningEmbedding(nn.Module):
    """Convert metadata dictionaries into dense conditioning embeddings.

    Expected metadata entries (batch-first tensors):
        - Continuous raw values: e.g., ``age_years`` (float tensor)
        - Normalized values: optionally ``age_years_norm`` (float tensor)
        - Missingness flags: e.g., ``is_missing_age_years`` (float tensor in {0,1})
        - Categorical codes: ``sex_code`` (int tensor), ``pacemaker_status`` (int tensor)

    If normalized values are absent, the module will normalize raw inputs using
    statistics loaded from ``scaling_params.json`` (means/stds). Missing values
    default to zero with an explicit mask appended to the conditioning vector.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        continuous_keys: Optional[Iterable[str]] = None,
        categorical_keys: Optional[Iterable[str]] = None,
        scaling_params_path: Optional[Path] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.continuous_keys = list(continuous_keys or [])
        self.categorical_keys = list(categorical_keys or ["sex_code", "pacemaker_status"])

        self.register_buffer("_means", torch.zeros(1), persistent=False)
        self.register_buffer("_stds", torch.ones(1), persistent=False)
        self._continuous_stats: Dict[str, Dict[str, float]] = {}

        if scaling_params_path is not None:
            self._load_scaling_params(scaling_params_path)
        elif not self.continuous_keys:
            raise ValueError("continuous_keys must be provided when scaling_params_path is None")

        if not self.continuous_keys and self._continuous_stats:
            self.continuous_keys = list(self._continuous_stats.keys())

        self.age_bucket_embed = nn.Embedding(121, 8, padding_idx=0)
        self.sex_embed = nn.Embedding(3, 4, padding_idx=0)
        self.pacemaker_embed = nn.Embedding(3, 4, padding_idx=0)

        categorical_output_dim = 8 + 4 + 4  # age bucket + sex + pacemaker
        self.mlp_input_dim = len(self.continuous_keys) + len(self.continuous_keys)  # values + masks
        self.mlp_input_dim += categorical_output_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

        self._reset_parameters()

    def _load_scaling_params(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        continuous = data.get("continuous", {})
        if not continuous:
            raise ValueError(f"No continuous stats found in scaling params: {path}")
        self._continuous_stats = continuous
        self.continuous_keys = list(continuous.keys())

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.age_bucket_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.sex_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pacemaker_embed.weight, mean=0.0, std=0.02)

        nn.init.kaiming_uniform_(self.mlp[0].weight, a=math.sqrt(5))  # type: ignore[name-defined]
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        nn.init.kaiming_uniform_(self.mlp[3].weight, a=math.sqrt(5))  # type: ignore[name-defined]
        if self.mlp[3].bias is not None:
            nn.init.zeros_(self.mlp[3].bias)

    def forward(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = self._infer_batch(metadata)
        device = next(self.parameters()).device

        continuous_features: List[torch.Tensor] = []
        missing_masks: List[torch.Tensor] = []

        for key in self.continuous_keys:
            tensor = self._extract_continuous(metadata, key, batch_size, device)
            mask = self._extract_missing_mask(metadata, key, batch_size, device)
            continuous_features.append(tensor)
            missing_masks.append(mask)

        cont_stack = torch.stack(continuous_features, dim=1) if continuous_features else torch.zeros(
            batch_size, 0, device=device
        )
        mask_stack = torch.stack(missing_masks, dim=1) if missing_masks else torch.zeros(
            batch_size, 0, device=device
        )

        categorical_embeddings = [
            self._embed_age(metadata, batch_size, device),
            self._embed_sex(metadata, batch_size, device),
            self._embed_pacemaker(metadata, batch_size, device),
        ]

        # Concatenate all embeddings
        cat_stack = torch.cat(categorical_embeddings, dim=1) if categorical_embeddings else torch.zeros(
            batch_size, 0, device=device
        )

        inputs = torch.cat([cont_stack, mask_stack, cat_stack], dim=1)
        return self.mlp(inputs)

    def _extract_continuous(
        self,
        metadata: Dict[str, torch.Tensor],
        key: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        norm_key = f"{key}_norm"
        if norm_key in metadata:
            tensor = metadata[norm_key].to(device)
            return tensor.view(batch_size, -1).squeeze(-1)

        if key in metadata:
            tensor = metadata[key].to(device).float()
        else:
            tensor = torch.zeros(batch_size, device=device)

        stats = self._continuous_stats.get(key, {"mean": 0.0, "std": 1.0})
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0) or 1.0
        tensor = (tensor - mean) / std
        return tensor.view(batch_size, -1).squeeze(-1)

    @staticmethod
    def _extract_missing_mask(
        metadata: Dict[str, torch.Tensor],
        key: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask_key = f"is_missing_{key}"
        if mask_key in metadata:
            mask = metadata[mask_key].to(device).float()
        else:
            mask = torch.zeros(batch_size, device=device)
        return mask.view(batch_size, -1).squeeze(-1)

    def _embed_age(self, metadata: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        age_key = "age_years"
        bucket = torch.zeros(batch_size, dtype=torch.long, device=device)
        if f"{age_key}_bucket" in metadata:
            bucket = metadata[f"{age_key}_bucket"].to(device).long()
        elif age_key in metadata:
            age = metadata[age_key].to(device).float()
            bucket = age.clamp_(0, 120).round().long()
            missing = metadata.get(f"is_missing_{age_key}")
            if missing is not None:
                bucket = torch.where(missing.to(device) > 0.5, torch.zeros_like(bucket), bucket)
        
        if bucket.dim() > 1:
            bucket = bucket.squeeze(-1)
        return self.age_bucket_embed(bucket)

    def _embed_sex(self, metadata: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        sex = metadata.get("sex_code")
        if sex is None:
            sex_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            sex_tensor = sex.to(device).long().clamp_(0, 2)
            
        if sex_tensor.dim() > 1:
            sex_tensor = sex_tensor.squeeze(-1)
        return self.sex_embed(sex_tensor)

    def _embed_pacemaker(self, metadata: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        pm = metadata.get("pacemaker_status")
        if pm is None:
            pm_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            pm_tensor = pm.to(device).long().clamp_(0, 2)
            
        if pm_tensor.dim() > 1:
            pm_tensor = pm_tensor.squeeze(-1)
        return self.pacemaker_embed(pm_tensor)



    @staticmethod
    def _infer_batch(metadata: Dict[str, torch.Tensor]) -> int:
        for value in metadata.values():
            if torch.is_tensor(value):
                return value.shape[0]
        raise ValueError("Unable to infer batch size from empty metadata dictionary")

