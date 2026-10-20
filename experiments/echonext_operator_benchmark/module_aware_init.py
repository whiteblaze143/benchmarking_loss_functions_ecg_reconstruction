#!/usr/bin/env python3
"""Module-aware deterministic initialization for SetOperator architectures (Paper 07).

Enforces:
1. Decoupled module-family RNG streams: Changing operator encoder dimensions or geometry
   features does NOT perturb the initialization draw of response, set_encoder, pool,
   latent, patient_context, or head.
2. LayerNorm weights initialize strictly to 1.0 (scale = 1.0) and biases to 0.0.
3. Linear layers initialize via Xavier uniform with zero bias.
4. MultiheadAttention layers initialize in_proj and out_proj via Xavier uniform with zero bias.
5. Conv1d layers initialize via Kaiming uniform with fan-in uniform bias.
6. Pool query initializes to zero.
7. SHA-256 parameter hashing across models for exact bitwise verification.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Any, Tuple
import torch
from torch import nn


def init_module_aware_deterministic(model: nn.Module, seed: int = 42) -> None:
    """Apply module-aware deterministic initialization with decoupled module-family RNG streams."""
    g_response = torch.Generator().manual_seed(seed + 101)
    g_set_encoder = torch.Generator().manual_seed(seed + 202)
    g_pool = torch.Generator().manual_seed(seed + 303)
    g_latent = torch.Generator().manual_seed(seed + 404)
    g_patient = torch.Generator().manual_seed(seed + 505)
    g_head = torch.Generator().manual_seed(seed + 606)
    g_operator = torch.Generator().manual_seed(seed + 707)

    def _init_layer(m: nn.Module, gen: torch.Generator):
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), generator=gen)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound, generator=gen)
        elif isinstance(m, nn.MultiheadAttention):
            if m._qkv_same_embed_dim and m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight, generator=gen)
            else:
                if m.q_proj_weight is not None:
                    nn.init.xavier_uniform_(m.q_proj_weight, generator=gen)
                if m.k_proj_weight is not None:
                    nn.init.xavier_uniform_(m.k_proj_weight, generator=gen)
                if m.v_proj_weight is not None:
                    nn.init.xavier_uniform_(m.v_proj_weight, generator=gen)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            if m.out_proj is not None:
                nn.init.xavier_uniform_(m.out_proj.weight, generator=gen)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, generator=gen)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_family(sub_mod: nn.Module, gen: torch.Generator):
        for name, m in sub_mod.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.Conv1d, nn.Linear, nn.MultiheadAttention)):
                # Skip out_proj if handled by MultiheadAttention
                parts = name.split(".")
                if len(parts) >= 1 and parts[-1] == "out_proj":
                    continue
                _init_layer(m, gen)

    # Core modules (works with both AblationSetOperatorEchoNext and SetOperatorEchoNext.core)
    core = getattr(model, "core", model)

    if hasattr(core, "response") and core.response is not None:
        _init_family(core.response, g_response)
    if hasattr(core, "operator") and core.operator is not None:
        _init_family(core.operator, g_operator)
    if hasattr(core, "set_encoder") and core.set_encoder is not None:
        _init_family(core.set_encoder, g_set_encoder)
    if hasattr(core, "pool") and core.pool is not None:
        _init_family(core.pool, g_pool)
    if hasattr(core, "pool_query") and core.pool_query is not None:
        nn.init.zeros_(core.pool_query)
    if hasattr(core, "latent") and core.latent is not None:
        _init_family(core.latent, g_latent)
    if hasattr(core, "patient_context") and core.patient_context is not None:
        _init_family(core.patient_context, g_patient)
    if hasattr(core, "head") and core.head is not None:
        _init_family(core.head, g_head)


def get_model_parameter_hashes(model: nn.Module) -> Dict[str, str]:
    """Compute bitwise SHA-256 hash for every named parameter tensor."""
    hashes = {}
    for name, p in model.named_parameters():
        data_bytes = p.detach().cpu().numpy().tobytes()
        hashes[name] = hashlib.sha256(data_bytes).hexdigest()
    return hashes


def verify_layernorm_ones(model: nn.Module) -> Tuple[bool, list[str]]:
    """Verify that all LayerNorm weights in the model are strictly 1.0 and biases 0.0."""
    violations = []
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                max_dev = float((m.weight - 1.0).abs().max().item())
                if max_dev > 1e-6:
                    violations.append(f"{name}.weight max dev from 1.0: {max_dev:.2e}")
            if m.bias is not None:
                max_bias = float(m.bias.abs().max().item())
                if max_bias > 1e-6:
                    violations.append(f"{name}.bias max dev from 0.0: {max_bias:.2e}")
    return len(violations) == 0, violations
