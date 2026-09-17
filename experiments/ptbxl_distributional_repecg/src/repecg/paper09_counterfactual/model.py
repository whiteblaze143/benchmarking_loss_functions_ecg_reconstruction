from __future__ import annotations

import torch
from torch import nn

from repecg.paper07_operator import OperatorSetModel


def mismatch_operators(operators: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Destroy q-response pairing while leaving every response unchanged."""
    if operators.ndim != 3 or permutation.shape != operators.shape[:2]:
        raise ValueError("expected operators (batch,set,8) and matching permutation")
    rows = torch.arange(len(operators), device=operators.device).unsqueeze(1)
    return operators[rows, permutation]


class CounterfactualOperatorSetModel(nn.Module):
    """Infer a record state from context pairs and synthesize F(q*)."""

    def __init__(self, response_dim: int = 128, classes: int = 5):
        super().__init__()
        self.backbone = OperatorSetModel(
            response_dim=response_dim, classes=classes, operator_mode="continuous"
        )

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        *,
        target_operator: torch.Tensor | None = None,
        return_counterfactual: bool = False,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        state = self.backbone.encode_context(operators, responses)
        logits = self.backbone.head(state)
        outputs: list[torch.Tensor] = [logits]
        if return_counterfactual:
            if target_operator is None or target_operator.ndim != 2:
                raise ValueError("counterfactual synthesis requires one target operator per record")
            target = self.backbone.encode_operator(target_operator, None)
            prediction = self.backbone.decoder(torch.cat((state, target), dim=-1)).reshape(
                len(state), 16, responses.shape[-1]
            )
            outputs.append(prediction)
        if return_state:
            outputs.append(state)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
