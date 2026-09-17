from __future__ import annotations

import pytest
import torch

from repecg.common.models import (
    PhaseCNN,
    RecurrenceCNN,
    PathSignatureClassifier,
    HankelDynamicsModel,
    KoopmanOperatorModel,
    ConditionalRepStatModel,
    OperatorReconstructionAuxiliary,
    LocalTokenCrossAttention,
    CounterfactualMeasurementOperator,
    InterventionalRepStatModel,
    CausalStateECGModel,
    StructuralInnovationModel,
    CounterfactualSurgeryModel,
    InvariantMechanismDiscoveryModel,
    CausalMechanismFactorizationModel,
)


def test_all_15_models_forward_and_backward():
    torch.manual_seed(42)
    B, T, D, classes = 4, 16, 128, 5
    x = torch.randn(B, T, D, requires_grad=True)
    labels = torch.randint(0, 2, (B, classes)).float()

    models = {
        "Paper01_RecurrenceCNN": (RecurrenceCNN(classes=classes), torch.cdist(x, x, p=2).pow(2)),
        "Paper02_PhaseCNN": (PhaseCNN(input_dim=D, classes=classes), x),
        "Paper03_PathSignatureClassifier": (PathSignatureClassifier(input_dim=D, classes=classes), x),
        "Paper04_HankelDynamicsModel": (HankelDynamicsModel(input_dim=D, classes=classes), x),
        "Paper05_KoopmanOperatorModel": (KoopmanOperatorModel(input_dim=D, classes=classes), x),
        "Paper06_ConditionalRepStatModel": (ConditionalRepStatModel(input_dim=D, classes=classes), x),
        "Paper07_OperatorReconstructionAuxiliary": (OperatorReconstructionAuxiliary(input_dim=D, classes=classes), x),
        "Paper08_LocalTokenCrossAttention": (LocalTokenCrossAttention(input_dim=D, classes=classes), x),
        "Paper09_CounterfactualMeasurementOperator": (CounterfactualMeasurementOperator(input_dim=D, classes=classes), x),
        "Paper10_InterventionalRepStatModel": (InterventionalRepStatModel(input_dim=D, classes=classes), x),
        "Paper11_CausalStateECGModel": (CausalStateECGModel(input_dim=D, classes=classes), x),
        "Paper12_StructuralInnovationModel": (StructuralInnovationModel(input_dim=D, classes=classes), x),
        "Paper13_CounterfactualSurgeryModel": (CounterfactualSurgeryModel(input_dim=D, classes=classes), x),
        "Paper14_InvariantMechanismDiscoveryModel": (InvariantMechanismDiscoveryModel(input_dim=D, classes=classes), x),
        "Paper15_CausalMechanismFactorizationModel": (CausalMechanismFactorizationModel(input_dim=D, classes=classes), x),
    }

    for name, (model, inp) in models.items():
        model.train()
        logits = model(inp)
        assert logits.shape == (B, classes), f"{name} expected shape ({B}, {classes}), got {logits.shape}"
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward(retain_graph=True)
        print(f"[PASSED] {name}: forward shape={logits.shape}, loss={loss.item():.4f}")


if __name__ == "__main__":
    test_all_15_models_forward_and_backward()
