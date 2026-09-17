from __future__ import annotations

import pytest
import torch
from torch import nn

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


def get_model(name: str, D: int, classes: int, ablation: str) -> nn.Module:
    if name == "Paper01_RecurrenceCNN": return RecurrenceCNN(classes=classes, ablation=ablation)
    if name == "Paper02_PhaseCNN": return PhaseCNN(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper03_PathSignatureClassifier": return PathSignatureClassifier(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper04_HankelDynamicsModel": return HankelDynamicsModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper05_KoopmanOperatorModel": return KoopmanOperatorModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper06_ConditionalRepStatModel": return ConditionalRepStatModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper07_OperatorReconstructionAuxiliary": return OperatorReconstructionAuxiliary(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper08_LocalTokenCrossAttention": return LocalTokenCrossAttention(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper09_CounterfactualMeasurementOperator": return CounterfactualMeasurementOperator(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper10_InterventionalRepStatModel": return InterventionalRepStatModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper11_CausalStateECGModel": return CausalStateECGModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper12_StructuralInnovationModel": return StructuralInnovationModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper13_CounterfactualSurgeryModel": return CounterfactualSurgeryModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper14_InvariantMechanismDiscoveryModel": return InvariantMechanismDiscoveryModel(input_dim=D, classes=classes, ablation=ablation)
    if name == "Paper15_CausalMechanismFactorizationModel": return CausalMechanismFactorizationModel(input_dim=D, classes=classes, ablation=ablation)
    raise ValueError(f"Unknown {name}")


def test_all_15_models_forward_and_backward():
    torch.manual_seed(42)
    B, T, D, classes = 4, 16, 128, 5
    x = torch.randn(B, T, D, requires_grad=True)
    labels = torch.randint(0, 2, (B, classes)).float()

    model_names = [
        "Paper01_RecurrenceCNN", "Paper02_PhaseCNN", "Paper03_PathSignatureClassifier",
        "Paper04_HankelDynamicsModel", "Paper05_KoopmanOperatorModel", "Paper06_ConditionalRepStatModel",
        "Paper07_OperatorReconstructionAuxiliary", "Paper08_LocalTokenCrossAttention", "Paper09_CounterfactualMeasurementOperator",
        "Paper10_InterventionalRepStatModel", "Paper11_CausalStateECGModel", "Paper12_StructuralInnovationModel",
        "Paper13_CounterfactualSurgeryModel", "Paper14_InvariantMechanismDiscoveryModel", "Paper15_CausalMechanismFactorizationModel"
    ]
    
    specific_ablations = {
        "Paper01_RecurrenceCNN": "no_recurrence",
        "Paper02_PhaseCNN": "no_circular",
        "Paper03_PathSignatureClassifier": "no_level2",
        "Paper04_HankelDynamicsModel": "lag_1",
        "Paper05_KoopmanOperatorModel": "static_koopman",
        "Paper06_ConditionalRepStatModel": "no_circular",
        "Paper07_OperatorReconstructionAuxiliary": "no_circular",
        "Paper08_LocalTokenCrossAttention": "no_attention",
        "Paper09_CounterfactualMeasurementOperator": "fixed_operator",
        "Paper10_InterventionalRepStatModel": "no_circular",
        "Paper11_CausalStateECGModel": "no_gru",
        "Paper12_StructuralInnovationModel": "no_ar_filter",
        "Paper13_CounterfactualSurgeryModel": "no_circular",
        "Paper14_InvariantMechanismDiscoveryModel": "no_circular",
        "Paper15_CausalMechanismFactorizationModel": "shared_mechanism"
    }

    for name in model_names:
        ablations = ["none", "linear_probe", specific_ablations[name]]
        inp = torch.cdist(x, x, p=2).pow(2) if name == "Paper01_RecurrenceCNN" else x
        for ablation in ablations:
            model = get_model(name, D, classes, ablation)
            model.train()
            if name == "Paper07_OperatorReconstructionAuxiliary":
                logits = model(inp, return_reconstruction=False)
            else:
                logits = model(inp)
            assert logits.shape == (B, classes), f"{name} ({ablation}) expected ({B}, {classes}), got {logits.shape}"
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward(retain_graph=True)
            print(f"[PASSED] {name} (ablation={ablation}): forward shape={logits.shape}, loss={loss.item():.4f}")


if __name__ == "__main__":
    test_all_15_models_forward_and_backward()
