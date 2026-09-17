from __future__ import annotations

from torch import nn

from .models import (
    CausalMechanismFactorizationModel,
    CausalStateECGModel,
    ConditionalRepStatModel,
    CounterfactualMeasurementOperator,
    CounterfactualSurgeryModel,
    HankelDynamicsModel,
    InterventionalRepStatModel,
    InvariantMechanismDiscoveryModel,
    KoopmanOperatorModel,
    LocalTokenCrossAttention,
    OperatorReconstructionAuxiliary,
    PathSignatureClassifier,
    PhaseCNN,
    RecurrenceCNN,
    StructuralInnovationModel,
)
from .variants import ExperimentVariant


def create_paper_model(
    paper_id: int,
    input_dim: int,
    classes: int,
    variant: ExperimentVariant,
) -> nn.Module:
    constructors = {
        1: lambda: RecurrenceCNN(classes=classes, variant=variant),
        2: lambda: PhaseCNN(input_dim=input_dim, classes=classes, variant=variant),
        3: lambda: PathSignatureClassifier(input_dim=input_dim, classes=classes, variant=variant),
        4: lambda: HankelDynamicsModel(input_dim=input_dim, classes=classes, variant=variant),
        5: lambda: KoopmanOperatorModel(input_dim=input_dim, classes=classes, variant=variant),
        6: lambda: ConditionalRepStatModel(input_dim=input_dim, classes=classes, variant=variant),
        7: lambda: OperatorReconstructionAuxiliary(input_dim=input_dim, classes=classes, variant=variant),
        8: lambda: LocalTokenCrossAttention(input_dim=input_dim, classes=classes, variant=variant),
        9: lambda: CounterfactualMeasurementOperator(input_dim=input_dim, classes=classes, variant=variant),
        10: lambda: InterventionalRepStatModel(input_dim=input_dim, classes=classes, variant=variant),
        11: lambda: CausalStateECGModel(input_dim=input_dim, classes=classes, variant=variant),
        12: lambda: StructuralInnovationModel(input_dim=input_dim, classes=classes, variant=variant),
        13: lambda: CounterfactualSurgeryModel(input_dim=input_dim, classes=classes, variant=variant),
        14: lambda: InvariantMechanismDiscoveryModel(input_dim=input_dim, classes=classes, variant=variant),
        15: lambda: CausalMechanismFactorizationModel(input_dim=input_dim, classes=classes, variant=variant),
    }
    try:
        return constructors[paper_id]()
    except KeyError as error:
        raise ValueError(f"paper_id must be in 1..15, got {paper_id}") from error
