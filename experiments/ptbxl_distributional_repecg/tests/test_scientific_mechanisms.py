import torch
import pytest
from repecg.common.variants import get_variants_for_paper
from repecg.common.models import (
    RecurrenceCNN, PhaseCNN, HankelDynamicsModel,
    KoopmanOperatorModel, ConditionalRepStatModel, OperatorReconstructionAuxiliary,
    LocalTokenCrossAttention, CounterfactualMeasurementOperator, InterventionalRepStatModel,
    CausalStateECGModel, StructuralInnovationModel, CounterfactualSurgeryModel,
    InvariantMechanismDiscoveryModel, CausalMechanismFactorizationModel
)

def test_paper01_mechanism():
    variants = get_variants_for_paper(1)
    # The full model and mechanism-destroying control should both work and output correct shapes
    x = torch.randn(2, 16, 16) # batch, G, G
    
    full = RecurrenceCNN(variant=variants["full"])
    assert full(x).shape == (2, 5)
    
    control = RecurrenceCNN(variant=variants["mean_distance_recurrence"])
    x_control = torch.randn(2, 16, 16)
    assert control(x_control).shape == (2, 5)

def test_paper02_mechanism():
    variants = get_variants_for_paper(2)
    x = torch.randn(2, 16, 8) # batch, phase, dim
    
    full = PhaseCNN(8, variant=variants["full"])
    assert full(x).shape == (2, 5)
    
    control = PhaseCNN(8, variant=variants["no_circular"])
    assert control(x).shape == (2, 5)

def test_paper03_mechanism():
    variants = get_variants_for_paper(3)
    x = torch.randn(2, 16, 8)
    
    full = PhaseCNN(8, variant=variants["full"])
    assert full(x).shape == (2, 5)
    
    control = PhaseCNN(8, variant=variants["unordered_kme"])
    assert control(x).shape == (2, 5)
    assert variants["order_scrambled"].representation == "order_scrambled"
    assert variants["monotone_warp_sham"].representation == "monotone_warp_sham"

def test_paper04_mechanism():
    variants = get_variants_for_paper(4)
    x = torch.randn(2, 16, 8)
    
    full = HankelDynamicsModel(8, lag=4, variant=variants["full"])
    assert full(x).shape == (2, 5)
    
    control = HankelDynamicsModel(8, lag=1, variant=variants["time_shuffled"])
    assert control(x).shape == (2, 5)

def test_paper05_mechanism():
    variants = get_variants_for_paper(5)
    x = torch.randn(2, 164)
    
    full = KoopmanOperatorModel(164, variant=variants["full"])
    assert full(x).shape == (2, 5)
    
    control = KoopmanOperatorModel(164, variant=variants["occupancy_only"])
    assert control(x).shape == (2, 5)

def test_paper08_mechanism():
    variants = get_variants_for_paper(8)
    x = torch.randn(2, 16, 8)
    
    full = LocalTokenCrossAttention(8, variant=variants["full"])
    assert full(x).shape == (2, 5)
    
    control = LocalTokenCrossAttention(8, variant=variants["kmeans_tokens"])
    with pytest.raises(RuntimeError, match="fitted train-only codebook"):
        control(x)
    control.set_codebook(torch.randn(64, 8))
    assert control(x).shape == (2, 5)


def test_paper15_shared_control_is_capacity_matched_and_actually_shared():
    variants = get_variants_for_paper(15)
    full = CausalMechanismFactorizationModel(8, variant=variants["full"])
    control = CausalMechanismFactorizationModel(
        8, variant=variants["capacity_matched_shared"]
    )

    full_parameters = sum(parameter.numel() for parameter in full.parameters())
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    assert abs(control_parameters - full_parameters) / full_parameters < 0.01
    assert len({id(module) for module in full.mechanisms}) == 15
    assert len({id(module) for module in control.mechanisms}) == 1

if __name__ == "__main__":
    pytest.main([__file__])
