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
from repecg.paper07_operator import OperatorSetModel


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

def test_paper07_mechanism():
    responses = torch.randn(2, 4, 16, 128)
    operators = torch.randn(2, 4, 8)
    operator_ids = torch.randint(0, 8, (2, 4))
    target_operator = torch.randn(2, 8)
    target_ids = torch.randint(0, 8, (2,))

    # 1. Continuous Primary
    cont_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    out_cont = cont_model(operators, responses)
    assert out_cont.shape == (2, 5)

    # 2. Categorical Primary
    cat_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="categorical", vocabulary_size=8)
    out_cat = cat_model(None, responses, operator_ids=operator_ids)
    assert out_cat.shape == (2, 5)

    # 3. Continuous Auxiliary (Reconstruction)
    logits_recon, recon = cont_model(
        operators, responses, target_operator=target_operator, return_reconstruction=True
    )
    assert logits_recon.shape == (2, 5)
    assert recon.shape == (2, 16, 128)

    # 4. Categorical Auxiliary (Reconstruction)
    logits_cat_recon, cat_recon = cat_model(
        None, responses, operator_ids=operator_ids, target_operator_ids=target_ids, return_reconstruction=True
    )
    assert logits_cat_recon.shape == (2, 5)
    assert cat_recon.shape == (2, 16, 128)


def test_paper08_mechanism():
    variants = get_variants_for_paper(8)
    x = torch.randn(2, 16, 8)
    
    full = LocalTokenCrossAttention(8, variant=variants["global_dynamic"])
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
    assert len({id(module) for module in full.mechanisms}) == 16
    assert len({id(module) for module in control.mechanisms}) == 1

if __name__ == "__main__":
    pytest.main([__file__])
