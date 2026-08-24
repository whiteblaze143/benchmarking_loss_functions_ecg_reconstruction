from scripts.af_protocol import lead_mask, parse_scp_codes, ptbxl_af_label, rdb_af_label
from scripts.af_probe import availability
from scripts.evaluate_af_preservation import assert_identity, build
import numpy as np


def test_dataset_specific_afib_trap():
    assert rdb_af_label("AF") == (1, "AF")
    assert rdb_af_label("AFIB") == (0, "AFL")
    assert rdb_af_label("AFL") == (0, "AFL")
    assert rdb_af_label("SA") == (0, "SI")
    assert rdb_af_label("VT") == (0, "SVT/VT")
    assert ptbxl_af_label({"AFIB"}) == (1, "AF")
    assert ptbxl_af_label({"AFLT"}) == (0, "AFL")
    assert ptbxl_af_label({"AFIB", "AFLT"}) == (None, "conflict_AFIB_AFLT")


def test_five_condition_lead_masks():
    assert lead_mask("real12", 0) == list(range(12))
    assert lead_mask("source", 0) == [0]
    assert lead_mask("real11", 0) == list(range(1, 12))
    assert lead_mask("synthetic11", 1) == [0, *range(2, 12)]


def test_ptbxl_zero_likelihood_statement_is_still_present():
    assert parse_scp_codes("{'AFIB': 0.0, 'NORM': 100.0}") == {"AFIB", "NORM"}
    assert ptbxl_af_label(parse_scp_codes("{'AFIB': 0.0}")) == (1, "AF")

def test_probe_availability_contract():
    assert availability("source_I").tolist() == [1] + [0] * 11
    assert availability("source_II").tolist() == [0, 1] + [0] * 10
    assert int(availability("real11_I").sum()) == 11

def test_hybrid_and_synthetic_identity_contracts():
    real=np.arange(120,dtype=np.float32).reshape(10,12); recon=-real
    hybrid,mask=build(real,recon,1,"D_hybrid12"); assert_identity(real,recon,hybrid,mask,1,"D_hybrid12")
    assert np.array_equal(hybrid[:,1],real[:,1])
    synthetic,mask=build(real,recon,1,"E_synthetic11"); assert_identity(real,recon,synthetic,mask,1,"E_synthetic11")
