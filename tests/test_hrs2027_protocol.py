from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hrs2027_protocol import add_ablation_dates, extract_features, parse_atr_gold_standard_episodes, parse_noise_labels, quality_mask
from prototype.metrics_engine import match_episodes,extract_continuous_episodes


def test_adjacent_af_flutter_are_one_endpoint_episode():
    atr = SimpleNamespace(sample=[0,100,300,600,1000], aux_note=['(N','(AFIB','(AFL','(AT','(N'])
    episodes = parse_atr_gold_standard_episodes(atr,1200,1)
    af = [x for x in episodes if x['is_af']]
    assert len(af)==1
    assert (af[0]['start_sec'],af[0]['end_sec'],af[0]['duration_sec'])==(100,1000,900)


def test_blanking_uses_latest_prior_procedure_and_requires_history():
    meta = pd.DataFrame(dict(Date_Holter=['2025-06-01']*3,
                            Date_1st_AF_Ablation=['2020-01-01']*3,
                            Date_Redo_AF_Ablation=['2025-05-01','2026-01-01',None],
                            Previous_AF_Ablation=[True,True,False]))
    result=add_ablation_dates(meta)
    assert result.post_blanking.tolist()==[False,True,False]
    assert result.days_from_ablation.iloc[0]==31


def test_features_do_not_invent_rhythm_for_constant_signal():
    anchor=np.random.default_rng(42).normal(size=(128,16)).astype(np.float32)
    kme,rr,hrv,sqi,n=extract_features(np.zeros(2000),200,anchor)
    assert n==0
    assert not kme.any() and not rr.any() and not hrv.any()
    assert sqi==0


def test_noise_labels_and_same_joint_quality_policy():
    assert parse_noise_labels(['CLASS:N1 | NOISE:None | EVENTS:A. Flutter'])[1]==1
    assert parse_noise_labels(['CLASS:N2 | NOISE:None | EVENTS:VPB'])[1]==0
    assert quality_mask(np.array([.5,.9,.1]),np.array([1.,.1,1.])).tolist()==[True,True,False]


def test_episode_matching_maximizes_number_of_detections():
    refs=[dict(start_sec=100,end_sec=200),dict(start_sec=300,end_sec=400)]
    preds=[dict(start_sec=150,end_sec=400),dict(start_sec=0,end_sec=120)]
    assert match_episodes(refs,preds)['detected_ref']==2


def test_partial_last_window_cannot_invent_six_minute_episode():
    assert extract_continuous_episodes(np.ones(12,dtype=bool),min_duration_sec=360,record_duration_sec=350)==[]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='A100 feature parity test')
def test_gpu_kme_matches_cpu_including_no_beats():
    from scripts.hrs2027_gpu_features import batch_phase_kme
    rng=np.random.default_rng(42)
    anchor=rng.normal(size=(128,16)).astype(np.float32)
    cells=[rng.normal(size=(n,16,16)).astype(np.float32) for n in [0,3,40]]
    expected=[]
    for c in cells:
        kme=np.zeros((16,128),dtype=np.float32)
        if len(c):
            for g in range(16):
                kme[g]=np.exp(-((c[:,g,None,:]-anchor[None,:,:])**2).sum(-1)/16).mean(0)
            kme/=np.linalg.norm(kme,axis=1,keepdims=True)+1e-8
        expected.append(kme)
    np.testing.assert_allclose(batch_phase_kme(cells,anchor),expected,atol=2e-6,rtol=2e-5)


def test_wear_smoothing_uses_only_available_past_values():
    from scripts.run_wearable_completeness_monte_carlo import causal_average
    values=np.array([.9,np.nan,.1,.8])
    wear=np.array([True,False,True,True])
    np.testing.assert_allclose(causal_average(values,wear,k=2),[.9,np.nan,.1,.45],equal_nan=True)


@pytest.mark.parametrize('length',[200,201,6000,6001])
def test_vectorized_welch_matches_reference(length):
    from scipy import signal
    from scripts.hrs2027_protocol import _welch_power
    ecg=np.random.default_rng(42).normal(size=length).astype(np.float32)
    f,power=_welch_power(ecg)
    expected_f,expected_power=signal.welch(ecg,fs=200,nperseg=min(length,400))
    np.testing.assert_array_equal(f,expected_f)
    np.testing.assert_allclose(power,expected_power,atol=1e-8,rtol=1e-5)
