"""Real CUDA kernel and official-model witnesses for the isolated HRS runtime."""
from pathlib import Path
import time

import numpy as np
import torch
import tensorflow as tf
import wfdb

torch.set_num_threads(2)
torch.manual_seed(42)
assert torch.backends.cudnn.version()==90300
layer=torch.nn.Conv1d(4,8,3).cuda()
inputs=torch.randn(64,4,128,device='cuda')
layer(inputs).sum().backward()
print('TORCH_CONV_WITNESS',tuple(layer(inputs).shape),flush=True)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu,True)
assert tf.config.list_physical_devices('GPU')
with tf.device('/GPU:0'):
    inputs=tf.random.stateless_normal((64,128,4),seed=(42,42))
    result=tf.keras.layers.Conv1D(8,3)(inputs)
    assert bool(tf.reduce_all(tf.math.is_finite(result)))
    print('TF_CONV_WITNESS',result.device,flush=True)
root=Path(__file__).resolve().parents[1]
model=tf.saved_model.load(str(root/'external/AFtoolkit/detection/ArNetECG_exported_model'))
record=wfdb.rdrecord('/data/mithunmanivannan/datasets/shdb_af/001',sampto=64*6000)
raw=record.p_signal[:,0].astype(np.float32).reshape(64,6000)
start=time.perf_counter()
result=model.signatures['predict_fixed'](x=tf.constant(raw),prec_windows=tf.range(64),
    succ_windows=tf.range(63,-1,-1),glob_lab=tf.ones(64),ids=tf.constant(['001']*64))
assert np.isfinite(result['probs'].numpy()).all()
print('OFFICIAL_MODEL_WITNESS',result['probs'].shape,'seconds',time.perf_counter()-start,flush=True)
