"""Validation-only frozen-model variation across declared channel patterns; no labels."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from set_waveform_model import AFSetOperator
from train_raw_waveform import setup

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def mean_sd(x):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    return float(x.std(axis=0).mean())


def main():
    rows = list(csv.DictReader((W/'file_manifest.csv').open()))
    selected = [];patients = set()
    for row in rows:
        if row['partition'] == 'validation' and row['patient_id'] not in patients:
            selected.append(row);patients.add(row['patient_id'])
            if len(selected) == 4: break
    assert len(selected) == 4
    responses=[];masks=[];indices=[];files=[]
    for row in selected:
        path=W/'phase_features_v1/shards'/f"file_{int(row['file_index']):03d}.npz"
        meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['output_sha256']
        with np.load(path) as z:
            good=np.flatnonzero(z['phase_valid'].all(axis=1));assert len(good)>=64
            chosen=good[np.linspace(0,len(good)-1,64,dtype=int)]
            responses.append(z['response'][chosen]);masks.append(z['phase_valid'][chosen]);indices.extend(z['window_index'][chosen].tolist())
        files.append({'file_index':int(row['file_index']),'patient_id':row['patient_id'],'sha256':sha(path)})
    values=np.concatenate(responses);valid=np.concatenate(masks);ids=np.array(indices);assert valid.all() and values.shape==(256,2,16,128)
    setup();x=torch.from_numpy(values).cuda();v=torch.from_numpy(valid).cuda();operator=torch.tensor([0,1],device='cuda').expand(len(x),-1)
    result={'partition':'validation','test_access':False,'labels_read':False,'windows':len(ids),'files':files,
        'sampling':'64 evenly spaced both-channel-valid windows for each of the first four distinct validation patients; not representative.',
        'models':{},'scope':'Observational frozen-checkpoint activation variation; hooks can change numerical execution, no causal or clinical claim.'}
    for mode in ['categorical','q_ablated']:
        root=W/'set_models_math_sdp_v2'/mode;lock=json.loads((root/'evaluation_lock.json').read_text());assert sha(root/'best.pt')==lock['best_checkpoint_sha256']
        model=AFSetOperator(mode).cuda().eval();model.load_state_dict(torch.load(root/'best.pt',weights_only=True)['state_dict'])
        with np.load(root/'development_manifest.npz') as z: reference=z['window_index'][504777:]
        lookup=np.searchsorted(reference,ids);assert np.array_equal(reference[lookup],ids)
        saved=np.load(root/'best_validation_logits.npy')[lookup]
        result['models'][mode]={}
        for j,pattern in enumerate([[True,False],[False,True],[True,True]]):
            availability=torch.tensor(pattern,device='cuda').expand(len(x),-1);layers=[[],[]];nested_modes=[[],[]]
            def make_hook(i):
                def hook(module,args,out):
                    nested_modes[i].append(bool(out.is_nested))
                    if out.is_nested:
                        out=out.to_padded_tensor(0.)[:, :sum(pattern)]
                    else:
                        out=out[:, pattern]
                    layers[i].append(out.detach().cpu().numpy())
                return hook
            handles=[layer.register_forward_hook(make_hook(i)) for i,layer in enumerate(model.original.set_encoder.layers)]
            with sdpa_kernel(SDPBackend.MATH),torch.inference_mode():
                latent=model.encode(x,availability,v,operator);logits=model.original.head(latent).squeeze(-1).cpu().numpy()
            for handle in handles:handle.remove()
            maxdiff=float(np.abs(logits-saved[:,j]).max());assert maxdiff<1e-4
            result['models'][mode][str(j)]={'channel_pattern':pattern,'nested_layer_outputs':[m[0] for m in nested_modes],
                'first_layer_packed_active_tokens_mean_coordinate_sd':mean_sd(layers[0][0][:,:sum(pattern)]),
                'second_layer_packed_active_tokens_mean_coordinate_sd':mean_sd(layers[1][0][:,:sum(pattern)]),
                'latent_mean_coordinate_sd':mean_sd(latent.cpu().numpy()),
                'logit_range':[float(logits.min()),float(logits.max())],
                'max_absolute_logit_difference_from_saved':maxdiff}
    result['source_sha256']=sha(Path(__file__))
    out=W/'pre_release_corrective_v1/validation_activation_patterns_v2.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
