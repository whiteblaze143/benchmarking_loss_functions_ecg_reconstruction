"""Extract every declared local channel with committed per-file resumable shards."""
import argparse,csv,hashlib,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import h5py,numpy as np
from observed_waveform import extract_channel
from waveform_protocol import W,sha
_DATA=None
_FREEZE=None

def initialize():
    global _DATA,_FREEZE
    _FREEZE=json.loads((W/'preprocessing_freeze.json').read_text())
    for name,digest in _FREEZE['source_hashes'].items():assert sha(name)==digest,name
    assert sha(W/'input_contract.json')==_FREEZE['input_contract_sha256']
    with np.load(W/'windows.npz') as z:_DATA={k:z[k] for k in z.files}

def process_file(row):
    fi=int(row['file_index']);directory=W/'observed_features';directory.mkdir(exist_ok=True)
    prefix=directory/f'file_{fi:03d}';dest=prefix.with_suffix('.npz');receipt_path=prefix.with_suffix('.json');errors_path=prefix.with_suffix('.errors.json')
    freeze_sha=sha(W/'preprocessing_freeze.json')
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        assert receipt['preprocessing_freeze_sha256']==freeze_sha
        assert receipt['h5_sha256']==row['h5_sha256'] and receipt['file_index']==fi
        assert sha(dest)==receipt['output_sha256'] and sha(errors_path)==receipt['errors_sha256']
        return {**receipt,'reused_verified_shard':True}
    assert not dest.exists(),f'Orphan uncommitted shard: {dest}'
    assert sha(row['h5_path'])==row['h5_sha256'],row['h5_path']
    with h5py.File(row['h5_path']) as h:voltage=h['ecg'].astype('float32')[:]
    assert voltage.shape==(int(row['source_samples']),2)
    indices=np.flatnonzero(_DATA['file_index']==fi);starts=_DATA['input_start'][indices];n=len(indices)
    out={'window_index':indices.astype(np.int32),'rr_features':np.full((n,2,16),np.nan,dtype=np.float32),'rr_valid':np.zeros((n,2),bool),'phase_valid':np.zeros((n,2),bool),'accepted_intervals':np.zeros((n,2),np.uint16),'represented_fraction':np.zeros((n,2),np.float32),'detector_status':np.zeros((n,2),np.int8),'peak_counts':np.zeros((n,2),np.uint16)}
    peaks=[[],[]];offsets=[[0],[0]];errors=[];started=time.time()
    for j,start in enumerate(starts):
        window=voltage[start:start+6000]
        for channel in [0,1]:
            r=extract_channel(window[:,channel])
            for key in ['rr_features','rr_valid','phase_valid','accepted_intervals','represented_fraction']:out[key][j,channel]=r[key]
            out['detector_status'][j,channel]=r['status'];out['peak_counts'][j,channel]=len(r['peaks'])
            peaks[channel].append(r['peaks'].astype(np.int16));offsets[channel].append(offsets[channel][-1]+len(r['peaks']))
            if r['error']:errors.append({'window_index':int(indices[j]),'channel':channel,'status':r['status'],'error':r['error']})
    for channel in [0,1]:
        out[f'peak_offsets_{channel}']=np.asarray(offsets[channel],dtype=np.int64)
        out[f'peak_samples_{channel}']=np.concatenate(peaks[channel])
    tmp=dest.with_suffix('.npz.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**out)
    os.replace(tmp,dest)
    errors_path.write_text(json.dumps(errors)+'\n')
    receipt={'file_index':fi,'record_id':row['record_id'],'daily_file_index':int(row['daily_file_index']),'partition':row['partition'],'windows':n,'phase_valid_by_channel':out['phase_valid'].sum(axis=0).astype(int).tolist(),'rr_valid_by_channel':out['rr_valid'].sum(axis=0).astype(int).tolist(),'both_phase_unavailable_windows':int((~out['phase_valid'].any(axis=1)).sum()),'detector_or_input_error_channels':len(errors),'h5_sha256':row['h5_sha256'],'preprocessing_freeze_sha256':freeze_sha,'output_sha256':sha(dest),'errors_sha256':sha(errors_path),'seconds':time.time()-started}
    temp=receipt_path.with_suffix('.json.partial');temp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(temp,receipt_path)
    return receipt

def smoke():
    rows=list(csv.DictReader((W/'file_manifest.csv').open()));row=next(r for r in rows if r['partition']=='train')
    with np.load(W/'windows.npz') as z:indices=np.flatnonzero(z['file_index']==int(row['file_index']))[:8];starts=z['input_start'][indices]
    out=[];started=time.time()
    with h5py.File(row['h5_path']) as h:
        for i,start in zip(indices,starts):
            voltage=h['ecg'][start:start+6000].astype(np.float32)
            for channel in [0,1]:
                r=extract_channel(voltage[:,channel]);out.append({'window_index':int(i),'channel':channel,'peaks':len(r['peaks']),'rr_valid':r['rr_valid'],'phase_valid':r['phase_valid'],'status':r['status']})
    assert len(out)==16 and any(r['rr_valid'] for r in out) and any(r['phase_valid'] for r in out)
    result={'record_id':row['record_id'],'partition':'train','windows':len(indices),'channel_attempts':len(out),'seconds':time.time()-started,'rows':out}
    (W/'extraction_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('WITNESS_AF_OBSERVED_QRS',json.dumps(result))

def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--smoke',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    if args.smoke:smoke();return
    assert 1<=args.workers<=len(os.sched_getaffinity(0))
    assert json.loads((W/'input_verification.json').read_text())['all_pass']
    freeze=json.loads((W/'preprocessing_freeze.json').read_text())
    assert args.workers==freeze['workers']
    for name,digest in freeze['source_hashes'].items():assert sha(name)==digest,name
    for name,digest in freeze['input_hashes'].items():assert sha(W/name)==digest,name
    rows=list(csv.DictReader((W/'file_manifest.csv').open()));receipts=[];start=time.time()
    (W/'extraction_status.json').write_text(json.dumps({'status':'running','pid':os.getpid(),'workers':args.workers,'declared_files':len(rows),'declared_windows':802491})+'\n')
    try:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
            pending=[pool.submit(process_file,row) for row in rows]
            for future in as_completed(pending):
                receipt=future.result();receipts.append(receipt)
                progress={'completed_files':len(receipts),'total_files':len(rows),'completed_windows':sum(r['windows'] for r in receipts),'declared_windows':802491,'elapsed_seconds':time.time()-start,'last_file':receipt['file_index']}
                temp=W/'extraction_progress.json.partial';temp.write_text(json.dumps(progress)+'\n');os.replace(temp,W/'extraction_progress.json');print(json.dumps(progress),flush=True)
        assert len(receipts)==289 and sum(r['windows'] for r in receipts)==802491
        summary={'status':'complete','complete':True,'files':289,'windows':802491,'channel_attempts':1604982,'phase_valid_by_channel':np.sum([r['phase_valid_by_channel'] for r in receipts],axis=0).astype(int).tolist(),'rr_valid_by_channel':np.sum([r['rr_valid_by_channel'] for r in receipts],axis=0).astype(int).tolist(),'both_phase_unavailable_windows':sum(r['both_phase_unavailable_windows'] for r in receipts),'seconds':time.time()-start}
        (W/'extraction_status.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    except Exception as e:
        (W/'extraction_status.json').write_text(json.dumps({'status':'failed','error':type(e).__name__+': '+str(e),'completed_files':len(receipts)})+'\n');raise
if __name__=='__main__':main()
