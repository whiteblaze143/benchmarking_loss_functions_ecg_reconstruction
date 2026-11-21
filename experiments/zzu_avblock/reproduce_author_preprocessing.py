"""Execute released preprocessing functions unchanged; no model imports or training.

Only data-location I/O and lineage tracing are supplied externally. Source AST
assignments and numerical/segmentation/selection functions are executed verbatim.
"""
import argparse,ast,contextlib,hashlib,io,json,os,re
from pathlib import Path
import zipfile,types
import numpy as np,pandas as pd
from scipy.io import loadmat

ROOT=Path('/data/mithunmanivannan/datasets/figshare_31229326_v3')
SOURCE=ROOT/'extracted/Quality Verification/utils/Segmentation Deep Learning Validation.py'
CONSTANTS={'SAMPLING_RATE','SEGMENT_DURATION_SEC','SEGMENT_LENGTH','NORMALIZE_RANGE','SLIDE_STEP_SEC','SLIDE_STEP_POINTS','OVERLAP_PERCENT','ORIGINAL_DISEASE_LABELS','EXCLUDE_LABEL_IDS','KEEP_LABEL_IDS','MERGE_RULE','NEW_CLASS_LABELS','NUM_CLASSES','LEAD_KEYS','NUM_LEADS','INPUT_SHAPE'}
FUNCTIONS={'parse_hea_file','get_file_diagnosis','get_original_label','merge_labels','min_max_normalize','calculate_sliding_segments','load_and_process_ecg_data'}

def source_environment():
    tree=ast.parse(SOURCE.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in CONSTANTS for t in node.targets):nodes.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in FUNCTIONS:nodes.append(node)
    env={'np':np,'os':os,'re':re,'loadmat':loadmat,'tqdm':lambda values,**kwargs:values}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    return env

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'preprocessing_receipt.json').exists():raise RuntimeError('Completed reference exists; do not overwrite')
    archive_path=ROOT/'Database # A publicly available 12-lead electrocardiogram criti-cal value dataset covering 13126 records.zip'
    source_dir=ROOT/'extracted'/archive_path.stem
    frame=pd.read_parquet(ROOT/'analysis/record_manifest.parquet');critical=frame[frame.cohort=='critical'].set_index('file_name')
    env=source_environment();env['DATA_DIR']=str(source_dir)
    order_path=out/'source_directory_order.json'
    enum=json.loads(order_path.read_text()) if order_path.exists() else [f for f in os.listdir(source_dir) if f.endswith('.mat') and not f.startswith('.')]
    (out/'source_directory_order.json').write_text(json.dumps(enum))
    trace=[];original_label=env['get_original_label']
    def traced_label(filename):
        label=original_label(filename);trace.append((filename,label));return label
    env['get_original_label']=traced_label
    with zipfile.ZipFile(archive_path) as archive:
        # The I/O adapter changes only storage access. Each original MAT byte is
        # the same checksum-locked archive member as the extracted file.
        def archive_load(path):
            filename=Path(path).stem;row=critical.loc[filename]
            return loadmat(io.BytesIO(archive.read(row.mat_archive_member)))
        def archive_open(path,mode='r',encoding='utf-8'):
            if mode!='r':raise ValueError('Source preprocessing attempted non-read header I/O')
            row=critical.loc[Path(path).stem];raw=archive.read(row.header_archive_member)
            assert hashlib.sha256(raw).hexdigest()==row.header_sha256
            return io.TextIOWrapper(io.BytesIO(raw),encoding=encoding,newline=None)
        env['loadmat']=archive_load;env['open']=archive_open
        env['os']=types.SimpleNamespace(path=os.path,listdir=lambda path:enum)
        with (out/'source_preprocessing.log').open('w') as log,contextlib.redirect_stdout(log):
            x,y=env['load_and_process_ecg_data']()
    assert [name for name,_ in trace]==enum
    if x is None:raise RuntimeError('Released preprocessing returned no data')
    records=[];windows=[];index=0
    for filename,label in trace:
        row=critical.loc[Path(filename).stem]
        if label is None:reason='source_excludes_multilabel_or_unrecognized_diagnosis'
        elif label not in env['MERGE_RULE']:reason='source_merge_rule_omits_this_diagnosis'
        else:reason=''
        n,starts=env['calculate_sliding_segments'](int(row.sample_count_per_lead))
        if not reason and n==0:reason='source_duration_below10s'
        records.append({'record_uid':row.record_uid,'original_label_id':label,'source_selected':not bool(reason),'exclusion_reason':reason,'generated_windows_if_selected':n if not reason else 0})
        if reason:continue
        for start in starts:
            windows.append({'window_index':index,'record_uid':row.record_uid,'start_sample':start,'end_sample':start+env['SEGMENT_LENGTH'],'source_merged_label':env['MERGE_RULE'][label],'original_label_id':label,'overlap_with_previous_window':bool(start!=starts[0] and start-starts[starts.index(start)-1]<env['SEGMENT_LENGTH']),'record_label_AVB':int('High-Grade/Third-Degree Atrioventricular Block' in json.loads(row.labels_json))});index+=1
    ledger=pd.DataFrame(windows);assert len(x)==len(y)==len(ledger)
    assert np.array_equal(y,ledger.source_merged_label.to_numpy())
    assert x.shape[1:]==(5000,3) and x.dtype==np.float32 and np.isfinite(x).all()
    np.save(out/'source_normalized_windows.npy',x);np.save(out/'source_merged_labels.npy',y)
    ledger.to_parquet(out/'window_manifest.parquet',index=False);pd.DataFrame(records).to_csv(out/'record_selection.csv',index=False)
    # Compare actual output to direct source-function calls from original MATs.
    parity=[]
    with zipfile.ZipFile(archive_path) as archive:
        chosen=np.unique(np.linspace(0,len(ledger)-1,12,dtype=int))
        for j in chosen:
            w=ledger.iloc[j];row=frame.set_index('record_uid').loc[w.record_uid]
            d=loadmat(io.BytesIO(archive.read(row.mat_archive_member)));raw=np.stack([d[k].flatten().astype(np.float32) for k in env['LEAD_KEYS']])
            expected=env['min_max_normalize'](raw[:,int(w.start_sample):int(w.end_sample)].T,np.min(raw),np.max(raw))
            assert np.array_equal(expected,x[j]);parity.append({'window_index':int(j),'record_uid':w.record_uid,'bitwise_source_function_parity':True})
    receipt={'scope':'released-script preprocessing reference, not manuscript-exact replication or new AV-block evaluation','source_script':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_functions_executed_verbatim':sorted(FUNCTIONS),'input_critical_records':len(critical),'selected_records':int(pd.DataFrame(records).source_selected.sum()),'windows':len(x),'tensor_shape':list(x.shape),'tensor_dtype':str(x.dtype),'class_counts':{str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},'record_AVB_windows':int(ledger.record_label_AVB.sum()),'record_AVB_records':ledger.loc[ledger.record_label_AVB==1,'record_uid'].nunique(),'overlapping_tail_windows':int(ledger.overlap_with_previous_window.sum()),'normalization':'released full-record global min/max across I,II,III; dimensionless [0,1]; no training-fitted mean/std','filtering':'no wavelet or median denoising is called by released preprocessing','segmentation':'source10s/10s step, including source overlapping final tail','parity_checks':parity,'training_performed':False,'source_random_window_split_executed':False,'manuscript_replication_status':'incomplete: missing wavelet family/order/levels/thresholding, median window widths/order/edge handling, five-second code, patient mapping and official folds'}
    for name in ['source_normalized_windows.npy','window_manifest.parquet']:
        with (out/name).open('rb') as h:receipt[name+'_sha256']=hashlib.file_digest(h,'sha256').hexdigest()
    (out/'preprocessing_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='parity_checks'},indent=2),flush=True)
if __name__=='__main__':main()
