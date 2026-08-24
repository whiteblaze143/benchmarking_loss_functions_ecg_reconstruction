#!/usr/bin/env python3
"""Build a hash-bound manifest for a fully rendered Quarto book."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,default=ROOT/'review-stage/BOOK_MECHANICAL_AUDIT.json')
    p.add_argument('--render-state',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'book/_book/RELEASE_MANIFEST.json')
    p.add_argument('--provisional',action='store_true',help='Emit a clearly provisional manifest when mandatory scientific gates fail.');a=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_quarto_book.py'),'--book',str(ROOT/'book'),'--json',str(a.audit)],cwd=ROOT,check=True)
    audit=json.loads(a.audit.read_text());render=json.loads(a.render_state.read_text())
    if render.get('status')!='complete':raise RuntimeError(f"Render sweep is not clean: {render.get('status')}")
    bad={k:v for k,v in audit['summary'].items() if k in {'missing_html','incomplete_html','stale_html','missing_resources','unlabeled_blocks','wrong_renderer','zero_output_chapters','external_runtime','identifier_hits'} and v}
    if audit['summary'].get('duplicate_labels'):bad['duplicate_labels']=audit['summary']['duplicate_labels']
    if bad:raise RuntimeError(f'Mechanical release gates failed: {bad}')
    final_gate=json.loads((ROOT/'results/factorial_v4_clinical/FINAL_COMPLETION.json').read_text())
    scientific_pass=final_gate.get('status')=='PASS' and all(v.get('passed',False) for v in final_gate.get('checks',{}).values())
    if not scientific_pass and not a.provisional:
        raise RuntimeError('Mandatory scientific release gate FINAL_COMPLETION.json is not PASS; use --provisional only for an explicitly non-deployable audit artifact.')
    config_sha=sha(ROOT/'book/_quarto.yml')
    requirements=ROOT/'book/requirements-book.txt'
    for row in render.get('chapters',{}).values():
        q=ROOT/row['source'];h=hashlib.sha256();h.update(q.read_bytes());h.update((ROOT/'book/_quarto.yml').read_bytes())
        if row.get('input_sha256')!=h.hexdigest():raise RuntimeError(f"Render state is not bound to current inputs: {q}")
    sources={};outputs={}
    for row in audit['chapters']:
        q=ROOT/'book'/row['chapter'];h=ROOT/'book/_book'/row['chapter'].replace('.qmd','.html')
        sources[str(q.relative_to(ROOT))]={'sha256':sha(q),'bytes':q.stat().st_size}
        outputs[str(h.relative_to(ROOT))]={'sha256':sha(h),'bytes':h.stat().st_size,'output_blocks':row['output_blocks']}
    key_artifacts=[
        ROOT/'results/factorial_v4/completeness.json',ROOT/'results/factorial_v4/protocol_audit.json',
        ROOT/'results/factorial_v4/main_48_cell_table.csv',ROOT/'results/factorial_v4_clinical/FINAL_COMPLETION.json',
        ROOT/'results/factorial_v4_clinical/smartwatch_task_fidelity.csv',
        ROOT/'results/clinical_biomarkers_multids/clinical_metrics.db',ROOT/'results/checkpoint_embeddings/compact.sqlite',
    ]
    artifacts={str(x.relative_to(ROOT)):{'exists':x.exists(),**({'sha256':sha(x),'bytes':x.stat().st_size} if x.is_file() else {})} for x in key_artifacts}
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
    dirty=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True).stdout.splitlines()
    payload={'schema_version':1,'generated_at':dt.datetime.now(dt.timezone.utc).isoformat(),
             'release_status':'production_verified' if scientific_pass else 'PROVISIONAL_NOT_DEPLOYABLE_SCIENTIFIC_GATE_FAILED',
             'render_contract':render['render_contract'],'git_commit':commit,'git_dirty':bool(dirty),'git_changed_paths':dirty,
             'quarto_config_sha256':config_sha,'mandatory_scientific_gate':final_gate,
             'audit_summary':audit['summary'],'sources':sources,'outputs':outputs,'key_artifacts':artifacts,
             'book_requirements':{'path':str(requirements.relative_to(ROOT)),'sha256':sha(requirements),'bytes':requirements.stat().st_size},
             'external_runtime_resources':sorted({x for r in audit['chapters'] for x in r['external_resources']})}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(tmp,a.output)
    print(json.dumps({'output':str(a.output),'sources':len(sources),'outputs':len(outputs),'external_resources':len(payload['external_runtime_resources'])}))
if __name__=='__main__':main()
