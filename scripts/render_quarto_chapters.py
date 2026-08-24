#!/usr/bin/env python3
"""Render configured Quarto chapters sequentially with resumable audit state."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/1024**2
    return 0.0


def free_gib(path:Path):
    s=os.statvfs(path);return s.f_bavail*s.f_frsize/1024**3


def chapters(config:Path):
    return re.findall(r"^\s{4,}-\s+([\w.-]+\.qmd)\s*$",config.read_text(),flags=re.M)


def save(path:Path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--round-dir',type=Path,required=True);p.add_argument('--min-ram-gib',type=float,default=4)
    p.add_argument('--min-disk-gib',type=float,default=8);p.add_argument('--timeout',type=int,default=1800);p.add_argument('--force',action='store_true')
    p.add_argument('--quarto',type=Path,default=ROOT/'.tools/quarto-1.10.18/bin/quarto')
    p.add_argument('--stop-file',type=Path,default=ROOT/'review-stage/STOP_BOOK_RENDER');a=p.parse_args()
    if not a.round_dir.is_absolute():a.round_dir=(ROOT/a.round_dir).resolve()
    if not a.stop_file.is_absolute():a.stop_file=(ROOT/a.stop_file).resolve()
    if not a.quarto.is_absolute():a.quarto=(ROOT/a.quarto).resolve()
    if not a.quarto.is_file():raise FileNotFoundError(f'Pinned Quarto binary not found: {a.quarto}')
    a.round_dir.mkdir(parents=True,exist_ok=True);state_path=a.round_dir/'render_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'started_at':now(),'status':'running','chapters':{}}
    version=subprocess.run([str(a.quarto),'--version'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True).stdout.strip()
    state['render_contract']={'quarto_binary':str(a.quarto.relative_to(ROOT)),'quarto_version':version,
                              'python':'/home/mithunmanivannan/.venv/bin/python3','sequential':True}
    save(state_path,state)
    env=os.environ.copy();env['QUARTO_PYTHON']='/home/mithunmanivannan/.venv/bin/python3';env['PATH']='/home/mithunmanivannan/.venv/bin:'+env['PATH']
    for name in chapters(ROOT/'book/_quarto.yml'):
        if a.stop_file.exists():state['status']='stopped';save(state_path,state);return 130
        if not a.force and state['chapters'].get(name,{}).get('status')=='complete':continue
        while available_gib()<a.min_ram_gib or free_gib(ROOT)<a.min_disk_gib:
            print(json.dumps({'event':'resource_wait','chapter':name,'ram_gib':available_gib(),'disk_gib':free_gib(ROOT)}),flush=True)
            if a.stop_file.exists():state['status']='stopped';save(state_path,state);return 130
            time.sleep(30)
        started=time.monotonic();entry={'status':'running','started_at':now(),'source':f'book/{name}'};state['chapters'][name]=entry;save(state_path,state)
        try:
            result=subprocess.run([str(a.quarto),'render',f'book/{name}','--to','html'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=a.timeout)
        except subprocess.TimeoutExpired as exc:
            result=subprocess.CompletedProcess(exc.cmd,124,(exc.stdout or '')+(exc.stderr or '')+'\nTIMEOUT\n')
        log=a.round_dir/(name.replace('.qmd','.log'));log.write_text(result.stdout)
        entry.update({'status':'complete' if result.returncode==0 else 'error','returncode':result.returncode,
                      'completed_at':now(),'duration_seconds':round(time.monotonic()-started,3),'log':str(log.relative_to(ROOT))})
        save(state_path,state);print(json.dumps({'event':'chapter_rendered','chapter':name,**entry}),flush=True)
    counts={s:sum(v['status']==s for v in state['chapters'].values()) for s in ('complete','error','running')}
    state.update({'status':'complete' if counts['error']==0 else 'completed_with_errors','completed_at':now(),'counts':counts});save(state_path,state)
    print(json.dumps({'event':'render_sweep_complete',**counts}),flush=True);return int(counts['error']>0)


if __name__=='__main__':raise SystemExit(main())
