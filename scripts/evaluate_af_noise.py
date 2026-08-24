#!/usr/bin/env python3
"""Validate the prespecified finalist-only AF noise bundle."""
import argparse,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser(); p.add_argument('--bundle',type=Path,required=True); p.add_argument('--confirm-frozen-final',action='store_true'); a=p.parse_args()
 if not a.confirm_frozen_final: raise SystemExit('BLOCKED: finalist-only')
 levels=sorted(set(map(float,np.load(a.bundle,allow_pickle=False)['snr_db']))); expected=[-6.,0.,6.,12.,18.]
 if levels!=expected: raise RuntimeError(f'expected {expected}, got {levels}')
 print(json.dumps({'status':'validated','snr_db':levels}))
if __name__=='__main__': main()
