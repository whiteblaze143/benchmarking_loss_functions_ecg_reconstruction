"""Inspect released RR prefixes; leading 1000 ms is evidence, not proof of calibration."""
from pathlib import Path
from collections import Counter
import io,json,zipfile,csv
import h5py,numpy as np
P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
rows=[]
with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z:
    for name in sorted(n for n in z.namelist() if '_rr_' in n and n.endswith('.h5')):
        with h5py.File(io.BytesIO(z.read(name))) as h:a=h['rr'][:100]
        leading=next((i for i,x in enumerate(a) if x!=1000),len(a))
        rows.append({'record_id':Path(name).parent.name,'daily_file_index':int(Path(name).stem.rsplit('_',1)[1]),'leading_exact_1000ms_intervals_capped_at_100':leading,'first_40_rr_milliseconds':';'.join(str(int(x)) for x in a[:40])})
with (P/'rr_calibration_prefix_audit.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
s={}
for label,rs in [('first_file',[r for r in rows if r['daily_file_index']==0]),('later_files',[r for r in rows if r['daily_file_index']>0])]:
    s[label]={'files':len(rs),'leading_1000ms_counts':dict(sorted(Counter(r['leading_exact_1000ms_intervals_capped_at_100'] for r in rs).items())),'at_least_20_identical_1000ms_intervals':sum(r['leading_exact_1000ms_intervals_capped_at_100']>=20 for r in rs)}
s['limitation']='Only RR prefix values checked. Native RR intervals come from automatic detector; confirm corresponding ECG calibration and source policy before final exclusion.'
(P/'rr_calibration_prefix_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
