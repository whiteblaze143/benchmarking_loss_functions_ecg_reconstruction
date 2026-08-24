import json
from pathlib import Path
from collections import Counter

p = Path('refine-logs/queue_3arch/queue_state.json')
if p.exists():
    with open(p) as f:
        q = json.load(f)
    jobs = q.get('jobs', [])
    print(f"Total jobs in queue_3arch: {len(jobs)}")
    
    # Check architecture field and name prefix
    names = Counter(j['name'].split('_')[0] for j in jobs)
    print("Names breakdown:", names)
    
    status_by_arch = Counter((j.get('architecture', j['name'].split('_')[0]), j.get('status')) for j in jobs)
    for k, v in sorted(status_by_arch.items()):
        print(f"  {k[0]}: {k[1]} = {v}")
