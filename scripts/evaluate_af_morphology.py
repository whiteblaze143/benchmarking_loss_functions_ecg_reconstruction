#!/usr/bin/env python3
"""Store independent frozen P/R detector outputs without CSV expansion."""
import argparse,json,sqlite3
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--measurements',type=Path,required=True); p.add_argument('--output-db',type=Path,required=True); p.add_argument('--confirm-frozen-final',action='store_true'); a=p.parse_args()
 if not a.confirm_frozen_final: raise SystemExit('BLOCKED: finalist-only')
 con=sqlite3.connect(a.output_db); con.execute('CREATE TABLE IF NOT EXISTS morphology(record_id TEXT,lead TEXT,condition TEXT,p_segments INTEGER,rr_mae REAL,rr_correlation REAL,detector_hash TEXT,PRIMARY KEY(record_id,lead,condition))')
 rows=[]
 for line in a.measurements.read_text().splitlines():
  x=json.loads(line); rows.append(tuple(x[k] for k in ('record_id','lead','condition','p_segments','rr_mae','rr_correlation','detector_hash')))
 con.executemany('INSERT OR REPLACE INTO morphology VALUES(?,?,?,?,?,?,?)',rows); con.commit(); con.close(); print(json.dumps({'stored':len(rows)}))
if __name__=='__main__': main()
