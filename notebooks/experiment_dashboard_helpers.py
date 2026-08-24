"""Read-only helpers shared by experiment dashboards."""
import hashlib,json,re,sqlite3,subprocess,tempfile
from pathlib import Path
import pandas as pd

def sql(path,q,params=()):
 with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True,timeout=30) as c:return pd.read_sql_query(q,c,params=params)
def js(x):
 try:return json.loads(x) if x else {}
 except:return {}
def flatten(s,prefix='metric.'):return pd.json_normalize([js(x) for x in s]).add_prefix(prefix)
def direction(n):
 n=n.lower()
 if any(x in n for x in ('loss','mse','mae','rmse','error','failure','duration','p95_abs')):return'min'
 if any(x in n for x in ('pearson','iou','f1','ppv','sens','spec','auroc','auprc','r2','retention')):return'max'
def best(df,metrics,groups=(),k=1,id_col='model_id'):
 out=[]
 for m in metrics:
  if m not in df or not pd.api.types.is_numeric_dtype(df[m]) or not direction(m):continue
  gs=df.groupby(list(groups),dropna=False) if groups else [((),df)]
  for key,g in gs:
   g=g[g[m].notna()]; chosen=(g.nsmallest if direction(m)=='min' else g.nlargest)(k,m)
   for rank,(_,r) in enumerate(chosen.iterrows(),1):
    z={'metric':m,'direction':direction(m),'rank':rank,'value':r[m],id_col:r.get(id_col)}; key=key if isinstance(key,tuple) else(key,);z.update(dict(zip(groups,key)));out.append(z)
 return pd.DataFrame(out)
def zstd_db(p):
 p=Path(p);d=Path(tempfile.gettempdir())/'ecgaim_dashboard_cache';d.mkdir(exist_ok=True);k=hashlib.sha256(f'{p.stat().st_size}:{p.stat().st_mtime_ns}'.encode()).hexdigest()[:12];o=d/f'{p.stem}.{k}.sqlite'
 if not o.exists():subprocess.run(['zstd','-d','-q','-f',str(p),'-o',str(o)],check=True)
 return o
def queue(p):
 q=sql(p,'select * from jobs order by ordinal');q=pd.concat([q.drop(columns=['cell_json','summary_json']),pd.json_normalize([js(x) for x in q.cell_json]).add_prefix('cell.'),flatten(q.summary_json)],axis=1);q['cell_name']=q.get('cell.name',q.id);q['seed']=pd.to_numeric(q.id.str.extract(r'_s(\d+)_')[0],errors='coerce');q['observed_lead']=q.id.str.extract(r'_l([01])$')[0].map({'0':'I','1':'II'});return q
def blinded(p,cohort):
 e=sql(p,'select * from evaluations');b=sql(p,'select * from boundary_summaries');e['cohort']=b['cohort']=cohort;return e,b
def oracle(p,cohort):
 e=sql(p,'select * from evaluations');e=pd.concat([e.drop(columns='primary_summary_json'),flatten(e.primary_summary_json)],axis=1);e['cohort']=cohort;return e
