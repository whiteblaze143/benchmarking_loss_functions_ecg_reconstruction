"""Execute unchanged author R function and compare explicit Python binned port."""
from pathlib import Path
import csv,hashlib,json,math,os,subprocess
import numpy as np
from metrics import charitos_author_binned_port
P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
A=P/'source_docs/charitos_author/AF.density.function.R'
rng=np.random.default_rng(20261120)
fixtures=[('zero',[0,0,0],1),('all_AF',[1,1,1],1),('single_bin',[0,1,0,0],1),('uniform',[0.5]*10,1),('endpoints',[1,0,0,1],1),('high_burden',[1,1,1,0.8],1)]
for i in range(60):
    n=int(rng.integers(3,65));x=rng.uniform(0,1,n);x[rng.random(n)<.4]=0
    fixtures.append((f'random_{i}',x.tolist(),1))
inputs=P/'author_r_fixtures.tsv'
with inputs.open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['id','timeunits','data'])
    for name,x,unit in fixtures:w.writerow([name,unit,';'.join(format(v,'.17g') for v in x)])
script=P/'run_author_density.R'
script.write_text('''args <- commandArgs(trailingOnly=TRUE)
source(args[1])
x <- read.delim(args[2], stringsAsFactors=FALSE)
y <- data.frame(id=x$id, density=NA_real_, raw_density=NA_real_)
for (i in seq_len(nrow(x))) {
  values <- as.numeric(strsplit(x$data[i], ";", fixed=TRUE)[[1]])
  y$density[i] <- AF.density(values,timeunits=x$timeunits[i],limit=TRUE)
  y$raw_density[i] <- AF.density(values,timeunits=x$timeunits[i],limit=FALSE)
}
write.table(y,args[3],sep="\\t",row.names=FALSE,quote=FALSE,digits=17)
'''.replace(',digits=17', ''))
# R's write.table uses high precision internally; retain options as well.
script.write_text('options(digits=17)\n'+script.read_text())
env=os.environ.copy();env['R_HOME']='/tmp/af_density_r_runtime/root/usr/lib/R';env['LD_LIBRARY_PATH']=env['R_HOME']+'/lib'
result=subprocess.run([env['R_HOME']+'/bin/exec/R','--vanilla','--slave','-f',str(script),'--args',str(A),str(inputs),str(P/'author_r_results.tsv')],env=env,capture_output=True,text=True,check=True)
r=list(csv.DictReader((P/'author_r_results.tsv').open(),delimiter='\t'))
checks=[]
for (name,x,unit),reference in zip(fixtures,r):
    assert name==reference['id']
    for limited,key in [(True,'density'),(False,'raw_density')]:
        actual=float(charitos_author_binned_port(x,timeunits=unit,limit=limited));expected=float(reference[key]);delta=0.0 if actual==expected else abs(actual-expected)
        checks.append({'id':name,'metric':key,'author_R':expected if math.isfinite(expected) else str(expected),'python_port':actual if math.isfinite(actual) else str(actual),'absolute_difference':delta,'pass':delta<1e-11})
s={'author_source_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'runtime':'R 4.3.3 Ubuntu r-base-core 4.3.3-2build2 unpacked locally without system installation','fixtures':len(fixtures),'comparisons':len(checks),'all_pass':all(c['pass'] for c in checks),'max_absolute_difference':max(c['absolute_difference'] for c in checks),'checks':checks,'limitation':'Verifies binned port against original R implementation. Does not make binned and continuous interval metrics identical.'}
(P/'author_r_verification.json').write_text(json.dumps(s,indent=2)+'\n');print({k:v for k,v in s.items() if k!='checks'})
assert s['all_pass']
