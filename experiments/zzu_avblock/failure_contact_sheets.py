"""Observed-waveform contact sheets for all indeterminate heldout records."""
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from leadii import OUT,cohort
f=cohort();e=pd.read_parquet(OUT/'extraction.parquet');e=e[f.partition.eq('test').to_numpy() & e.decision_status.ne('completed').to_numpy()].copy();x=np.load(OUT/'leadII_mV.npy',mmap_mode='r');index={u:i for i,u in enumerate(f.record_uid)};out=OUT/'figures';out.mkdir(exist_ok=True)
for start in range(0,len(e),30):
 chunk=e.iloc[start:start+30];fig,axes=plt.subplots(10,3,figsize=(18,19))
 for ax,(_,row) in zip(axes.ravel(),chunk.iterrows()):
  sig=x[index[row.record_uid]];ax.plot(np.arange(5000)/500,sig,lw=.4,color='black');peaks=np.asarray(row.rpeaks,dtype=int);ax.scatter(peaks/500,sig[peaks],s=8,color='red');ax.set_title(f'{row.record_uid} AVB={int(f.label.iloc[index[row.record_uid]])} peaks={row.detected_peaks} validRR={row.valid_beats}',fontsize=8);ax.set_xlim(0,10);ax.tick_params(labelsize=6)
 for ax in axes.ravel()[len(chunk):]:ax.set_visible(False)
 fig.tight_layout();fig.savefig(out/f'all_failures_{start//30+1}.png',dpi=100);plt.close(fig)
print('CONTACT_SHEETS',len(e),'failures',flush=True)
