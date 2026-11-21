"""Investigate fresh-process SetII numerical replay without changing frozen scores."""
import argparse,json,time
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score
from study import OUT,frame,model_for,predict,TACHY,save
from analyze import clinical

def main():
 p=argparse.ArgumentParser();p.add_argument('--tf32',choices=['on','off'],required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=a.tf32=='on'
 assert torch.backends.cudnn.version()==90300;f=frame();name='set_II';directory=OUT/name;archive=np.load(directory/'predictions.npz');cp=torch.load(directory/'best.pt',weights_only=True);model=model_for('set','II').cuda();model.load_state_dict(cp['state_dict']);x=torch.tensor(np.load(OUT/'responses_II.npy'),device='cuda');lock=json.loads((directory/'evaluation_lock.json').read_text());y=archive['target'];results={};outputs={}
 for part in ['validation','test']:
  idx=np.flatnonzero(f.partition.eq(part).to_numpy()&archive['completed']);logits=predict(model,x,torch.tensor(idx,device='cuda'),'set');score=1/(1+np.exp(-logits));old=archive['probabilities'][idx];logitdelta=logits-archive['logits'][idx];diffs=[]
  for j,t in enumerate(lock['label_thresholds']):diffs.append(int(((score[:,j]>=t)!=(old[:,j]>=t)).sum()) if t is not None else None)
  threshold=lock['severe_group_threshold'];group=(y[idx][:,TACHY].sum(axis=1)>0).astype(int);newclinical=clinical(group,score[:,TACHY].max(axis=1),np.ones(len(idx),bool),threshold);oldclinical=clinical(group,old[:,TACHY].max(axis=1),np.ones(len(idx),bool),threshold)
  classes=json.loads((OUT/'protocol.json').read_text())['macro_indices'];newmacro=float(np.mean([average_precision_score(y[idx,j],score[:,j]) for j in classes]));oldmacro=float(np.mean([average_precision_score(y[idx,j],old[:,j]) for j in classes]));results[part]={'completed_n':len(idx),'max_logit_difference':float(abs(logitdelta).max()),'max_probability_difference':float(abs(score-old).max()),'per_label_decision_changes':diffs,'severe_group_decision_changes':int(((score[:,TACHY].max(axis=1)>=threshold)!=(old[:,TACHY].max(axis=1)>=threshold)).sum()),'old_macro_auprc_completed':oldmacro,'new_macro_auprc_completed':newmacro,'new_minus_old_macro':newmacro-oldmacro,'old_group_detected':oldclinical['detected_positive_n'],'new_group_detected':newclinical['detected_positive_n'],'old_group_false_positive':oldclinical['false_positive_n'],'new_group_false_positive':newclinical['false_positive_n']};outputs[part+'_logits']=logits;outputs[part+'_indices']=idx;print(part,json.dumps(results[part]),flush=True)
 np.savez_compressed(OUT/f'setII_replay_tf32_{a.tf32}.npz',**outputs);save(OUT/f'setII_replay_tf32_{a.tf32}.json',{'cudnn_benchmark':True,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'method':'original full-cohort response tensor, validation first then test; no training, thresholds unchanged; frozen archive untouched','results':results})
if __name__=='__main__':main()
