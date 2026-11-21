"""Training-only post-run diagnostics; does not change models or predictions."""
import sys,json
import numpy as np,pandas as pd,torch
from leadii import OUT,cohort,model_for,save

def main():
 torch.set_num_threads(2);torch.manual_seed(42)
 f=cohort();ex=pd.read_parquet(OUT/'extraction.parquet');idx=np.flatnonzero(f.partition.eq('train').to_numpy()&ex.decision_status.eq('completed').to_numpy())
 responses=np.load(OUT/'responses.npy');x=torch.tensor(responses[idx[:512]],device='cuda');y=torch.tensor(f.iloc[idx[:512]].label.to_numpy(),dtype=torch.float32,device='cuda')
 model=model_for('M1').cuda();model.load_state_dict(torch.load(OUT/'M1/best.pt',weights_only=True)['state_dict']);model.eval()
 with torch.no_grad():
  fp=model(x).float()
  with torch.autocast('cuda',dtype=torch.bfloat16):bf=model(x.to(torch.bfloat16)).float()
 result={'data':'first512 completed training records only; no development/test diagnostic fitting','feature_std_across_training_records_mean':float(responses[idx].std(axis=0).mean()),'feature_abs_max':float(abs(responses[idx]).max()),'best_logits_fp32_std':float(fp.std()),'best_logits_bf16_std':float(bf.std()),'unique_fp32_scores':int(torch.sigmoid(fp).unique().numel()),'unique_bf16_scores':int(torch.sigmoid(bf).unique().numel()),'fp32_bf16_max_logit_difference':float((fp-bf).abs().max())}
 model.train();model.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[:128]),y[:128]);loss.backward()
 result['training_fp32_loss']=float(loss.detach());result['gradient_norms']={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
 result['interpretation']='BF16 may quantize small score variation; a failed development fit does not establish representation inferiority. No model/threshold changes or reruns based on heldout outcomes.'
 save(OUT/'training_only_diagnostics.json',result);print(json.dumps({k:v for k,v in result.items() if k!='gradient_norms'},indent=2))
if __name__=='__main__':main()
