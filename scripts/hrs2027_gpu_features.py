"""Batched A100 phase-KME calculation, sharing the CPU signal/beat preprocessing."""
import numpy as np
import torch


def batch_phase_kme(cells_list, anchors, batch_size=128):
    outputs=[]
    anchor=torch.as_tensor(anchors,device='cuda')
    # Feature equivalence requires full float32 products rather than TF32 approximation.
    torch.backends.cuda.matmul.allow_tf32=False
    for start in range(0,len(cells_list),batch_size):
        group=cells_list[start:start+batch_size]
        counts=np.array([len(c) for c in group])
        size=max(1,int(counts.max()))
        padded=np.zeros((len(group),size,16,16),dtype=np.float32)
        for i,cells in enumerate(group):
            padded[i,:len(cells)]=cells
        cells=torch.from_numpy(padded).cuda()
        valid=(torch.arange(size,device='cuda')[None,:] < torch.tensor(counts,device='cuda')[:,None])
        with torch.no_grad():
            distance=(cells.square().sum(-1,keepdim=True)+anchor.square().sum(-1)-2*(cells@anchor.T)).clamp_min_(0)
            kernels=torch.exp(-distance/16)*valid[:,:,None,None]
            kme=kernels.sum(1)/torch.tensor(np.maximum(1,counts),device='cuda')[:,None,None]
            kme=kme/(kme.norm(dim=-1,keepdim=True)+1e-8)
            outputs.append(kme.float().cpu().numpy())
    return np.concatenate(outputs)
