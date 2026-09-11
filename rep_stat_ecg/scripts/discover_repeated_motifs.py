"""Restart-safe repSpat M3: biased IMQ MMD², fixed block permutations, BH graph."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rep_stat_ecg.src.motifs.mmd import (benjamini_hochberg, build_attribute_blocks,
    compute_n_blocks, precompute_block_kernel_sums, run_block_permutation_test)
from rep_stat_ecg.src.motifs.quotient import QuotientClosure
from rep_stat_ecg.src.motifs.spatial_domains import CoordinateStandardizer, VoxelGrid3D

PAIR_KEYS = {"domain_i", "domain_j", "n_i", "n_j", "n_blocks_i", "n_blocks_j",
             "mmd2_obs", "n_perm", "n_exceed", "p_perm", "seed"}


def atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False) as f:
        tmp = Path(f.name)
    try:
        np.savez(tmp, **values)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_pair(path: Path, d1: int, d2: int, n_perm: int, seed: int) -> dict | None:
    if not path.exists(): return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if not PAIR_KEYS.issubset(z.files): return None
            row = {k: z[k].item() for k in PAIR_KEYS}
        if (int(row["domain_i"]), int(row["domain_j"]), int(row["n_perm"]), int(row["seed"])) != (d1, d2, n_perm, seed):
            return None
        return row
    except (OSError, ValueError, EOFError):
        return None


def assign_domains(df, registry, std_path):
    params=json.load(Path(std_path).open()); std=CoordinateStandardizer()
    std.median=np.asarray(params["median"],dtype=np.float32); std.mad=np.asarray(params["mad"],dtype=np.float32); std.is_fitted=True
    mapping=dict(zip(registry.voxel_id,registry.domain_id)); grid=VoxelGrid3D(grid_dim=24,coord_min=-3.,coord_max=3.)
    grid.eligible_voxels=set(mapping); grid.voxel_to_domain=mapping
    vox,inside=grid.point_to_voxel_index(std.transform(df[["v_x","v_y","v_z"]].to_numpy()))
    out=df.copy(); out["domain_id"]=np.fromiter((mapping.get(int(v),-1) if ok else -1 for v,ok in zip(vox,inside)),dtype=np.int32,count=len(df))
    return out[out.domain_id>=0].copy()


def main():
    p=argparse.ArgumentParser(description="Paper-faithful repSpat inference on Q-VCG domains")
    p.add_argument("--micro-table",default="refine-logs/qvcg/VCG_MICROSTATE_TABLE.parquet")
    p.add_argument("--domain-stats",default="refine-logs/qvcg/VCG_DOMAIN_STATS.parquet")
    p.add_argument("--domain-registry",default="refine-logs/qvcg/VCG_SPATIAL_DOMAIN_REGISTRY.parquet")
    p.add_argument("--std-json",default="refine-logs/qvcg/VCG_COORDINATE_STANDARDIZER.json")
    p.add_argument("--out-dir",default="refine-logs/qvcg"); p.add_argument("--cahc-neighborhood-size",type=int,default=10)
    p.add_argument("--n-perm",type=int,default=999); p.add_argument("--alpha",type=float,default=.05)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--max-pairs",type=int,default=None,help="Validation-only pair cap; skips global BH")
    a=p.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); checkpoint_dir=out/"m3_pair_checkpoints"
    stats=pd.read_parquet(a.domain_stats); registry=pd.read_parquet(a.domain_registry)
    df=assign_domains(pd.read_parquet(a.micro_table),registry,a.std_json)
    domains=sorted(map(int,stats.domain_id.unique())); expected=list(combinations(domains,2)); work=expected if a.max_pairs is None else expected[:a.max_pairs]
    print(f"Assigned {len(df):,} microstates to {len(domains)} domains; {len(work)} pair(s) requested",flush=True)

    rows=[]; missing=[]
    for ix,(d1,d2) in enumerate(expected):
        pair_seed=a.seed+ix; path=checkpoint_dir/f"pair_{d1:03d}_{d2:03d}.npz"
        cached=load_pair(path,d1,d2,a.n_perm,pair_seed)
        if cached is not None: rows.append(cached)
        elif (d1,d2) in work: missing.append((ix,d1,d2,path))
    print(f"Validated {len(rows)} completed checkpoints; {len(missing)} requested pairs missing",flush=True)

    active=sorted({d for _,d1,d2,_ in missing for d in (d1,d2)}); blocks={}
    for d in tqdm(active,desc="K-means attribute blocks"):
        xi=df.loc[df.domain_id==d,["s","rho","kappa"]].to_numpy(dtype=np.float64)
        blocks[d]=build_attribute_blocks(xi,compute_n_blocks(len(xi),a.cahc_neighborhood_size),a.seed+d)
    for ix,d1,d2,path in tqdm(missing,desc="IMQ fixed block-permutation tests"):
        S,sizes,diag=precompute_block_kernel_sums(blocks[d1],blocks[d2],"IMQ",1.)
        obs,pv,n_done,_=run_block_permutation_test(S,sizes,len(blocks[d1]),len(blocks[d2]),B_max=a.n_perm,rng=np.random.RandomState(a.seed+ix),self_diagonal=diag)
        n_exceed=int(round(pv*(a.n_perm+1)-1)); row={"domain_i":d1,"domain_j":d2,"n_i":int(sizes[:len(blocks[d1])].sum()),"n_j":int(sizes[len(blocks[d1]):].sum()),"n_blocks_i":len(blocks[d1]),"n_blocks_j":len(blocks[d2]),"mmd2_obs":obs,"n_perm":a.n_perm,"n_exceed":n_exceed,"p_perm":pv,"seed":a.seed+ix}
        atomic_npz(path,**row); rows.append(row)

    # BH is deliberately impossible until every distinct pair has a valid result.
    complete=[]
    for ix,(d1,d2) in enumerate(expected):
        row=load_pair(checkpoint_dir/f"pair_{d1:03d}_{d2:03d}.npz",d1,d2,a.n_perm,a.seed+ix)
        if row is not None: complete.append(row)
    if len(complete)!=len(expected):
        print(f"Checkpointed {len(complete)}/{len(expected)} pairs; global BH deferred.",flush=True); return
    complete.sort(key=lambda r:(r["domain_i"],r["domain_j"])); q=benjamini_hochberg(np.asarray([r["p_perm"] for r in complete]),a.alpha)
    graph_rows=[]; matrix=np.zeros((max(domains)+1,max(domains)+1),dtype=np.float64)
    for r,qv in zip(complete,q):
        d1,d2=int(r["domain_i"]),int(r["domain_j"]); matrix[d1,d2]=matrix[d2,d1]=r["mmd2_obs"]
        graph_rows.append({"domain_1":d1,"domain_2":d2,"mmd2_imq_obs":r["mmd2_obs"],"p_perm_imq":r["p_perm"],"q_bh_imq":float(qv),"bh_reject_imq":bool(qv<a.alpha),"similarity_edge":bool(qv>=a.alpha),**r})
    pd.DataFrame(graph_rows).to_parquet(out/"VCG_REPSPAT_PAIR_TESTS.parquet",index=False); np.save(out/"VCG_MMD_MATRIX.npy",matrix)
    qc=QuotientClosure(max(domains)+1,0.); qc.build_similarity_graph(graph_rows); mapping=qc.compute_quotient_classes()
    motifs=[{"motif_id":m,"domain_ids":json.dumps(ds),"num_domains":len(ds),"status":qc.motif_status[m],"is_clique":qc.motif_is_clique[m]} for m,ds in qc.motif_to_domains.items()]
    pd.DataFrame(motifs).to_parquet(out/"VCG_MOTIF_REGISTRY.parquet",index=False); registry["motif_id"]=registry.domain_id.map(mapping); registry.to_parquet(out/"VCG_VOXEL_TO_MOTIF.parquet",index=False)
    ambiguous=[m for m,ok in qc.motif_is_clique.items() if not ok]
    summary={"method":"repSpat adapted block construction + exact inference","mmd_estimator":"biased empirical MMD squared (Eq. 6)","primary_kernel":"IMQ","imq_c_squared":1.,"block_count_parameter_m":a.cahc_neighborhood_size,"n_domain_pairs":len(complete),"n_permutations_each":a.n_perm,"bh_alpha":a.alpha,"edge_semantics":"BH non-rejection; not equivalence","num_similarity_edges":qc.graph.number_of_edges(),"num_connected_components":qc.num_motifs,"non_clique_component_ids":ambiguous,"num_non_clique_components":len(ambiguous)}
    json.dump(summary,(out/"VCG_MOTIF_SUMMARY.json").open("w"),indent=2); print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
