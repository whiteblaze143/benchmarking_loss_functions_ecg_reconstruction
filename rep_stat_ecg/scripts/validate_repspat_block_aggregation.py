"""Real-data regression gate for exact raw vs block-aggregated repSpat MMD²."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from rep_stat_ecg.scripts.discover_repeated_motifs import assign_domains
from rep_stat_ecg.src.motifs.mmd import (biased_mmd2_batch_from_block_sums,
    build_attribute_blocks, compute_n_blocks, generate_repspat_assignments,
    precompute_block_kernel_sums, raw_biased_mmd2_for_assignments,
    count_permutation_exceedances)

def main():
    root=Path("refine-logs/qvcg"); stats=pd.read_parquet(root/"VCG_DOMAIN_STATS.parquet")
    registry=pd.read_parquet(root/"VCG_SPATIAL_DOMAIN_REGISTRY.parquet")
    df=assign_domains(pd.read_parquet(root/"VCG_MICROSTATE_TABLE.parquet"),registry,root/"VCG_COORDINATE_STANDARDIZER.json")
    domains=list(map(int,stats.nsmallest(5,"num_microstates").domain_id)); reports=[]
    for pair_index,(d1,d2) in enumerate(combinations(domains,2)):
        x=df.loc[df.domain_id==d1,["s","rho","kappa"]].to_numpy(dtype=np.float64)
        y=df.loc[df.domain_id==d2,["s","rho","kappa"]].to_numpy(dtype=np.float64)
        bx=build_attribute_blocks(x,compute_n_blocks(len(x),10),42+d1)
        by=build_attribute_blocks(y,compute_n_blocks(len(y),10),42+d2)
        blocks=bx+by; S,sizes,_=precompute_block_kernel_sums(bx,by,"IMQ",1.0)
        obs=np.r_[np.ones(len(bx)),np.zeros(len(by))][None,:]
        Z=generate_repspat_assignments(sizes,min(len(x),len(y)),99,np.random.RandomState(9000+pair_index))
        assignments=np.vstack([obs,Z]); raw=raw_biased_mmd2_for_assignments(blocks,assignments)
        agg=biased_mmd2_batch_from_block_sums(S,sizes,assignments)
        max_error=float(np.max(np.abs(raw-agg))); raw_ex=count_permutation_exceedances(raw[1:],raw[0]); agg_ex=count_permutation_exceedances(agg[1:],agg[0])
        if max_error>=1e-10 or raw_ex!=agg_ex: raise AssertionError((d1,d2,max_error,raw_ex,agg_ex))
        reports.append({"domain_i":d1,"domain_j":d2,"n_i":len(x),"n_j":len(y),"max_abs_error":max_error,"n_exceed_raw":raw_ex,"n_exceed_block":agg_ex,"p_value":(1+raw_ex)/100})
    output={"status":"PASS","tolerance":1e-10,"n_real_pairs":len(reports),"n_fixed_permutations_each":99,"pairs":reports}
    json.dump(output,(root/"REPSPAT_BLOCK_AGGREGATION_REGRESSION.json").open("w"),indent=2)
    print(json.dumps(output,indent=2))
if __name__=="__main__": main()
