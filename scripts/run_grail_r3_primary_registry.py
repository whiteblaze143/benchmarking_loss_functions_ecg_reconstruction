#!/usr/bin/env python3
"""R3 Pass A: exact primary probes and permanent training-only C registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ub", "b0", "b1", "model_001", "model_101", "model_m"]
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def metric_row(y, score):
    fpr, tpr, _ = roc_curve(y, score); spec = 1-fpr
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "sens_at_95spec": float(np.max(tpr[fpr <= .05])),
        "spec_at_95sens": float(np.max(spec[tpr >= .95])) if np.any(tpr >= .95) else 0.0,
    }


def fit_one(model, tier, concept, y, x6, x7, x17, x8, tol):
    y = y.astype(np.int8)
    y6, y7, y17, y8 = y[:len(x6)], None, None, None
    # Labels are passed already concatenated in split order.
    n6, n7, n17 = len(x6), len(x7), len(x17)
    y6, y7, y17, y8 = y[:n6], y[n6:n6+n7], y[n6+n7:n6+n7+n17], y[n6+n7+n17:]
    if any(len(np.unique(v)) < 2 for v in (y6, y7, y17, y8)):
        return {"model":model,"concept":concept,"tier":tier,"converged":False,"invalid_reason":"single_class_split"}
    clf = LogisticRegression(penalty="l2", solver="lbfgs", warm_start=True,
                             max_iter=5000, tol=tol, random_state=42)
    candidates=[]
    for c in CS:
        clf.set_params(C=c)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning); clf.fit(x6,y6)
        conv = not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        if conv:
            candidates.append((roc_auc_score(y7,clf.predict_proba(x7)[:,1]),c,int(clf.n_iter_[0])))
    if not candidates:
        return {"model":model,"concept":concept,"tier":tier,"converged":False,"invalid_reason":"C_grid_nonconvergence"}
    fold7_auc, selected_c, selection_n_iter = max(candidates,key=lambda q:(q[0],-CS.index(q[1])))
    final=LogisticRegression(C=selected_c,penalty="l2",solver="lbfgs",max_iter=20000,
                             tol=tol,random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning); final.fit(x17,y17)
    conv=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    if not conv:
        return {"model":model,"concept":concept,"tier":tier,"selected_C":selected_c,
                "fold7_auroc":fold7_auc,"converged":False,"invalid_reason":"final_nonconvergence"}
    score=final.predict_proba(x8)[:,1]
    return {"model":model,"concept":concept,"tier":tier,"selected_C":selected_c,
            "fold7_auroc":float(fold7_auc),"selection_n_iter":selection_n_iter,
            "final_n_iter":int(final.n_iter_[0]),"converged":True,
            "n_positive":int(y8.sum()),"n_negative":int((1-y8).sum()),**metric_row(y8,score)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--tol",type=float,default=1e-7)
    ap.add_argument("--embedding-dir",type=Path,default=ROOT/"results/grail_v2/r3_linear_probe_audit/embeddings")
    ap.add_argument("--output-dir",type=Path,default=ROOT/"results/grail_v2/r3_linear_probe_audit")
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/"configs/ptbxl_concept_tiers.yaml").read_text())
    tier={c:"P1" for c in cfg["tier_p1_codes"]}; tier.update({c:"P2" for c in cfg["tier_p2_codes"]}); tier.update({c:"P3" for c in cfg["tier_p3_codes"]})
    all_rows=[]
    for model in MODELS:
        path=args.embedding_dir/model/"embeddings_folds1_8.npz"
        with np.load(path) as d:
            z=d["z"].astype(np.float64); folds=d["strat_fold"]
            anchors=d["anchor_labels"]; probes=d["probe_labels"]
        m6=folds<=6; m7=folds==7; m17=folds<=7; m8=folds==8
        s6=StandardScaler().fit(z[m6]); s17=StandardScaler().fit(z[m17])
        x6=s6.transform(z[m6]); x7=s6.transform(z[m7]); x17=s17.transform(z[m17]); x8=s17.transform(z[m8])
        specs=[("anchor",c,anchors[:,i]) for i,c in enumerate(cfg["all_anchor_codes"])]
        specs += [(tier[c],c,probes[:,i]) for i,c in enumerate(cfg["all_probe_codes"])]
        jobs=[]
        for t,c,yall in specs:
            packed=np.concatenate([yall[m6],yall[m7],yall[m17],yall[m8]])
            jobs.append(joblib.delayed(fit_one)(model,t,c,packed,x6,x7,x17,x8,args.tol))
        rows=joblib.Parallel(n_jobs=args.workers,backend="loky",verbose=10)(jobs)
        all_rows.extend(rows)
        partial=pd.DataFrame(all_rows)
        tmp=args.output_dir/"probe_regularization_registry.partial.tmp.parquet"
        partial.to_parquet(tmp,index=False); tmp.replace(args.output_dir/"probe_regularization_registry.partial.parquet")
    df=pd.DataFrame(all_rows)
    tmp=args.output_dir/"probe_regularization_registry.tmp.parquet"
    df.to_parquet(tmp,index=False); tmp.replace(args.output_dir/"probe_regularization_registry.parquet")
    manifest={"status":"complete","pass":"A_primary_registry","models":MODELS,"C_grid":CS,
              "fit_folds":[1,2,3,4,5,6],"selection_fold":7,"refit_folds":[1,2,3,4,5,6,7],
              "evaluation_fold":8,"tol":args.tol,"workers":args.workers,"blas_threads_per_worker":1,
              "fold8_used_for_selection":False}
    (args.output_dir/"primary_registry_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("R3 Pass A complete",flush=True)


if __name__=="__main__": main()
