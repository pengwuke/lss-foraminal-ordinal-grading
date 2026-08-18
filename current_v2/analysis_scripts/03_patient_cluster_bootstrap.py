#!/usr/bin/env python3
"""Paired patient-cluster bootstrap for the frozen reader-facing five comparators.

Provided for reproducibility. The submission build itself never executes this script.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np

import numpy as np

def qwk(y_true,y_pred,n_classes=4):
    y_true=np.asarray(y_true,dtype=int)
    y_pred=np.asarray(y_pred,dtype=int)
    O=np.zeros((n_classes,n_classes),dtype=float)
    for a,b in zip(y_true,y_pred):
        O[a,b]+=1
    act=np.bincount(y_true,minlength=n_classes).astype(float)
    pred=np.bincount(y_pred,minlength=n_classes).astype(float)
    E=np.outer(act,pred)/max(1.0,len(y_true))
    W=np.zeros_like(O)
    den=float((n_classes-1)**2)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i,j]=((i-j)**2)/den
    d=(W*E).sum()
    return 1.0-(W*O).sum()/d if d else 1.0

def metrics(y_true,y_pred):
    y_true=np.asarray(y_true,dtype=int)
    y_pred=np.asarray(y_pred,dtype=int)
    acc=float((y_true==y_pred).mean())
    recalls=[]; f1s=[]; supports=[]
    for c in range(4):
        tp=((y_true==c)&(y_pred==c)).sum()
        fn=((y_true==c)&(y_pred!=c)).sum()
        fp=((y_true!=c)&(y_pred==c)).sum()
        rec=tp/(tp+fn) if tp+fn else 0.0
        pre=tp/(tp+fp) if tp+fp else 0.0
        f1=2*pre*rec/(pre+rec) if pre+rec else 0.0
        recalls.append(rec); f1s.append(f1); supports.append((y_true==c).sum())
    return {
        "Accuracy":acc,
        "Balanced Accuracy":float(np.mean(recalls)),
        "Macro-F1":float(np.mean(f1s)),
        "Weighted-F1":float(np.average(f1s,weights=supports)),
        "QWK":float(qwk(y_true,y_pred)),
        "MAE":float(np.abs(y_true-y_pred).mean()),
    }

REFERENCE="CORAL-MSaux + fixed DeiT"
COMPARATORS=[
"CNN + fixed DeiT",
"CORAL + fixed DeiT",
"CNN-MSaux + fixed DeiT",
"ConvNeXt-CORAL + fixed DeiT",
"ConvNeXt-CORAL-MSaux + fixed DeiT",
]

def pick(df,names):
    low={c.lower():c for c in df.columns}
    for n in names:
        if n.lower() in low:return low[n.lower()]
    raise KeyError(f"Missing {names}; got {list(df.columns)}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--af3",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--reps",type=int,default=20000)
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    oof=pd.read_csv(Path(a.af3)/"OOF_Predictions_14Systems_Long.csv")
    sc=pick(oof,["system","System"])
    pat=pick(oof,["patient_id","patient","PatientID","patient"])
    tc=pick(oof,["y_true","true_grade","label","grade_true"])
    pc=pick(oof,["y_pred","pred_grade","prediction","grade_pred"])
    key=None
    for k in ["case_id","roi_id","sample_id","case_uid","roi_uid"]:
        if k in oof.columns:
            key=k;break
    preds={};base=None
    for s in [REFERENCE]+COMPARATORS:
        q=oof[oof[sc].astype(str)==s].copy()
        q=q.sort_values([pat]+([key] if key else []),kind="stable")
        if base is None:base=q[[pat,tc]].reset_index(drop=True)
        preds[s]=q[pc].to_numpy(dtype=int)
    y=base[tc].to_numpy(dtype=int)
    pats=base[pat].astype(str).to_numpy()
    uniq=np.array(sorted(set(pats)))
    idx_by={p:np.where(pats==p)[0] for p in uniq}
    rng=np.random.default_rng(a.seed)
    point={s:metrics(y,preds[s]) for s in [REFERENCE]+COMPARATORS}
    metrics_order=["Accuracy","Balanced Accuracy","Macro-F1","Weighted-F1","QWK","MAE"]
    dist={(c,m):[] for c in COMPARATORS for m in metrics_order}
    for _ in range(a.reps):
        sampled=rng.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([idx_by[p] for p in sampled])
        refm=metrics(y[idx],preds[REFERENCE][idx])
        for c in COMPARATORS:
            cm=metrics(y[idx],preds[c][idx])
            for m in metrics_order:
                d=(cm[m]-refm[m]) if m=="MAE" else (refm[m]-cm[m])
                dist[(c,m)].append(d)
    rows=[]
    for c in COMPARATORS:
        for m in metrics_order:
            arr=np.asarray(dist[(c,m)])
            pdiff=(point[c][m]-point[REFERENCE][m]) if m=="MAE" else (point[REFERENCE][m]-point[c][m])
            rows.append({
                "Reference":REFERENCE,"Comparator":c,"Metric":m,
                "Delta_reference_better":pdiff,
                "CI_low":float(np.quantile(arr,.025)),
                "CI_high":float(np.quantile(arr,.975)),
                "Probability_reference_better":float((arr>0).mean()),
                "Replicates":a.reps,"Seed":a.seed
            })
    pd.DataFrame(rows).to_csv(a.out,index=False)
    print(f"PASS_AF4_BOOTSTRAP | rows={len(rows)} reps={a.reps} seed={a.seed}")
if __name__=="__main__":
    main()
