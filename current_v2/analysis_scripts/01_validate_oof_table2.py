#!/usr/bin/env python3
"""Validate the 14-system pooled OOF table against sanitised AF3 predictions."""
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

KEEP14=['Original DeiT', 'Original CNN', 'CORAL', 'CNN-MSaux', 'CORAL-MSaux', 'CNN + fixed DeiT', 'CORAL + fixed DeiT', 'CNN-MSaux + fixed DeiT', 'CORAL-MSaux + fixed DeiT', 'ConvNeXt-CE', 'ConvNeXt-CORAL', 'ConvNeXt-CORAL-MSaux', 'ConvNeXt-CORAL + fixed DeiT', 'ConvNeXt-CORAL-MSaux + fixed DeiT']

def pick(df,names):
    low={c.lower():c for c in df.columns}
    for n in names:
        if n.lower() in low:return low[n.lower()]
    raise KeyError(f"Missing one of {names}; got {list(df.columns)}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--af3",required=True)
    ap.add_argument("--af2",required=True)
    ap.add_argument("--tol",type=float,default=5e-4)
    a=ap.parse_args()
    oof=pd.read_csv(Path(a.af3)/"OOF_Predictions_14Systems_Long.csv")
    tab=pd.read_csv(Path(a.af2)/"Table2_14Systems.csv")
    sc=pick(oof,["system","System"])
    tc=pick(oof,["y_true","true_grade","label","grade_true"])
    pc=pick(oof,["y_pred","pred_grade","prediction","grade_pred"])
    if list(dict.fromkeys(oof[sc].astype(str)))!=KEEP14:
        raise SystemExit("ROSTER_ORDER_FAIL")
    if len(oof)!=41692:
        raise SystemExit(f"OOF_ROWS_FAIL {len(oof)}")
    results=[]
    for s in KEEP14:
        q=oof[oof[sc].astype(str)==s]
        m=metrics(q[tc].to_numpy(),q[pc].to_numpy())
        m["System"]=s
        results.append(m)
    calc=pd.DataFrame(results)
    sys2=pick(tab,["System","system"])
    merged=tab.merge(calc,left_on=sys2,right_on="System",suffixes=("_public","_calc"))
    checked=0
    for metric in ["Accuracy","Balanced Accuracy","Macro-F1","Weighted-F1","QWK","MAE"]:
        pub=metric+"_public"; cal=metric+"_calc"
        if pub in merged.columns and cal in merged.columns:
            d=np.abs(merged[pub].astype(float)-merged[cal].astype(float))
            if (d>a.tol).any():
                raise SystemExit(f"METRIC_PARITY_FAIL {metric} max={d.max()}")
            checked+=1
    print(f"PASS_AF4_VALIDATE | systems=14 rows=41692 metrics_checked={checked}")
if __name__=="__main__":
    main()
