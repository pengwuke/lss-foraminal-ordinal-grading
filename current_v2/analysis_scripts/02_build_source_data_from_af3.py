#!/usr/bin/env python3
"""Rebuild pooled metric and confusion source tables from AF3 without training/inference."""
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
    raise KeyError(f"Missing {names}; got {list(df.columns)}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--af3",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    oof=pd.read_csv(Path(a.af3)/"OOF_Predictions_14Systems_Long.csv")
    sc=pick(oof,["system","System"])
    tc=pick(oof,["y_true","true_grade","label","grade_true"])
    pc=pick(oof,["y_pred","pred_grade","prediction","grade_pred"])
    rows=[];conf=[]
    for s in KEEP14:
        q=oof[oof[sc].astype(str)==s]
        m=metrics(q[tc],q[pc]);m["System"]=s;rows.append(m)
        for t in range(4):
            for p in range(4):
                conf.append({"System":s,"TrueGrade":t,"PredGrade":p,
                             "Count":int(((q[tc]==t)&(q[pc]==p)).sum())})
    pd.DataFrame(rows)[["System","Accuracy","Balanced Accuracy","Macro-F1","Weighted-F1","QWK","MAE"]].to_csv(out/"Table2_14Systems.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"Figure3_SourceData.csv",index=False)
    pd.DataFrame(conf).to_csv(out/"Figure6_Confusion_SourceData.csv",index=False)
    print("PASS_AF4_BUILD_SOURCE_DATA | systems=14")
if __name__=="__main__":
    main()
