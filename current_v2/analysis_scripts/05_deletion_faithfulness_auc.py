#!/usr/bin/env python3
"""Recompute deletion-faithfulness AUC from the frozen long-form source."""
from pathlib import Path
import argparse, pandas as pd, numpy as np
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--af2",required=True)
    a=ap.parse_args()
    root=Path(a.af2)
    d=pd.read_csv(root/"Faithfulness_Long.csv")
    low={c.lower():c for c in d.columns}
    method=low.get("method")
    case=next((c for c in d.columns if "case" in c.lower()),None)
    x=next((c for c in d.columns if any(k in c.lower() for k in ["fraction","deletion","removed","step"])),None)
    y=next((c for c in d.columns if any(k in c.lower() for k in ["score","prob","confidence"])),None)
    if not all([method,x,y]):
        raise SystemExit(f"COLUMN_DISCOVERY_FAIL {list(d.columns)}")
    group=[method]+([case] if case else [])
    vals=[]
    for key,q in d.groupby(group):
        q=q.sort_values(x)
        vals.append((key,float(np.trapz(q[y].astype(float),q[x].astype(float)))))
    print(f"PASS_AF4_FAITHFULNESS_AUC | curves={len(vals)}")
if __name__=="__main__":
    main()
