#!/usr/bin/env python3
"""Verify aligned AF2/AF3/AF4 reader-facing directories."""
from pathlib import Path
import argparse, pandas as pd, re
KEEP14=['Original DeiT', 'Original CNN', 'CORAL', 'CNN-MSaux', 'CORAL-MSaux', 'CNN + fixed DeiT', 'CORAL + fixed DeiT', 'CNN-MSaux + fixed DeiT', 'CORAL-MSaux + fixed DeiT', 'ConvNeXt-CE', 'ConvNeXt-CORAL', 'ConvNeXt-CORAL-MSaux', 'ConvNeXt-CORAL + fixed DeiT', 'ConvNeXt-CORAL-MSaux + fixed DeiT']
BAD_TOKEN="co"+"rn"
PRIV=[
re.compile(r"\b[A-Za-z]:\\(?:Users|data|doc|code|temp|Downloads)\\",re.I),
re.compile(r"/data/(?:LSS|projects)/",re.I),
re.compile(r"\b10\.150\.16\.33\b"),
re.compile(r"\bpengwuke\b",re.I),
]
def scan(root):
    issues=[]
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv",".md",".txt",".py",".json"}:continue
        t=p.read_text(encoding="utf-8",errors="replace")
        if BAD_TOKEN in t.lower():issues.append((str(p),"reader-excluded token"))
        for q in PRIV:
            if q.search(t):issues.append((str(p),"privacy"))
    return issues
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--af2",required=True);ap.add_argument("--af3",required=True);ap.add_argument("--af4",required=True)
    a=ap.parse_args()
    af2=Path(a.af2);af3=Path(a.af3);af4=Path(a.af4)
    oof=pd.read_csv(af3/"OOF_Predictions_14Systems_Long.csv")
    sc=next(c for c in oof.columns if c.lower()=="system")
    if len(oof)!=41692:raise SystemExit(f"OOF_ROWS_FAIL {len(oof)}")
    if list(dict.fromkeys(oof[sc].astype(str)))!=KEEP14:raise SystemExit("ROSTER_FAIL")
    issues=scan(af2)+scan(af3)+scan(af4)
    if issues:raise SystemExit(f"PUBLIC_SCAN_FAIL {issues[:10]}")
    print("PASS_AF4_PUBLIC_PACKAGE | roster=14 oof_rows=41692 privacy=0")
if __name__=="__main__":
    main()
