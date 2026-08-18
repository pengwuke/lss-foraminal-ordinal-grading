#!/usr/bin/env python3
"""Audit frozen interpretability case locks; never generates new attribution maps."""
from pathlib import Path
import argparse, pandas as pd
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--af2",required=True)
    a=ap.parse_args()
    root=Path(a.af2)
    all20=pd.read_csv(root/"Interpretability_20CaseLock.csv")
    main6=pd.read_csv(root/"Figure7_Main6_CaseLock.csv")
    if len(all20)!=20: raise SystemExit(f"CASE20_ROWS_FAIL {len(all20)}")
    if len(main6)!=6: raise SystemExit(f"MAIN6_ROWS_FAIL {len(main6)}")
    print("PASS_AF4_INTERPRETABILITY_CASE_AUDIT | locked20=20 displayed6=6")
if __name__=="__main__":
    main()
