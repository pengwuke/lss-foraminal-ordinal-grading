#!/usr/bin/env python3
"""Generate portable numerical visualisations from frozen reader-facing AF2 source data.

These are reproducibility views, not replacements for the journal-layout manuscript figures.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--af2",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    root=Path(a.af2);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)

    f3=pd.read_csv(root/"Figure3_SourceData.csv")
    sc=next(c for c in f3.columns if c.lower()=="system")
    mets=[c for c in f3.columns if c!=sc and pd.api.types.is_numeric_dtype(f3[c])][:6]
    fig,ax=plt.subplots(figsize=(9,5))
    im=ax.imshow(f3[mets].to_numpy(float),aspect="auto")
    ax.set_yticks(range(len(f3)));ax.set_yticklabels(f3[sc],fontsize=7)
    ax.set_xticks(range(len(mets)));ax.set_xticklabels(mets,rotation=30,ha="right")
    fig.colorbar(im,ax=ax);fig.tight_layout()
    fig.savefig(out/"Figure3_repro.png",dpi=200);plt.close(fig)

    f5=pd.read_csv(root/"Figure5_20kBootstrap.csv")
    num=[c for c in f5.columns if pd.api.types.is_numeric_dtype(f5[c])]
    fig,ax=plt.subplots(figsize=(8,5))
    if num:ax.plot(np.arange(len(f5)),f5[num[0]].to_numpy())
    ax.set_title("Frozen bootstrap source-data view")
    fig.tight_layout();fig.savefig(out/"Figure5_repro.png",dpi=200);plt.close(fig)
    print("PASS_AF4_MAKE_FIGURES | numerical reproducibility views written")
if __name__=="__main__":
    main()
