#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
61_aggregate_optional_seed_sensitivity_branch.py

Supplementary optimization-seed sensitivity analysis for CORAL vs CORAL-Risk.
No p-values are produced. The 15 seed-fold rows are not treated as independent.
One pooled Full-468 OOF result is produced per seed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROB_COLS = [
    "prob_0_normal",
    "prob_1_mild",
    "prob_2_moderate",
    "prob_3_severe",
]


def load_helper():
    path = Path(__file__).with_name("23_aggregate_ordinal_baselines_multiseed.py")
    spec = importlib.util.spec_from_file_location("lss_metrics_helper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helper()


def parse_seed_root(value: str) -> Tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use SEED=PATH.")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-root",
        action="append",
        type=parse_seed_root,
        required=True,
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--expected-patients", type=int, default=468)
    parser.add_argument("--expected-roi", type=int, default=2978)
    return parser.parse_args()


def normalize_patient(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(4) if text.isdigit() else text
    return series.map(one)


def keys(frame: pd.DataFrame) -> List[str]:
    if "roi_path" in frame.columns:
        return ["roi_path"]
    for candidate in [
        ["patient_id", "image_name", "object_index"],
        ["patient_id", "source_image", "object_index"],
    ]:
        if all(column in frame.columns for column in candidate):
            return candidate
    raise ValueError("Could not detect ROI identity columns.")


def load_prediction(
    path: Path,
    fold: int,
    require_aux: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file:\n  {path}")
    frame = pd.read_csv(path)
    required = ["patient_id", "y_true", *PROB_COLS]
    if require_aux:
        required.append("risk_prob_aux")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    frame = frame.copy()
    frame["patient_id"] = normalize_patient(frame["patient_id"])
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    for column in PROB_COLS:
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        ).astype(float)
    if require_aux:
        frame["risk_prob_aux"] = pd.to_numeric(
            frame["risk_prob_aux"], errors="raise"
        ).astype(float)
    frame["cv_fold"] = int(fold)
    return frame


def normalize_grade(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    probability = output[PROB_COLS].to_numpy(dtype=float)
    probability = probability / np.clip(
        probability.sum(axis=1, keepdims=True), 1e-12, None
    )
    for index, column in enumerate(PROB_COLS):
        output[column] = probability[:, index]
    output["y_pred"] = probability.argmax(axis=1)
    return output


def summarize(
    frame: pd.DataFrame,
    risk_column: str,
    folds: List[int],
    target_recall: float,
) -> Dict[str, float]:
    work = normalize_grade(frame)
    work["risk_score"] = pd.to_numeric(
        work[risk_column], errors="raise"
    ).astype(float)
    summary, _ = H.summarize(work, folds, target_recall)
    return summary


def main() -> None:
    args = parse_args()
    roots = dict(args.seed_root)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed, root in sorted(roots.items()):
        coral_folds = []
        risk_folds = []

        for fold in args.folds:
            if seed == 42 and root.name.startswith("main_seed_"):
                coral_path = root / "CORAL" / f"fold_{fold}" / "val_predictions_best_loss.csv"
                risk_path = root / "CORAL_RISK_lambda0p50" / f"fold_{fold}" / "val_predictions_best_ordinal_loss.csv"
            else:
                coral_path = root / "CORAL" / f"fold_{fold}" / "val_predictions_best_loss.csv"
                risk_path = root / "CORAL_RISK_lambda0p50" / f"fold_{fold}" / "val_predictions_best_ordinal_loss.csv"

            coral_folds.append(load_prediction(coral_path, fold, require_aux=False))
            risk_folds.append(load_prediction(risk_path, fold, require_aux=True))

        coral = pd.concat(coral_folds, ignore_index=True)
        risk = pd.concat(risk_folds, ignore_index=True)
        roi_keys = keys(coral)
        coral = coral.sort_values(roi_keys).reset_index(drop=True)
        risk = risk.sort_values(roi_keys).reset_index(drop=True)
        identity = [*roi_keys, "patient_id", "y_true", "cv_fold"]
        if not coral[identity].equals(risk[identity]):
            raise ValueError(f"Seed {seed}: CORAL/CORAL-Risk identities differ.")

        if len(coral) != args.expected_roi:
            raise ValueError(
                f"Seed {seed}: expected {args.expected_roi} ROI, found {len(coral)}."
            )
        if coral["patient_id"].nunique() != args.expected_patients:
            raise ValueError(
                f"Seed {seed}: patient count mismatch."
            )

        coral["risk_score_matched"] = coral[PROB_COLS[2:]].sum(axis=1)
        risk["risk_score_matched"] = risk[PROB_COLS[2:]].sum(axis=1)

        coral_summary = summarize(
            coral, "risk_score_matched", args.folds, args.target_recall
        )
        risk_native_summary = summarize(
            risk, "risk_prob_aux", args.folds, args.target_recall
        )
        risk_matched_summary = summarize(
            risk, "risk_score_matched", args.folds, args.target_recall
        )

        row = {
            "seed": int(seed),
            **{f"coral_{k}": v for k, v in coral_summary.items()},
            **{f"risk_native_{k}": v for k, v in risk_native_summary.items()},
            **{f"risk_matched_{k}": v for k, v in risk_matched_summary.items()},
        }
        for metric in coral_summary:
            row[f"native_difference_{metric}"] = (
                risk_native_summary[metric] - coral_summary[metric]
            )
            row[f"matched_difference_{metric}"] = (
                risk_matched_summary[metric] - coral_summary[metric]
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(
        args.out / "seed_sensitivity_branch_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    numeric = [
        column for column in frame.columns
        if column != "seed"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary = {"seed_count": int(len(frame))}
    for column in numeric:
        values = frame[column].dropna()
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_sd"] = (
            float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        )
        summary[f"{column}_min"] = float(values.min())
        summary[f"{column}_max"] = float(values.max())
    summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(
        args.out / "seed_sensitivity_branch_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "schema_version": "lss_protocol_v3_optional_seed_sensitivity_v1",
        "role": "supplementary",
        "seeds": sorted(roots),
        "patient_count": args.expected_patients,
        "roi_count": args.expected_roi,
        "lambda": 0.50,
        "separate_test_used": False,
        "statistical_rule": (
            "Report one pooled OOF result per seed and descriptive mean/SD/range. "
            "Do not use seed-fold rows as independent observations and do not "
            "base clinical p-values on three seeds."
        ),
    }
    with open(
        args.out / "seed_sensitivity_audit.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    print("=" * 116)
    print("OPTIONAL branch seed-sensitivity aggregation completed")
    print(frame.to_string(index=False))
    print("Output:", args.out)
    print("=" * 116)


if __name__ == "__main__":
    main()
