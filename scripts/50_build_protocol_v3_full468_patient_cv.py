#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
50_build_protocol_v3_full468_patient_cv.py

Build Protocol V3 Full-468 patient-level five-fold cross-validation.

Design
------
- Use every ROI row with a valid grade 0..3 from the supplied classification
  manifest, including the historical development and historical test patients.
- Never treat unannotated cases as Normal.
- Assign every patient to exactly one of five folds.
- Optimize fold balance using patient-level grade counts, high-risk status,
  ROI burden, and optional side/level metadata.
- Preserve the historical source split for audit only.
- Do not use model predictions during fold construction.

Expected current task size
--------------------------
- 468 evaluable patients
- 2978 ROI
- 5 patient-level folds, approximately 94/94/94/93/93 patients

Outputs
-------
patient_protocol_v3_full468.csv
roi_splits_protocol_v3_full468.csv
protocol_v3_fold_balance.csv
protocol_v3_source_split_by_fold.csv
protocol_v3_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PATIENT_CANDIDATES = ["patient_id", "case_id", "patient", "case"]
LABEL_CANDIDATES = ["severity", "y_true", "label", "class_id", "grade"]
SOURCE_SPLIT_CANDIDATES = [
    "protocol_v2_split",
    "split_patient",
    "final_split",
    "split",
]
SIDE_CANDIDATES = ["side", "laterality"]
LEVEL_CANDIDATES = ["level", "ivd_level", "disc_level", "lumbar_level"]
PATH_CANDIDATES = ["roi_path", "path", "image_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--candidate-seeds", type=int, default=20000)
    parser.add_argument(
        "--fold-search-base-seed",
        type=int,
        default=20260719,
        help="Seed used only to search for a balanced fixed fold assignment.",
    )
    parser.add_argument("--expected-patients", type=int, default=468)
    parser.add_argument("--expected-roi", type=int, default=2978)
    return parser.parse_args()


def detect_column(
    frame: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise ValueError(
            f"Could not detect a required column from {candidates}. "
            f"Available columns: {list(frame.columns)}"
        )
    return None


def normalize_patient_id(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(4) if text.isdigit() else text
    return series.map(one)


def build_patient_table(
    roi: pd.DataFrame,
    patient_col: str,
    label_col: str,
    source_split_col: Optional[str],
    side_col: Optional[str],
    level_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    balance_features = [
        "n_roi",
        "n_grade_0",
        "n_grade_1",
        "n_grade_2",
        "n_grade_3",
        "has_grade_0",
        "has_grade_1",
        "has_grade_2",
        "has_grade_3",
        "n_high_risk",
        "has_high_risk",
    ]

    side_values: List[str] = []
    if side_col:
        side_values = sorted(
            roi[side_col].dropna().astype(str).unique().tolist()
        )
        balance_features.extend([f"n_side_{value}" for value in side_values])

    level_values: List[str] = []
    if level_col:
        level_values = sorted(
            roi[level_col].dropna().astype(str).unique().tolist()
        )
        balance_features.extend([f"n_level_{value}" for value in level_values])

    for patient_id, group in roi.groupby(patient_col, sort=True):
        labels = group[label_col].to_numpy(dtype=int)
        row: Dict[str, object] = {
            "patient_id": patient_id,
            "n_roi": int(len(group)),
            "max_grade": int(labels.max()),
            "n_high_risk": int((labels >= 2).sum()),
            "has_high_risk": int((labels >= 2).any()),
        }

        if source_split_col:
            source_values = sorted(
                group[source_split_col].dropna().astype(str).str.lower().unique()
            )
            row["historical_source_split"] = (
                source_values[0] if len(source_values) == 1 else "|".join(source_values)
            )
        else:
            row["historical_source_split"] = "unknown"

        for grade in range(4):
            count = int((labels == grade).sum())
            row[f"n_grade_{grade}"] = count
            row[f"has_grade_{grade}"] = int(count > 0)

        for value in side_values:
            row[f"n_side_{value}"] = int(
                (group[side_col].astype(str) == value).sum()
            )

        for value in level_values:
            row[f"n_level_{value}"] = int(
                (group[level_col].astype(str) == value).sum()
            )

        rows.append(row)

    patient = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    quantiles = min(4, max(int(patient["n_roi"].nunique()), 1))
    patient["roi_count_bin"] = pd.qcut(
        patient["n_roi"],
        q=quantiles,
        labels=False,
        duplicates="drop",
    ).fillna(0).astype(int)

    for value in sorted(patient["roi_count_bin"].unique()):
        column = f"roi_bin_{value}"
        patient[column] = (patient["roi_count_bin"] == value).astype(int)
        balance_features.append(column)

    return patient, balance_features


def balance_score(
    patient: pd.DataFrame,
    assignment: np.ndarray,
    features: List[str],
    folds: int,
) -> float:
    matrix = patient[features].to_numpy(dtype=float)
    target = matrix.sum(axis=0) / folds
    scale = np.maximum(target, 1.0)

    score = 0.0
    for fold in range(folds):
        observed = matrix[assignment == fold].sum(axis=0)
        score += float(np.mean(((observed - target) / scale) ** 2))

    for column, weight in [
        ("n_grade_2", 4.0),
        ("n_grade_3", 4.0),
        ("has_grade_2", 2.5),
        ("has_grade_3", 2.5),
        ("n_high_risk", 3.0),
        ("has_high_risk", 3.0),
        ("n_roi", 1.5),
    ]:
        if column not in patient.columns:
            continue
        values = patient[column].to_numpy(dtype=float)
        target_value = values.sum() / folds
        denom = max(target_value, 1.0)
        score += weight * float(np.mean([
            ((values[assignment == fold].sum() - target_value) / denom) ** 2
            for fold in range(folds)
        ]))

    patient_counts = np.array([(assignment == fold).sum() for fold in range(folds)])
    score += 5.0 * float(np.var(patient_counts))
    return score


def optimize_folds(
    patient: pd.DataFrame,
    features: List[str],
    folds: int,
    candidate_seeds: int,
    base_seed: int,
) -> Tuple[np.ndarray, float, int]:
    strata = patient["max_grade"].astype(int).to_numpy()
    stratum_counts = pd.Series(strata).value_counts()
    if stratum_counts.min() < folds:
        raise ValueError(
            "A max-grade stratum has fewer patients than folds: "
            f"{stratum_counts.to_dict()}"
        )

    dummy = np.zeros((len(patient), 1), dtype=float)
    best_assignment = None
    best_score = float("inf")
    best_seed = None

    for offset in range(candidate_seeds):
        seed = base_seed + offset
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        assignment = np.full(len(patient), -1, dtype=int)
        for fold, (_, holdout) in enumerate(splitter.split(dummy, strata)):
            assignment[holdout] = fold

        score = balance_score(patient, assignment, features, folds)
        if score < best_score:
            best_assignment = assignment.copy()
            best_score = score
            best_seed = seed

    if best_assignment is None or best_seed is None:
        raise RuntimeError("Fold optimization failed.")
    return best_assignment, float(best_score), int(best_seed)


def fold_balance(
    roi: pd.DataFrame,
    patient_col: str,
    label_col: str,
) -> pd.DataFrame:
    rows = []
    for fold, group in roi.groupby("protocol_v3_cv_fold", sort=True):
        labels = group[label_col].astype(int)
        rows.append({
            "fold": int(fold),
            "patient_count": int(group[patient_col].nunique()),
            "roi_count": int(len(group)),
            "normal": int((labels == 0).sum()),
            "mild": int((labels == 1).sum()),
            "moderate": int((labels == 2).sum()),
            "severe": int((labels == 3).sum()),
            "high_risk": int((labels >= 2).sum()),
            "high_risk_patient_count": int(
                group.loc[labels >= 2, patient_col].nunique()
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    roi = pd.read_csv(args.roi_manifest)
    patient_col = detect_column(roi, PATIENT_CANDIDATES)
    label_col = detect_column(roi, LABEL_CANDIDATES)
    source_split_col = detect_column(
        roi, SOURCE_SPLIT_CANDIDATES, required=False
    )
    side_col = detect_column(roi, SIDE_CANDIDATES, required=False)
    level_col = detect_column(roi, LEVEL_CANDIDATES, required=False)
    path_col = detect_column(roi, PATH_CANDIDATES, required=False)

    roi[patient_col] = normalize_patient_id(roi[patient_col])
    roi[label_col] = pd.to_numeric(roi[label_col], errors="raise").astype(int)
    invalid = sorted(set(roi[label_col]) - {0, 1, 2, 3})
    if invalid:
        raise ValueError(f"Labels outside 0..3: {invalid}")

    if path_col and roi[path_col].duplicated().any():
        duplicates = int(roi[path_col].duplicated().sum())
        raise ValueError(f"Found {duplicates} duplicated ROI paths.")

    patient, features = build_patient_table(
        roi,
        patient_col,
        label_col,
        source_split_col,
        side_col,
        level_col,
    )

    if len(patient) != args.expected_patients:
        raise ValueError(
            f"Expected {args.expected_patients} patients, found {len(patient)}."
        )
    if len(roi) != args.expected_roi:
        raise ValueError(
            f"Expected {args.expected_roi} ROI, found {len(roi)}."
        )

    assignment, objective, selected_seed = optimize_folds(
        patient,
        features,
        args.folds,
        args.candidate_seeds,
        args.fold_search_base_seed,
    )
    patient["protocol_v3_cv_fold"] = assignment
    patient["protocol_v3_split"] = "full_cohort_cv"

    mapping = patient.set_index("patient_id")["protocol_v3_cv_fold"]
    roi_out = roi.copy()
    roi_out["historical_source_split"] = (
        roi_out[source_split_col].astype(str)
        if source_split_col
        else "unknown"
    )
    roi_out["protocol_v3_split"] = "full_cohort_cv"
    roi_out["protocol_v3_cv_fold"] = (
        roi_out[patient_col].map(mapping).astype(int)
    )

    if roi_out["protocol_v3_cv_fold"].isna().any():
        raise RuntimeError("At least one ROI could not be mapped to a fold.")

    patient.to_csv(
        args.out / "patient_protocol_v3_full468.csv",
        index=False,
        encoding="utf-8-sig",
    )
    roi_out.to_csv(
        args.out / "roi_splits_protocol_v3_full468.csv",
        index=False,
        encoding="utf-8-sig",
    )

    balance = fold_balance(roi_out, patient_col, label_col)
    balance.to_csv(
        args.out / "protocol_v3_fold_balance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if source_split_col:
        source_table = pd.crosstab(
            patient["protocol_v3_cv_fold"],
            patient["historical_source_split"],
        ).reset_index()
    else:
        source_table = pd.DataFrame()
    source_table.to_csv(
        args.out / "protocol_v3_source_split_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overlap_count = int(
        patient.groupby("patient_id")["protocol_v3_cv_fold"].nunique().gt(1).sum()
    )
    summary = {
        "schema_version": "lss_protocol_v3_full468_v1",
        "input": str(args.roi_manifest),
        "locked_test_used": False,
        "design": "retrospective full-cohort patient-level five-fold CV",
        "patient_count": int(len(patient)),
        "roi_count": int(len(roi_out)),
        "folds": int(args.folds),
        "fold_search_base_seed": int(args.fold_search_base_seed),
        "candidate_fold_assignments_evaluated": int(args.candidate_seeds),
        "selected_fold_assignment_seed": int(selected_seed),
        "balance_objective": float(objective),
        "model_predictions_used_for_fold_selection": False,
        "patient_overlap_across_folds": overlap_count,
        "detected_columns": {
            "patient": patient_col,
            "label": label_col,
            "source_split": source_split_col,
            "side": side_col,
            "level": level_col,
            "path": path_col,
        },
        "fold_patient_counts": {
            str(row["fold"]): int(row["patient_count"])
            for _, row in balance.iterrows()
        },
        "fold_roi_counts": {
            str(row["fold"]): int(row["roi_count"])
            for _, row in balance.iterrows()
        },
        "notes": [
            "Historical development/test labels are preserved only for audit.",
            "All 468 evaluable patients participate in the same five-fold CV.",
            "The 32 original cases without usable four-grade ROI labels remain outside the supervised classification task.",
            "Model seeds are independent of the fold-assignment search seed.",
        ],
    }
    with open(
        args.out / "protocol_v3_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("Protocol V3 Full-468 created")
    print(f"Patients={len(patient)} | ROI={len(roi_out)}")
    print(f"Selected fold-assignment seed={selected_seed}")
    print(balance.to_string(index=False))
    print(f"Output: {args.out}")
    print("=" * 100)


if __name__ == "__main__":
    main()
