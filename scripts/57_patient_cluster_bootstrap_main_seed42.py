#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
57_patient_cluster_bootstrap_main_seed42.py

Patient-cluster paired bootstrap for the pre-specified main-seed Full-468 OOF
predictions. Patients are sampled with replacement and all their ROI are carried
together.

This analysis quantifies patient-sampling uncertainty conditional on the fixed
training seed. It does not use multiple seeds as independent statistical units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
)

PROB_COLS = [
    "prob_0_normal",
    "prob_1_mild",
    "prob_2_moderate",
    "prob_3_severe",
]

DEFAULT_COMPARISONS = [
    ("ordinal_increment", "Original_CORAL_60_40", "Original_60_40"),
    (
        "risk_auxiliary_increment",
        "Original_CORAL_RISK_G_60_40",
        "Original_CORAL_60_40",
    ),
    (
        "total_proposed_vs_original",
        "Original_CORAL_RISK_G_60_40",
        "Original_60_40",
    ),
    ("branch_risk_auxiliary_increment", "CORAL_RISK_G", "CORAL"),
    ("safety_checkpoint_vs_grade_checkpoint", "CORAL_RISK_S", "CORAL_RISK_G"),
]

METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "qwk",
    "mae",
    "high_risk_recall_argmax",
    "false_clear_rate_argmax",
    "risk_auprc",
    "risk_brier",
    "risk_ece",
    "crossfit_roi_recall",
    "crossfit_roi_specificity",
    "crossfit_roi_flag_rate",
]


def parse_comparison(value: str) -> Tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Use NAME:CANDIDATE:REFERENCE."
        )
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--main-seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--risk-mode", choices=["native", "matched"], default="native")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--comparison",
        action="append",
        type=parse_comparison,
    )
    parser.add_argument(
        "--save-bootstrap-replicates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def normalize_patient(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(4) if text.isdigit() else text
    return series.map(one)


def detect_keys(frame: pd.DataFrame) -> List[str]:
    if "roi_path" in frame.columns:
        return ["roi_path"]
    for keys in [
        ["patient_id", "image_name", "object_index"],
        ["patient_id", "source_image", "object_index"],
    ]:
        if all(column in frame.columns for column in keys):
            return keys
    raise ValueError("Could not detect ROI identity columns.")


def load_variant(
    root: Path,
    seed: int,
    variant: str,
    risk_mode: str,
) -> pd.DataFrame:
    path = (
        root / "oof_predictions" / f"seed_{seed}" / f"{variant}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file:\n  {path}")
    frame = pd.read_csv(path)
    required = ["patient_id", "y_true", "cv_fold", *PROB_COLS]
    risk_column = (
        "risk_score_native" if risk_mode == "native"
        else "risk_score_matched"
    )
    required.append(risk_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    frame = frame.copy()
    frame["patient_id"] = normalize_patient(frame["patient_id"])
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["cv_fold"] = pd.to_numeric(frame["cv_fold"], errors="raise").astype(int)
    for column in PROB_COLS:
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        ).astype(float)
    frame["risk_score"] = pd.to_numeric(
        frame[risk_column], errors="raise"
    ).astype(float)
    probability = frame[PROB_COLS].to_numpy(dtype=float)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{path}: probability rows do not sum to one.")
    frame["y_pred"] = probability.argmax(axis=1)
    return frame


def choose_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    target_recall: float,
) -> float:
    candidates = np.unique(
        np.concatenate([
            np.array([0.0, 1.0]),
            score,
            np.nextafter(score, -np.inf),
        ])
    )
    best = None
    for threshold in np.sort(candidates)[::-1]:
        prediction = score >= threshold
        positive = y_true == 1
        tp = int(np.sum(positive & prediction))
        fn = int(np.sum(positive & ~prediction))
        tn = int(np.sum(~positive & ~prediction))
        fp = int(np.sum(~positive & prediction))
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        flag_rate = (tp + fp) / max(len(y_true), 1)
        if recall + 1e-12 < target_recall:
            continue
        key = (specificity, threshold, -flag_rate)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    if best is None:
        raise RuntimeError(
            f"No threshold reached target recall {target_recall}."
        )
    return best[1]


def attach_crossfit_flags(
    frame: pd.DataFrame,
    folds: List[int],
    target_recall: float,
) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    output = frame.copy()
    output["crossfit_flag"] = False
    details = []

    for fold in folds:
        selection = output.loc[output["cv_fold"] != fold]
        holdout_mask = output["cv_fold"] == fold
        holdout = output.loc[holdout_mask]
        threshold = choose_threshold(
            (selection["y_true"].to_numpy(dtype=int) >= 2).astype(int),
            selection["risk_score"].to_numpy(dtype=float),
            target_recall,
        )
        output.loc[holdout_mask, "crossfit_flag"] = (
            holdout["risk_score"].to_numpy(dtype=float) >= threshold
        )
        details.append({
            "fold": int(fold),
            "threshold": float(threshold),
            "selection_roi_count": int(len(selection)),
            "holdout_roi_count": int(len(holdout)),
        })

    output["crossfit_flag"] = output["crossfit_flag"].astype(bool)
    return output, details


def expected_calibration_error(
    y_true: np.ndarray,
    score: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (score >= edges[index]) & (score <= edges[index + 1])
        else:
            mask = (score >= edges[index]) & (score < edges[index + 1])
        if not mask.any():
            continue
        result += float(mask.mean()) * abs(
            float(score[mask].mean()) - float(y_true[mask].mean())
        )
    return float(result)


def metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    probability = frame[PROB_COLS].to_numpy(dtype=float)
    y_pred = probability.argmax(axis=1)
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    risk_score = frame["risk_score"].to_numpy(dtype=float)
    flag = frame["crossfit_flag"].to_numpy(dtype=bool)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "qwk": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "high_risk_recall_argmax": float(
            np.sum(high_true & high_pred) / max(int(high_true.sum()), 1)
        ),
        "false_clear_rate_argmax": float(
            np.sum(high_true & ~high_pred) / max(int(high_true.sum()), 1)
        ),
        "risk_auprc": float(
            average_precision_score(high_true.astype(int), risk_score)
        ),
        "risk_brier": float(
            np.mean((risk_score - high_true.astype(int)) ** 2)
        ),
        "risk_ece": expected_calibration_error(
            high_true.astype(int), risk_score
        ),
        "crossfit_roi_recall": float(
            np.sum(high_true & flag) / max(int(high_true.sum()), 1)
        ),
        "crossfit_roi_specificity": float(
            np.sum(~high_true & ~flag) / max(int((~high_true).sum()), 1)
        ),
        "crossfit_roi_flag_rate": float(flag.mean()),
    }


def align_pair(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = detect_keys(candidate)
    if keys != detect_keys(reference):
        raise ValueError("ROI identity definitions differ.")
    identity = [*keys, "patient_id", "y_true", "cv_fold"]
    candidate = candidate.sort_values(keys).reset_index(drop=True)
    reference = reference.sort_values(keys).reset_index(drop=True)
    if not candidate[identity].equals(reference[identity]):
        raise ValueError("Candidate/reference OOF identities differ.")
    return candidate, reference


def patient_groups(frame: pd.DataFrame):
    patients = np.array(
        sorted(frame["patient_id"].unique().tolist()),
        dtype=object,
    )
    groups = {
        patient: np.flatnonzero(
            frame["patient_id"].to_numpy() == patient
        )
        for patient in patients
    }
    return patients, groups


def bootstrap_comparison(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    reps: int,
    random_seed: int,
    ci: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate, reference = align_pair(candidate, reference)
    point_candidate = metrics(candidate)
    point_reference = metrics(reference)
    patients, groups = patient_groups(candidate)
    rng = np.random.default_rng(random_seed)

    distributions = {
        metric: np.empty(reps, dtype=float)
        for metric in METRICS
    }
    replicate_rows = []

    for replicate in range(reps):
        sampled = rng.choice(
            patients, size=len(patients), replace=True
        )
        indices = np.concatenate([
            groups[str(patient)] for patient in sampled
        ])
        candidate_metrics = metrics(candidate.iloc[indices])
        reference_metrics = metrics(reference.iloc[indices])
        row = {"bootstrap_rep": int(replicate)}
        for metric in METRICS:
            difference = (
                candidate_metrics[metric]
                - reference_metrics[metric]
            )
            distributions[metric][replicate] = difference
            row[f"{metric}_difference"] = difference
        replicate_rows.append(row)

    alpha = 1.0 - ci
    summary_rows = []
    for metric in METRICS:
        values = distributions[metric]
        nonpositive = (np.sum(values <= 0) + 1) / (len(values) + 1)
        nonnegative = (np.sum(values >= 0) + 1) / (len(values) + 1)
        p_value = float(
            min(1.0, 2.0 * min(nonpositive, nonnegative))
        )
        summary_rows.append({
            "metric": metric,
            "candidate_point": point_candidate[metric],
            "reference_point": point_reference[metric],
            "difference_point": (
                point_candidate[metric] - point_reference[metric]
            ),
            "ci_low": float(
                np.quantile(values, alpha / 2)
            ),
            "ci_high": float(
                np.quantile(values, 1 - alpha / 2)
            ),
            "two_sided_empirical_p": p_value,
            "bootstrap_reps": int(reps),
            "patient_count": int(len(patients)),
            "roi_count": int(len(candidate)),
        })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(replicate_rows),
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    if not 0 < args.ci < 1:
        raise ValueError("--ci must be in (0,1).")

    comparisons = args.comparison or DEFAULT_COMPARISONS
    variants = sorted({
        item
        for _, candidate, reference in comparisons
        for item in [candidate, reference]
    })
    args.out.mkdir(parents=True, exist_ok=True)

    cache = {}
    threshold_rows = []
    for variant in variants:
        frame = load_variant(
            args.analysis_root,
            args.main_seed,
            variant,
            args.risk_mode,
        )
        frame, details = attach_crossfit_flags(
            frame, args.folds, args.target_recall
        )
        cache[variant] = frame
        threshold_rows.extend([
            {"variant": variant, **detail}
            for detail in details
        ])

    all_summary = []
    all_replicates = []
    for index, (
        name, candidate_name, reference_name
    ) in enumerate(comparisons):
        summary, replicates = bootstrap_comparison(
            cache[candidate_name],
            cache[reference_name],
            args.bootstrap_reps,
            args.bootstrap_seed + index * 1000,
            args.ci,
        )
        summary.insert(0, "comparison", name)
        summary.insert(1, "candidate", candidate_name)
        summary.insert(2, "reference", reference_name)
        all_summary.append(summary)

        if args.save_bootstrap_replicates:
            replicates.insert(0, "comparison", name)
            all_replicates.append(replicates)

    summary_frame = pd.concat(all_summary, ignore_index=True)
    summary_frame.to_csv(
        args.out / "patient_cluster_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(threshold_rows).to_csv(
        args.out / "bootstrap_crossfit_threshold_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if all_replicates:
        pd.concat(all_replicates, ignore_index=True).to_csv(
            args.out / "patient_cluster_bootstrap_replicates.csv",
            index=False,
            encoding="utf-8-sig",
        )

    audit = {
        "schema_version": "lss_protocol_v3_main_seed_bootstrap_v1",
        "main_seed": int(args.main_seed),
        "risk_mode": args.risk_mode,
        "patient_resampling_unit": "patient",
        "roi_carried_with_patient": True,
        "crossfit_thresholds_frozen_before_bootstrap": True,
        "target_recall": args.target_recall,
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "separate_test_used": False,
        "interpretation": (
            "Confidence intervals quantify patient-sampling uncertainty "
            "conditional on the pre-specified main training seed."
        ),
    }
    with open(
        args.out / "patient_cluster_bootstrap_audit.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    print("=" * 118)
    print("Protocol V3 main-seed patient-cluster bootstrap completed")
    print("Risk mode:", args.risk_mode)
    print(summary_frame.to_string(index=False))
    print("Output:", args.out)
    print("=" * 118)


if __name__ == "__main__":
    main()
