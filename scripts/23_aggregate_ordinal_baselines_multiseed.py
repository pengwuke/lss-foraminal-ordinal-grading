#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
23_aggregate_ordinal_baselines_multiseed.py

Aggregate CORAL/CORN Protocol V2 OOF predictions across seeds and compare them
with the existing C0, M1-G, R2-CV, and M1CV-G baselines.

Branch-level models
-------------------
C0
M1_G
CORAL
CORN

System-level models
-------------------
R2CV       = 0.25 DeiT + 0.75 C0
M1CV_G     = 0.25 DeiT + 0.75 M1-G grade probabilities
CORALCV    = 0.25 DeiT + 0.75 CORAL
CORNCV     = 0.25 DeiT + 0.75 CORN

Risk scores
-----------
- C0, CORAL, CORN, R2CV, CORALCV, CORNCV:
  P(Moderate) + P(Severe)
- M1_G, M1CV_G:
  M1 auxiliary risk probability

The locked test set is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROB_COLS = [
    "prob_0_normal",
    "prob_1_mild",
    "prob_2_moderate",
    "prob_3_severe",
]

VARIANTS = [
    "C0",
    "M1_G",
    "CORAL",
    "CORN",
    "R2CV",
    "M1CV_G",
    "CORALCV",
    "CORNCV",
]

COMPARISONS = [
    ("CORAL_vs_C0", "CORAL", "C0"),
    ("CORN_vs_C0", "CORN", "C0"),
    ("M1_G_vs_CORAL", "M1_G", "CORAL"),
    ("M1_G_vs_CORN", "M1_G", "CORN"),
    ("CORALCV_vs_R2CV", "CORALCV", "R2CV"),
    ("CORNCV_vs_R2CV", "CORNCV", "R2CV"),
    ("M1CV_G_vs_CORALCV", "M1CV_G", "CORALCV"),
    ("M1CV_G_vs_CORNCV", "M1CV_G", "CORNCV"),
]

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "qwk",
    "mae",
    "within_one_grade_accuracy",
    "undergrade_by_2plus_rate",
    "overgrade_by_2plus_rate",
    "normal_recall",
    "mild_recall",
    "moderate_recall",
    "severe_recall",
    "high_risk_recall_argmax",
    "false_clear_rate_argmax",
    "moderate_false_clear_rate",
    "severe_false_clear_rate",
    "risk_auroc",
    "risk_auprc",
    "risk_brier",
    "risk_ece",
    "crossfit_roi_recall_at_target",
    "crossfit_roi_specificity_at_target",
    "crossfit_roi_flag_rate_at_target",
    "crossfit_patient_recall_at_target",
    "crossfit_patient_specificity_at_target",
    "crossfit_patient_flag_rate_at_target",
]


def parse_seed_root(value: str) -> Tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use SEED=PATH.")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-root",
        action="append",
        type=parse_seed_root,
        required=True,
        help="Repeat as SEED=PATH.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--deit-weight", type=float, default=0.25)
    parser.add_argument("--cnn-weight", type=float, default=0.75)
    parser.add_argument("--expected-roi-count", type=int, default=2359)
    parser.add_argument("--expected-patient-count", type=int, default=374)
    return parser.parse_args()


def normalize_patient_id(series: pd.Series) -> pd.Series:
    def normalize(value):
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(4) if text.isdigit() else text

    return series.map(normalize)


def detect_keys(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    if "roi_path" in left.columns and "roi_path" in right.columns:
        return ["roi_path"]
    for keys in [
        ["patient_id", "image_name", "object_index"],
        ["patient_id", "source_image", "object_index"],
    ]:
        if all(column in left.columns and column in right.columns for column in keys):
            return keys
    raise ValueError("Could not detect a shared unique ROI key.")


def load_component(
    path: Path,
    prefix: str,
    fold: int,
    require_aux_risk: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required prediction file not found:\n  {path}")

    frame = pd.read_csv(path)
    required = ["patient_id", "y_true", *PROB_COLS]
    if require_aux_risk:
        required.append("risk_prob_aux")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    frame["patient_id"] = normalize_patient_id(frame["patient_id"])
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    for column in PROB_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    probability = frame[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError(f"{path}: non-finite probabilities.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{path}: probabilities do not sum to one.")
    if (probability < -1e-8).any() or (probability > 1 + 1e-8).any():
        raise ValueError(f"{path}: probabilities outside [0,1].")

    if require_aux_risk:
        frame["risk_prob_aux"] = pd.to_numeric(
            frame["risk_prob_aux"], errors="raise"
        )

    if "protocol_v2_cv_fold" in frame.columns:
        embedded = pd.to_numeric(
            frame["protocol_v2_cv_fold"], errors="raise"
        ).astype(int)
        if not (embedded == fold).all():
            raise ValueError(f"{path}: embedded fold mismatch.")

    rename = {column: f"{prefix}_{column}" for column in PROB_COLS}
    if require_aux_risk:
        rename["risk_prob_aux"] = f"{prefix}_risk_prob_aux"
    frame = frame.rename(columns=rename)
    return frame


def merge_fold(components: Dict[str, pd.DataFrame], fold: int) -> Tuple[pd.DataFrame, List[str]]:
    names = list(components)
    merged = components[names[0]].copy()

    for name in names[1:]:
        right = components[name]
        keys = detect_keys(merged, right)
        keep = keys + ["y_true"] + [
            column for column in right.columns if column.startswith(f"{name}_")
        ]
        merged = merged.merge(
            right[keep],
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("", f"_{name}"),
        )

    y_columns = [
        column
        for column in merged.columns
        if column == "y_true" or column.startswith("y_true_")
    ]
    reference_y = merged[y_columns[0]].to_numpy(dtype=int)
    for column in y_columns[1:]:
        if not np.array_equal(reference_y, merged[column].to_numpy(dtype=int)):
            raise ValueError(f"Fold {fold}: labels disagree across components.")
    merged["y_true"] = reference_y
    for column in y_columns:
        if column != "y_true":
            merged.drop(columns=column, inplace=True)

    expected = len(components[names[0]])
    if len(merged) != expected:
        raise ValueError(f"Fold {fold}: component merge lost rows.")
    merged["cv_fold"] = int(fold)
    return merged, detect_keys(merged, merged)


def probability_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_{column}" for column in PROB_COLS]].to_numpy(dtype=float)


def build_variant(
    merged: pd.DataFrame,
    variant: str,
    deit_weight: float,
    cnn_weight: float,
) -> pd.DataFrame:
    output = merged.copy()

    if variant == "C0":
        probability = probability_matrix(output, "c0")
        risk = probability[:, 2:].sum(axis=1)
    elif variant == "M1_G":
        probability = probability_matrix(output, "m1")
        risk = output["m1_risk_prob_aux"].to_numpy(dtype=float)
    elif variant == "CORAL":
        probability = probability_matrix(output, "coral")
        risk = probability[:, 2:].sum(axis=1)
    elif variant == "CORN":
        probability = probability_matrix(output, "corn")
        risk = probability[:, 2:].sum(axis=1)
    elif variant == "R2CV":
        probability = (
            deit_weight * probability_matrix(output, "deit")
            + cnn_weight * probability_matrix(output, "c0")
        )
        risk = probability[:, 2:].sum(axis=1)
    elif variant == "M1CV_G":
        probability = (
            deit_weight * probability_matrix(output, "deit")
            + cnn_weight * probability_matrix(output, "m1")
        )
        risk = output["m1_risk_prob_aux"].to_numpy(dtype=float)
    elif variant == "CORALCV":
        probability = (
            deit_weight * probability_matrix(output, "deit")
            + cnn_weight * probability_matrix(output, "coral")
        )
        risk = probability[:, 2:].sum(axis=1)
    elif variant == "CORNCV":
        probability = (
            deit_weight * probability_matrix(output, "deit")
            + cnn_weight * probability_matrix(output, "corn")
        )
        risk = probability[:, 2:].sum(axis=1)
    else:
        raise ValueError(variant)

    probability = probability / np.clip(
        probability.sum(axis=1, keepdims=True), 1e-12, None
    )
    for index, column in enumerate(PROB_COLS):
        output[column] = probability[:, index]
    output["y_pred"] = probability.argmax(axis=1)
    output["risk_score"] = risk
    output["variant"] = variant
    return output


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
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


def classification_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    moderate = y_true == 2
    severe = y_true == 3

    result = {
        "roi_count": int(len(frame)),
        "patient_count": int(frame["patient_id"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "within_one_grade_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "undergrade_by_2plus_rate": float(np.mean((y_true - y_pred) >= 2)),
        "overgrade_by_2plus_rate": float(np.mean((y_pred - y_true) >= 2)),
        "high_risk_recall_argmax": float(
            np.sum(high_true & high_pred) / max(int(high_true.sum()), 1)
        ),
        "false_clear_rate_argmax": float(
            np.sum(high_true & ~high_pred) / max(int(high_true.sum()), 1)
        ),
        "moderate_false_clear_rate": float(
            np.sum(moderate & ~high_pred) / max(int(moderate.sum()), 1)
        ),
        "severe_false_clear_rate": float(
            np.sum(severe & ~high_pred) / max(int(severe.sum()), 1)
        ),
    }

    for grade, name in enumerate(["normal", "mild", "moderate", "severe"]):
        mask = y_true == grade
        result[f"{name}_recall"] = float(
            np.sum(mask & (y_pred == grade)) / max(int(mask.sum()), 1)
        )
    return result


def risk_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = (frame["y_true"].to_numpy(dtype=int) >= 2).astype(int)
    score = frame["risk_score"].to_numpy(dtype=float)
    return {
        "risk_auroc": float(roc_auc_score(y_true, score)),
        "risk_auprc": float(average_precision_score(y_true, score)),
        "risk_brier": float(np.mean((score - y_true) ** 2)),
        "risk_ece": expected_calibration_error(y_true, score),
    }


def binary_target(frame: pd.DataFrame) -> np.ndarray:
    values = frame["y_true"].to_numpy(dtype=int)
    if set(np.unique(values).tolist()).issubset({0, 1}):
        return values
    return (values >= 2).astype(int)


def binary_counts(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, int]:
    prediction = score >= threshold
    positive = y_true == 1
    negative = ~positive
    return {
        "tp": int(np.sum(positive & prediction)),
        "fn": int(np.sum(positive & ~prediction)),
        "tn": int(np.sum(negative & ~prediction)),
        "fp": int(np.sum(negative & prediction)),
    }


def counts_to_metrics(counts: Dict[str, int], threshold: Optional[float] = None) -> Dict[str, float]:
    tp, fn, tn, fp = counts["tp"], counts["fn"], counts["tn"], counts["fp"]
    total = tp + fn + tn + fp
    result = {
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "flag_rate": float((tp + fp) / max(total, 1)),
    }
    if threshold is not None:
        result["threshold"] = float(threshold)
    return result


def choose_threshold(y_true: np.ndarray, score: np.ndarray, target: float) -> Tuple[float, Dict[str, float]]:
    candidates = np.unique(
        np.concatenate([np.array([0.0, 1.0]), score, np.nextafter(score, -np.inf)])
    )
    best = None
    for threshold in np.sort(candidates)[::-1]:
        metrics = counts_to_metrics(
            binary_counts(y_true, score, float(threshold)), float(threshold)
        )
        if metrics["recall"] + 1e-12 < target:
            continue
        key = (metrics["specificity"], metrics["threshold"], -metrics["flag_rate"])
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise RuntimeError(f"No threshold reached target recall {target}.")
    return best[1], best[2]


def crossfit_threshold(
    frame: pd.DataFrame,
    folds: List[int],
    target: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    total = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    detail = []
    for fold in folds:
        selection = frame.loc[frame["cv_fold"] != fold]
        holdout = frame.loc[frame["cv_fold"] == fold]
        threshold, selection_metrics = choose_threshold(
            binary_target(selection),
            selection["risk_score"].to_numpy(dtype=float),
            target,
        )
        counts = binary_counts(
            binary_target(holdout),
            holdout["risk_score"].to_numpy(dtype=float),
            threshold,
        )
        holdout_metrics = counts_to_metrics(counts, threshold)
        for key in total:
            total[key] += counts[key]
        detail.append(
            {
                "fold": int(fold),
                "threshold": threshold,
                "selection_recall": selection_metrics["recall"],
                "selection_specificity": selection_metrics["specificity"],
                "holdout_recall": holdout_metrics["recall"],
                "holdout_specificity": holdout_metrics["specificity"],
                "holdout_flag_rate": holdout_metrics["flag_rate"],
                **{f"holdout_{key}": value for key, value in counts.items()},
            }
        )
    return counts_to_metrics(total), detail


def patient_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        folds = group["cv_fold"].unique()
        if len(folds) != 1:
            raise ValueError(f"Patient {patient_id} appears in multiple folds.")
        rows.append(
            {
                "patient_id": patient_id,
                "cv_fold": int(folds[0]),
                "y_true": int((group["y_true"].to_numpy(dtype=int) >= 2).any()),
                "risk_score": float(group["risk_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, folds: List[int], target: float):
    roi_result, roi_detail = crossfit_threshold(frame, folds, target)
    patient_result, patient_detail = crossfit_threshold(
        patient_frame(frame), folds, target
    )
    summary = {
        **classification_metrics(frame),
        **risk_metrics(frame),
        "crossfit_roi_recall_at_target": roi_result["recall"],
        "crossfit_roi_specificity_at_target": roi_result["specificity"],
        "crossfit_roi_flag_rate_at_target": roi_result["flag_rate"],
        "crossfit_patient_recall_at_target": patient_result["recall"],
        "crossfit_patient_specificity_at_target": patient_result["specificity"],
        "crossfit_patient_flag_rate_at_target": patient_result["flag_rate"],
    }
    detail = [{"level": "ROI", **row} for row in roi_detail]
    detail.extend({"level": "patient", **row} for row in patient_detail)
    return summary, detail


def mean_sd(frame: pd.DataFrame, group: str, metrics: List[str], count_name: str) -> pd.DataFrame:
    rows = []
    for name, subset in frame.groupby(group, sort=False):
        row = {group: name, count_name: int(len(subset))}
        for metric in metrics:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    keys: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    left = frame.loc[frame["variant"] == candidate, [*keys, *metrics]].copy()
    right = frame.loc[frame["variant"] == reference, [*keys, *metrics]].copy()
    merged = left.merge(
        right,
        on=keys,
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    output = merged[keys].copy()
    output["candidate"] = candidate
    output["reference"] = reference
    for metric in metrics:
        output[f"{metric}_difference"] = (
            merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
        )
    return output


def main() -> None:
    args = parse_args()
    seed_roots = dict(args.seed_root)
    if len(seed_roots) != len(args.seed_root):
        raise ValueError("Duplicate seed supplied.")
    if not np.isclose(args.deit_weight + args.cnn_weight, 1.0):
        raise ValueError("Fusion weights must sum to one.")

    args.out.mkdir(parents=True, exist_ok=True)
    prediction_root = args.out / "oof_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    pooled_rows = []
    fold_rows = []
    threshold_rows = []
    seed_audit = {}

    for seed, root in sorted(seed_roots.items()):
        merged_folds = []
        source_rows = []
        merge_keys = None

        for fold in args.folds:
            paths = {
                "c0": root / "C0_single_head" / f"fold_{fold}" / "val_predictions_best_loss.csv",
                "m1": root / "M1_lambda0p50" / f"fold_{fold}" / "val_predictions_best_grade_loss.csv",
                "deit": root / "DeiT_e20" / f"fold_{fold}" / "val_predictions.csv",
                "coral": root / "ordinal" / "CORAL" / f"fold_{fold}" / "val_predictions_best_loss.csv",
                "corn": root / "ordinal" / "CORN" / f"fold_{fold}" / "val_predictions_best_loss.csv",
            }
            components = {
                "c0": load_component(paths["c0"], "c0", fold),
                "m1": load_component(paths["m1"], "m1", fold, require_aux_risk=True),
                "deit": load_component(paths["deit"], "deit", fold),
                "coral": load_component(paths["coral"], "coral", fold),
                "corn": load_component(paths["corn"], "corn", fold),
            }
            merged, keys = merge_fold(components, fold)
            if merge_keys is None:
                merge_keys = keys
            elif merge_keys != keys:
                raise ValueError(f"Seed {seed}: merge keys changed across folds.")
            merged_folds.append(merged)
            source_rows.append(
                {
                    "fold": int(fold),
                    "roi_count": int(len(merged)),
                    "patient_count": int(merged["patient_id"].nunique()),
                    **{f"{name}_file": str(path) for name, path in paths.items()},
                }
            )

        combined = pd.concat(merged_folds, ignore_index=True)
        if merge_keys is None:
            raise RuntimeError("No folds loaded.")
        if int(combined.duplicated(merge_keys).sum()) != 0:
            raise ValueError(f"Seed {seed}: duplicate ROI keys.")
        if (combined.groupby("patient_id")["cv_fold"].nunique() != 1).any():
            raise ValueError(f"Seed {seed}: patient appears in multiple folds.")
        if len(combined) != args.expected_roi_count:
            raise ValueError(
                f"Seed {seed}: expected {args.expected_roi_count} ROI, found {len(combined)}."
            )
        patient_count = int(combined["patient_id"].nunique())
        if patient_count != args.expected_patient_count:
            raise ValueError(
                f"Seed {seed}: expected {args.expected_patient_count} patients, found {patient_count}."
            )

        variants = {
            variant: build_variant(
                combined, variant, args.deit_weight, args.cnn_weight
            )
            for variant in VARIANTS
        }

        seed_prediction_root = prediction_root / f"seed_{seed}"
        seed_prediction_root.mkdir(parents=True, exist_ok=True)
        for variant, frame in variants.items():
            frame.to_csv(
                seed_prediction_root / f"{variant}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            summary, details = summarize(frame, args.folds, args.target_recall)
            pooled_rows.append({"seed": int(seed), "variant": variant, **summary})
            for fold in args.folds:
                subset = frame.loc[frame["cv_fold"] == fold]
                fold_rows.append(
                    {
                        "seed": int(seed),
                        "fold": int(fold),
                        "variant": variant,
                        **classification_metrics(subset),
                        **risk_metrics(subset),
                    }
                )
            for detail in details:
                threshold_rows.append(
                    {"seed": int(seed), "variant": variant, **detail}
                )

        seed_audit[str(seed)] = {
            "root": str(root),
            "roi_count": int(len(combined)),
            "patient_count": patient_count,
            "duplicate_roi_count": 0,
            "each_patient_in_one_fold": True,
            "merge_keys": merge_keys,
            "fold_sources": source_rows,
        }

    pooled = pd.DataFrame(pooled_rows)
    folds = pd.DataFrame(fold_rows)
    thresholds = pd.DataFrame(threshold_rows)
    available_metrics = [metric for metric in METRICS if metric in pooled.columns]
    fold_metric_names = [
        metric
        for metric in METRICS
        if metric in folds.columns and not metric.startswith("crossfit_")
    ]

    pooled.to_csv(
        args.out / "ordinal_baselines_pooled_oof_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_csv(
        args.out / "ordinal_baselines_fold_metrics_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    thresholds.to_csv(
        args.out / "ordinal_baselines_crossfit_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_mean_sd = mean_sd(pooled, "variant", available_metrics, "seed_count")
    pooled_mean_sd.to_csv(
        args.out / "ordinal_baselines_pooled_oof_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_mean_sd = mean_sd(folds, "variant", fold_metric_names, "seed_fold_count")
    fold_mean_sd["note"] = (
        "Descriptive only: the same patient folds are reused across seeds."
    )
    fold_mean_sd.to_csv(
        args.out / "ordinal_baselines_fold_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    seed_difference_frames = []
    seed_summary_rows = []
    fold_difference_frames = []
    for comparison_name, candidate, reference in COMPARISONS:
        difference = paired_difference(
            pooled, candidate, reference, ["seed"], available_metrics
        )
        difference.insert(0, "comparison", comparison_name)
        seed_difference_frames.append(difference)
        summary_row = {
            "comparison": comparison_name,
            "candidate": candidate,
            "reference": reference,
            "paired_seed_count": int(len(difference)),
        }
        for metric in available_metrics:
            values = difference[f"{metric}_difference"]
            summary_row[f"{metric}_difference_mean"] = float(values.mean())
            summary_row[f"{metric}_difference_sd"] = (
                float(values.std(ddof=1)) if len(values) >= 2 else np.nan
            )
            summary_row[f"{metric}_positive_seed_count"] = int((values > 0).sum())
            summary_row[f"{metric}_negative_seed_count"] = int((values < 0).sum())
        seed_summary_rows.append(summary_row)

        fold_difference = paired_difference(
            folds, candidate, reference, ["seed", "fold"], fold_metric_names
        )
        fold_difference.insert(0, "comparison", comparison_name)
        fold_difference_frames.append(fold_difference)

    pd.concat(seed_difference_frames, ignore_index=True).to_csv(
        args.out / "ordinal_baselines_paired_seed_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(seed_summary_rows).to_csv(
        args.out / "ordinal_baselines_paired_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(fold_difference_frames, ignore_index=True).to_csv(
        args.out / "ordinal_baselines_paired_seed_fold_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "schema_version": "lss_ordinal_baselines_multiseed_v1",
        "locked_test_used": False,
        "seeds": sorted(seed_roots),
        "folds": args.folds,
        "variants": VARIANTS,
        "comparisons": COMPARISONS,
        "fusion": {
            "deit_weight": args.deit_weight,
            "cnn_or_ordinal_weight": args.cnn_weight,
        },
        "risk_score_definitions": {
            "C0_CORAL_CORN_and_fused": "P(Moderate)+P(Severe)",
            "M1_G_and_M1CV_G": "M1 auxiliary risk probability",
        },
        "primary_interpretation": (
            "Use one complete pooled OOF estimate per seed and report mean±SD "
            "across three seeds. The 15 seed-fold rows are descriptive only."
        ),
        "statistical_warning": (
            "Do not treat the 15 seed-fold rows as independent observations."
        ),
        "seed_audit": seed_audit,
    }
    with open(
        args.out / "ordinal_baselines_audit.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    display_metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "qwk",
        "mae",
        "high_risk_recall_argmax",
        "risk_auprc",
        "crossfit_roi_recall_at_target",
        "crossfit_roi_specificity_at_target",
        "crossfit_roi_flag_rate_at_target",
    ]
    display_columns = ["variant", "seed_count"]
    for metric in display_metrics:
        display_columns.extend([f"{metric}_mean", f"{metric}_sd"])

    print("=" * 124)
    print("CORAL/CORN multi-seed OOF aggregation completed")
    print("Locked test used: NO")
    print(f"Seeds: {sorted(seed_roots)}")
    print("-" * 124)
    print(pooled_mean_sd[display_columns].to_string(index=False))
    print("-" * 124)
    print(f"Output: {args.out}")
    print("=" * 124)


if __name__ == "__main__":
    main()
