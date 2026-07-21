#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
58_generate_main_diagnostics.py

Generate manuscript-ready diagnostic outputs for the pre-specified main seed:
- Confusion matrices
- Per-grade metrics
- Risk calibration tables and plots
- Risk threshold / workload curves
- Entropy-based selective review curves
- Persistent false-clear ROI lists

This script performs no training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

PROB_COLS = [
    "prob_0_normal",
    "prob_1_mild",
    "prob_2_moderate",
    "prob_3_severe",
]
CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
DEFAULT_VARIANTS = [
    "Original_60_40",
    "Original_CORAL_60_40",
    "Original_CORAL_RISK_G_60_40",
    "CORAL",
    "CORAL_RISK_G",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--main-seed", type=int, default=42)
    parser.add_argument(
        "--variants", nargs="+", default=DEFAULT_VARIANTS
    )
    parser.add_argument(
        "--risk-mode", choices=["native", "matched"], default="native"
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--review-fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    )
    return parser.parse_args()


def normalize_patient(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(4) if text.isdigit() else text
    return series.map(one)


def load_prediction(
    root: Path,
    seed: int,
    variant: str,
    risk_mode: str,
) -> pd.DataFrame:
    path = root / "oof_predictions" / f"seed_{seed}" / f"{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF prediction file:\n  {path}")
    frame = pd.read_csv(path)
    risk_column = (
        "risk_score_native" if risk_mode == "native"
        else "risk_score_matched"
    )
    required = ["patient_id", "y_true", "cv_fold", risk_column, *PROB_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    frame = frame.copy()
    frame["patient_id"] = normalize_patient(frame["patient_id"])
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["cv_fold"] = pd.to_numeric(frame["cv_fold"], errors="raise").astype(int)
    frame["risk_score"] = pd.to_numeric(
        frame[risk_column], errors="raise"
    ).astype(float)
    for column in PROB_COLS:
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        ).astype(float)
    probability = frame[PROB_COLS].to_numpy(dtype=float)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{path}: probability rows do not sum to one.")
    frame["y_pred"] = probability.argmax(axis=1)
    return frame


def classification_summary(frame: pd.DataFrame) -> Dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    return {
        "roi_count": int(len(frame)),
        "patient_count": int(frame["patient_id"].nunique()),
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
            average_precision_score(
                high_true.astype(int),
                frame["risk_score"].to_numpy(dtype=float),
            )
        ),
    }


def per_grade(frame: pd.DataFrame) -> pd.DataFrame:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    rows = []
    for grade, name in enumerate(CLASS_NAMES):
        actual = y_true == grade
        predicted = y_pred == grade
        tp = int(np.sum(actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        fp = int(np.sum(~actual & predicted))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({
            "grade": grade,
            "grade_name": name,
            "support": int(actual.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)


def calibration_table(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    truth = (frame["y_true"].to_numpy(dtype=int) >= 2).astype(int)
    score = frame["risk_score"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        if index == bins - 1:
            mask = (score >= edges[index]) & (score <= edges[index + 1])
        else:
            mask = (score >= edges[index]) & (score < edges[index + 1])
        if not mask.any():
            continue
        rows.append({
            "bin": index,
            "bin_low": edges[index],
            "bin_high": edges[index + 1],
            "count": int(mask.sum()),
            "mean_predicted_risk": float(score[mask].mean()),
            "observed_high_risk_rate": float(truth[mask].mean()),
            "absolute_gap": abs(
                float(score[mask].mean()) - float(truth[mask].mean())
            ),
        })
    return pd.DataFrame(rows)


def threshold_curve(frame: pd.DataFrame) -> pd.DataFrame:
    truth = frame["y_true"].to_numpy(dtype=int) >= 2
    score = frame["risk_score"].to_numpy(dtype=float)
    thresholds = np.unique(
        np.concatenate([np.linspace(0, 1, 201), score])
    )
    rows = []
    for threshold in thresholds:
        flag = score >= threshold
        tp = int(np.sum(truth & flag))
        fn = int(np.sum(truth & ~flag))
        tn = int(np.sum(~truth & ~flag))
        fp = int(np.sum(~truth & flag))
        rows.append({
            "threshold": float(threshold),
            "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "flag_rate": (tp + fp) / max(len(frame), 1),
            "precision": tp / max(tp + fp, 1),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
        })
    return pd.DataFrame(rows)


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    return (
        -np.sum(
            probability * np.log(np.clip(probability, 1e-12, 1.0)),
            axis=1,
        )
        / math.log(4)
    )


def selective_review(
    frame: pd.DataFrame,
    fractions: Sequence[float],
) -> pd.DataFrame:
    probability = frame[PROB_COLS].to_numpy(dtype=float)
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = probability.argmax(axis=1)
    order = np.argsort(-normalized_entropy(probability))
    errors = y_pred != y_true
    dangerous = (y_true >= 2) & (y_pred < 2)
    rows = []

    for fraction in sorted(set(float(value) for value in fractions)):
        count = int(round(fraction * len(frame)))
        reviewed = np.zeros(len(frame), dtype=bool)
        if count > 0:
            reviewed[order[:count]] = True
        corrected = y_pred.copy()
        corrected[reviewed] = y_true[reviewed]
        rows.append({
            "review_fraction": fraction,
            "review_count": count,
            "error_capture_rate": float(
                np.sum(errors & reviewed) / max(int(errors.sum()), 1)
            ),
            "dangerous_false_clear_capture_rate": float(
                np.sum(dangerous & reviewed) / max(int(dangerous.sum()), 1)
            ),
            "oracle_accuracy_after_review": float(
                accuracy_score(y_true, corrected)
            ),
            "oracle_macro_f1_after_review": float(
                f1_score(
                    y_true,
                    corrected,
                    average="macro",
                    zero_division=0,
                )
            ),
            "oracle_qwk_after_review": float(
                cohen_kappa_score(
                    y_true, corrected, weights="quadratic"
                )
            ),
        })
    return pd.DataFrame(rows)


def save_confusion(
    matrix: np.ndarray,
    title: str,
    path: Path,
) -> None:
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sum > 0,
    )
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(normalized, vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(4), CLASS_NAMES, rotation=25, ha="right")
    axis.set_yticks(range(4), CLASS_NAMES)
    axis.set_xlabel("Predicted grade")
    axis.set_ylabel("True grade")
    axis.set_title(title)
    for row in range(4):
        for column in range(4):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{normalized[row, column]:.1%}",
                ha="center",
                va="center",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_calibration(
    table: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.plot(
        table["mean_predicted_risk"],
        table["observed_high_risk_rate"],
        marker="o",
        label="OOF model",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted high-risk probability")
    axis.set_ylabel("Observed high-risk rate")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_threshold(
    table: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.plot(table["flag_rate"], table["recall"], label="Recall")
    axis.plot(table["flag_rate"], table["specificity"], label="Specificity")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Flag rate")
    axis.set_ylabel("Metric")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_review(
    table: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.plot(
        table["review_fraction"],
        table["error_capture_rate"],
        marker="o",
        label="All errors captured",
    )
    axis.plot(
        table["review_fraction"],
        table["dangerous_false_clear_capture_rate"],
        marker="o",
        label="Dangerous false-clears captured",
    )
    axis.set_xlim(
        0,
        max(float(table["review_fraction"].max()), 0.5),
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Reviewed ROI fraction")
    axis.set_ylabel("Captured fraction")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    false_clear_rows = []

    for variant in args.variants:
        variant_out = args.out / variant
        variant_out.mkdir(parents=True, exist_ok=True)
        frame = load_prediction(
            args.analysis_root,
            args.main_seed,
            variant,
            args.risk_mode,
        )
        frame.to_csv(
            variant_out / "main_seed_oof_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary_rows.append({
            "variant": variant,
            **classification_summary(frame),
        })

        y_true = frame["y_true"].to_numpy(dtype=int)
        y_pred = frame["y_pred"].to_numpy(dtype=int)
        matrix = confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2, 3]
        )
        pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(
            variant_out / "confusion_matrix_counts.csv",
            encoding="utf-8-sig",
        )
        save_confusion(
            matrix,
            f"{variant}: main-seed Full-468 OOF",
            variant_out / "confusion_matrix.png",
        )

        per_grade(frame).to_csv(
            variant_out / "per_grade_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        calibration = calibration_table(
            frame, args.calibration_bins
        )
        calibration.to_csv(
            variant_out / "risk_calibration_bins.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_calibration(
            calibration,
            f"{variant}: high-risk reliability",
            variant_out / "risk_calibration.png",
        )

        threshold = threshold_curve(frame)
        threshold.to_csv(
            variant_out / "risk_threshold_curve.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_threshold(
            threshold,
            f"{variant}: risk / workload trade-off",
            variant_out / "risk_threshold_tradeoff.png",
        )

        review = selective_review(
            frame, args.review_fractions
        )
        review.to_csv(
            variant_out / "selective_review_entropy.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_review(
            review,
            f"{variant}: entropy-based selective review",
            variant_out / "selective_review_entropy.png",
        )

        false_clear = frame.loc[
            (frame["y_true"] >= 2) & (frame["y_pred"] < 2)
        ].copy()
        false_clear["variant"] = variant
        false_clear_rows.append(false_clear)
        false_clear.to_csv(
            variant_out / "false_clear_rois.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(summary_rows).to_csv(
        args.out / "main_diagnostic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if false_clear_rows:
        pd.concat(false_clear_rows, ignore_index=True).to_csv(
            args.out / "false_clear_rois_all_variants.csv",
            index=False,
            encoding="utf-8-sig",
        )

    audit = {
        "schema_version": "lss_protocol_v3_main_diagnostics_v1",
        "main_seed": int(args.main_seed),
        "risk_mode": args.risk_mode,
        "variants": args.variants,
        "separate_test_used": False,
        "selective_review_note": (
            "Oracle correction is used only to quantify prioritization "
            "potential and is not a clinical performance claim."
        ),
    }
    with open(
        args.out / "main_diagnostics_audit.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    print("=" * 108)
    print("Protocol V3 main diagnostic analysis completed")
    print("Output:", args.out)
    print("=" * 108)


if __name__ == "__main__":
    main()
