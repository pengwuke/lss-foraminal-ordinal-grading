#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
56_aggregate_main_seed42.py

Aggregate the single pre-specified main-seed patient-level five-fold OOF
predictions for Protocol V3 Full-468.

This is the primary manuscript analysis. It does not use seed averaging.
Optimization-seed sensitivity is reserved for a supplementary optional analysis.

Primary baseline:
    Original_60_40 = 0.60 * Original DeiT + 0.40 * Original CNN

Intermediate ordinal system:
    Original_CORAL_60_40 = 0.60 * Original DeiT + 0.40 * CORAL

Primary proposed system:
    Original_CORAL_RISK_G_60_40 =
        0.60 * Original DeiT + 0.40 * CORAL-Risk grade-oriented checkpoint

The original fusion weight and lambda=0.50 are frozen before Full-468 analysis.
No separate test set is read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROB_COLS = [
    "prob_0_normal",
    "prob_1_mild",
    "prob_2_moderate",
    "prob_3_severe",
]

PRIMARY_VARIANTS = [
    "Original_60_40",
    "Original_CORAL_60_40",
    "Original_CORAL_RISK_G_60_40",
]

SECONDARY_VARIANTS = [
    "Original_CNN",
    "Original_DeiT",
    "CORAL",
    "CORAL_RISK_G",
    "CORAL_RISK_S",
    "Original_CORAL_RISK_S_60_40",
    "Original_25_75",
    "Original_CORAL_RISK_G_25_75",
]

COMPARISONS = [
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


def load_helper():
    path = Path(__file__).with_name("23_aggregate_ordinal_baselines_multiseed.py")
    if not path.exists():
        raise FileNotFoundError(f"Missing helper script: {path}")
    spec = importlib.util.spec_from_file_location("lss_metrics_helper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--main-seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--expected-patients", type=int, default=468)
    parser.add_argument("--expected-roi", type=int, default=2978)
    parser.add_argument("--original-deit-weight", type=float, default=0.60)
    parser.add_argument("--original-cnn-weight", type=float, default=0.40)
    parser.add_argument("--sensitivity-deit-weight", type=float, default=0.25)
    parser.add_argument("--sensitivity-cnn-weight", type=float, default=0.75)
    return parser.parse_args()


def normalize_patient(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(4) if text.isdigit() else text
    return series.map(one)


def detect_keys(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    if "roi_path" in left.columns and "roi_path" in right.columns:
        return ["roi_path"]
    for keys in [
        ["patient_id", "image_name", "object_index"],
        ["patient_id", "source_image", "object_index"],
    ]:
        if all(column in left.columns and column in right.columns for column in keys):
            return keys
    raise ValueError("Could not detect a shared ROI key.")


def load_component(
    path: Path,
    prefix: str,
    fold: int,
    require_aux: bool = False,
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
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    probability = frame[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError(f"{path}: non-finite probability.")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{path}: probability rows do not sum to one.")

    if require_aux:
        frame["risk_prob_aux"] = pd.to_numeric(
            frame["risk_prob_aux"], errors="raise"
        ).astype(float)

    for fold_column in ["protocol_v3_cv_fold", "cv_fold"]:
        if fold_column in frame.columns:
            embedded = pd.to_numeric(
                frame[fold_column], errors="raise"
            ).astype(int)
            if not (embedded == fold).all():
                raise ValueError(f"{path}: embedded fold mismatch.")

    rename = {column: f"{prefix}_{column}" for column in PROB_COLS}
    if require_aux:
        rename["risk_prob_aux"] = f"{prefix}_risk_prob_aux"
    return frame.rename(columns=rename)


def merge_fold(components: Dict[str, pd.DataFrame], fold: int) -> pd.DataFrame:
    names = list(components)
    merged = components[names[0]].copy()
    expected = len(merged)

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
        column for column in merged.columns
        if column == "y_true" or column.startswith("y_true_")
    ]
    reference = merged[y_columns[0]].to_numpy(dtype=int)
    for column in y_columns[1:]:
        if not np.array_equal(reference, merged[column].to_numpy(dtype=int)):
            raise ValueError(f"Fold {fold}: labels disagree.")
    merged["y_true"] = reference
    merged.drop(
        columns=[column for column in y_columns if column != "y_true"],
        inplace=True,
    )

    if len(merged) != expected:
        raise ValueError(f"Fold {fold}: merging lost ROI rows.")
    merged["cv_fold"] = int(fold)
    return merged


def probability(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[
        [f"{prefix}_{column}" for column in PROB_COLS]
    ].to_numpy(dtype=float)


def build_variant(
    merged: pd.DataFrame,
    variant: str,
    original_deit_weight: float,
    original_cnn_weight: float,
    sensitivity_deit_weight: float,
    sensitivity_cnn_weight: float,
) -> pd.DataFrame:
    output = merged.copy()
    native_risk: Optional[np.ndarray] = None

    if variant == "Original_CNN":
        grade = probability(output, "c0")
    elif variant == "Original_DeiT":
        grade = probability(output, "deit")
    elif variant == "CORAL":
        grade = probability(output, "coral")
    elif variant == "CORAL_RISK_G":
        grade = probability(output, "riskg")
        native_risk = output["riskg_risk_prob_aux"].to_numpy(dtype=float)
    elif variant == "CORAL_RISK_S":
        grade = probability(output, "risks")
        native_risk = output["risks_risk_prob_aux"].to_numpy(dtype=float)
    elif variant == "Original_60_40":
        grade = (
            original_deit_weight * probability(output, "deit")
            + original_cnn_weight * probability(output, "c0")
        )
    elif variant == "Original_CORAL_60_40":
        grade = (
            original_deit_weight * probability(output, "deit")
            + original_cnn_weight * probability(output, "coral")
        )
    elif variant == "Original_CORAL_RISK_G_60_40":
        grade = (
            original_deit_weight * probability(output, "deit")
            + original_cnn_weight * probability(output, "riskg")
        )
        native_risk = output["riskg_risk_prob_aux"].to_numpy(dtype=float)
    elif variant == "Original_CORAL_RISK_S_60_40":
        grade = (
            original_deit_weight * probability(output, "deit")
            + original_cnn_weight * probability(output, "risks")
        )
        native_risk = output["risks_risk_prob_aux"].to_numpy(dtype=float)
    elif variant == "Original_25_75":
        grade = (
            sensitivity_deit_weight * probability(output, "deit")
            + sensitivity_cnn_weight * probability(output, "c0")
        )
    elif variant == "Original_CORAL_RISK_G_25_75":
        grade = (
            sensitivity_deit_weight * probability(output, "deit")
            + sensitivity_cnn_weight * probability(output, "riskg")
        )
        native_risk = output["riskg_risk_prob_aux"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    grade = grade / np.clip(grade.sum(axis=1, keepdims=True), 1e-12, None)
    for index, column in enumerate(PROB_COLS):
        output[column] = grade[:, index]
    output["y_pred"] = grade.argmax(axis=1)
    output["risk_score_matched"] = grade[:, 2:].sum(axis=1)
    output["risk_score_native"] = (
        native_risk if native_risk is not None else output["risk_score_matched"].to_numpy()
    )
    output["variant"] = variant
    return output


def summarize_risk_mode(
    frame: pd.DataFrame,
    risk_mode: str,
    folds: List[int],
    target_recall: float,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    work = frame.copy()
    column = (
        "risk_score_native"
        if risk_mode == "native"
        else "risk_score_matched"
    )
    work["risk_score"] = work[column]
    summary, details = H.summarize(work, folds, target_recall)
    return summary, [
        {"risk_mode": risk_mode, **detail}
        for detail in details
    ]


def main() -> None:
    args = parse_args()
    if not np.isclose(
        args.original_deit_weight + args.original_cnn_weight, 1.0
    ):
        raise ValueError("Original fusion weights must sum to one.")
    if not np.isclose(
        args.sensitivity_deit_weight + args.sensitivity_cnn_weight, 1.0
    ):
        raise ValueError("Sensitivity fusion weights must sum to one.")

    args.out.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.out / "oof_predictions" / f"seed_{args.main_seed}"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    fold_frames = []
    source_audit = []
    for fold in args.folds:
        paths = {
            "c0": args.seed_root / "Original_CNN" / f"fold_{fold}" / "val_predictions_best_loss.csv",
            "deit": args.seed_root / "Original_DeiT" / f"fold_{fold}" / "val_predictions.csv",
            "coral": args.seed_root / "CORAL" / f"fold_{fold}" / "val_predictions_best_loss.csv",
            "riskg": args.seed_root / "CORAL_RISK_lambda0p50" / f"fold_{fold}" / "val_predictions_best_ordinal_loss.csv",
            "risks": args.seed_root / "CORAL_RISK_lambda0p50" / f"fold_{fold}" / "val_predictions_best_safety_specificity.csv",
        }
        components = {
            "c0": load_component(paths["c0"], "c0", fold),
            "deit": load_component(paths["deit"], "deit", fold),
            "coral": load_component(paths["coral"], "coral", fold),
            "riskg": load_component(paths["riskg"], "riskg", fold, require_aux=True),
            "risks": load_component(paths["risks"], "risks", fold, require_aux=True),
        }
        merged = merge_fold(components, fold)
        fold_frames.append(merged)
        source_audit.append({
            "fold": int(fold),
            "roi_count": int(len(merged)),
            "patient_count": int(merged["patient_id"].nunique()),
            **{f"{name}_file": str(path) for name, path in paths.items()},
        })

    merged = pd.concat(fold_frames, ignore_index=True)
    keys = detect_keys(merged, merged)
    if merged.duplicated(keys).any():
        raise ValueError("Duplicated ROI identities across OOF folds.")
    if len(merged) != args.expected_roi:
        raise ValueError(
            f"Expected {args.expected_roi} ROI, found {len(merged)}."
        )
    if merged["patient_id"].nunique() != args.expected_patients:
        raise ValueError(
            f"Expected {args.expected_patients} patients, "
            f"found {merged['patient_id'].nunique()}."
        )
    if (merged.groupby("patient_id")["cv_fold"].nunique() != 1).any():
        raise ValueError("At least one patient appears in multiple folds.")

    variants = PRIMARY_VARIANTS + SECONDARY_VARIANTS
    summary_rows = []
    threshold_rows = []

    for variant in variants:
        frame = build_variant(
            merged,
            variant,
            args.original_deit_weight,
            args.original_cnn_weight,
            args.sensitivity_deit_weight,
            args.sensitivity_cnn_weight,
        )
        frame.to_csv(
            prediction_dir / f"{variant}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        row = {
            "analysis_role": (
                "primary" if variant in PRIMARY_VARIANTS else "secondary"
            ),
            "main_seed": int(args.main_seed),
            "variant": variant,
        }
        for risk_mode in ["native", "matched"]:
            summary, details = summarize_risk_mode(
                frame, risk_mode, args.folds, args.target_recall
            )
            for metric, value in summary.items():
                row[f"{risk_mode}_{metric}"] = value
            threshold_rows.extend([
                {
                    "main_seed": int(args.main_seed),
                    "variant": variant,
                    **detail,
                }
                for detail in details
            ])
        summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    summary_frame.to_csv(
        args.out / "main_oof_summary_seed42.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_frame.to_csv(
        args.out / "main_crossfit_thresholds_seed42.csv",
        index=False,
        encoding="utf-8-sig",
    )

    difference_rows = []
    for name, candidate, reference in COMPARISONS:
        left = summary_frame.loc[
            summary_frame["variant"] == candidate
        ].iloc[0]
        right = summary_frame.loc[
            summary_frame["variant"] == reference
        ].iloc[0]
        row = {
            "comparison": name,
            "candidate": candidate,
            "reference": reference,
        }
        numeric_columns = [
            column for column in summary_frame.columns
            if column.startswith(("native_", "matched_"))
            and pd.api.types.is_numeric_dtype(summary_frame[column])
        ]
        for column in numeric_columns:
            row[f"{column}_difference"] = float(left[column] - right[column])
        difference_rows.append(row)

    pd.DataFrame(difference_rows).to_csv(
        args.out / "main_point_differences_seed42.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "schema_version": "lss_protocol_v3_final_main_seed_v1",
        "design": "full-cohort patient-level five-fold OOF",
        "separate_test_used": False,
        "main_seed": int(args.main_seed),
        "patient_count": int(merged["patient_id"].nunique()),
        "roi_count": int(len(merged)),
        "folds": args.folds,
        "primary_baseline": "Original_60_40",
        "primary_proposed_model": "Original_CORAL_RISK_G_60_40",
        "intermediate_model": "Original_CORAL_60_40",
        "frozen_hyperparameters": {
            "original_fusion": {
                "deit": args.original_deit_weight,
                "cnn_or_ordinal": args.original_cnn_weight,
            },
            "risk_loss_weight_lambda": 0.50,
            "lambda_source": (
                "Selected previously using the 374-patient development analysis; "
                "not re-selected on Full-468."
            ),
            "target_recall": args.target_recall,
        },
        "risk_score_analysis": {
            "native": (
                "Auxiliary risk head for CORAL-Risk variants; "
                "P(Moderate)+P(Severe) otherwise."
            ),
            "matched": "P(Moderate)+P(Severe) for every variant.",
        },
        "source_files": source_audit,
        "reporting_rule": (
            "Primary manuscript results use this pre-specified main seed. "
            "Additional training seeds are supplementary optimization-sensitivity "
            "analyses and are not required before the main analysis."
        ),
    }
    with open(
        args.out / "main_analysis_audit_seed42.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    display = [
        "analysis_role",
        "variant",
        "native_accuracy",
        "native_macro_f1",
        "native_weighted_f1",
        "native_qwk",
        "native_high_risk_recall_argmax",
        "native_false_clear_rate_argmax",
        "native_risk_auprc",
        "native_crossfit_roi_recall_at_target",
        "native_crossfit_roi_specificity_at_target",
        "native_crossfit_roi_flag_rate_at_target",
    ]
    print("=" * 132)
    print("Protocol V3 final MAIN aggregation completed")
    print("Main seed:", args.main_seed)
    print("Separate test used: NO")
    print(summary_frame[[c for c in display if c in summary_frame.columns]].to_string(index=False))
    print("Output:", args.out)
    print("=" * 132)


if __name__ == "__main__":
    main()
