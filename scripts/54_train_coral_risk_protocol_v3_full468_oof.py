#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
54_train_coral_risk_protocol_v3_full468_oof.py

Patient-level Protocol V3 Full-468 OOF training for a risk-aware ordinal multitask CNN.

Architecture
------------
- Shared four-block Custom CNN encoder and 32-dimensional feature.
- Ordinal grade head: CORAL or CORN.
- Auxiliary binary risk head: grade >= 2 (Moderate or Severe).

Loss
----
L_total = L_ordinal + lambda_risk * L_risk_BCE + L_regularization

Primary model state
-------------------
The default model uses the minimum validation ordinal-objective-loss checkpoint
(`best_val_ordinal_loss.pth`). This keeps checkpoint selection matched to the
strong CORAL/CORN baseline and is the primary state for lambda selection.

A safety-oriented checkpoint is also saved. It maximizes validation
specificity at a threshold satisfying the predeclared target risk recall.
This checkpoint is a secondary high-sensitivity operating mode and is not used
for selecting lambda unless explicitly stated.

Development-only safety
-----------------------
Locked test evaluation is disabled by default and should remain disabled during
OOF development.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:
    raise SystemExit(
        "Missing albumentations. Install with: "
        "python -m pip install albumentations opencv-python"
    ) from exc


CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
PROB_COLS = [
    "prob_0_normal",
    "prob_1_mild",
    "prob_2_moderate",
    "prob_3_severe",
]
CUMULATIVE_COLS = ["prob_gt_0", "prob_gt_1", "prob_gt_2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--ordinal-method",
        choices=["coral", "corn"],
        default="coral",
    )
    parser.add_argument("--risk-loss-weight", type=float, required=True)
    parser.add_argument("--risk-pos-weight", type=float, default=1.0)
    parser.add_argument("--target-risk-recall", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0007723)
    parser.add_argument("--weight-decay", type=float, default=0.002453)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fc-l2", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sampler",
        choices=["weighted", "none"],
        default="weighted",
    )
    parser.add_argument(
        "--cv-fold",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-monitor",
        choices=[
            "val_joint_loss",
            "val_ordinal_loss",
            "val_qwk",
            "val_risk_auprc",
        ],
        default="val_joint_loss",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep disabled during OOF development.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    cv2.setNumThreads(0)


def detect_columns(frame: pd.DataFrame) -> Tuple[str, str, str]:
    path_candidates = ["roi_path", "path", "image_path"]
    label_candidates = ["severity", "y_true", "label", "class_id", "grade"]
    split_candidates = ["protocol_v3_split", "protocol_v2_split", "split_patient", "split"]

    path_col = next((c for c in path_candidates if c in frame.columns), None)
    label_col = next((c for c in label_candidates if c in frame.columns), None)
    split_col = next((c for c in split_candidates if c in frame.columns), None)

    if not path_col or not label_col or not split_col:
        raise ValueError(
            "Could not detect path/label/split columns. "
            f"Available columns: {list(frame.columns)}"
        )
    return path_col, label_col, split_col


def build_transforms(seed: int):
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-30, 30),
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7,
            ),
            A.GaussNoise(p=0.3),
            A.Resize(64, 64),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ],
        seed=seed,
    )
    eval_transform = A.Compose(
        [
            A.Resize(64, 64),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ],
        seed=seed,
    )
    return train_transform, eval_transform


class ROIDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        path_col: str,
        label_col: str,
        transform,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = str(row[self.path_col])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read ROI: {path}")
        image = cv2.equalizeHist(image)
        tensor = self.transform(image=image)["image"]
        grade = int(row[self.label_col])
        risk = float(grade >= 2)
        return tensor, grade, risk, index


class CustomCNNEncoder(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(dropout)
        self.fc_shared = nn.Linear(4 * 4 * 256, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        return self.dropout(F.relu(self.fc_shared(x)))


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class RankConsistentCoralHead(nn.Module):
    def __init__(self, in_features: int = 32, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.score = nn.Linear(in_features, 1, bias=False)
        self.threshold_base = nn.Parameter(torch.tensor(-1.0))
        initial_step = inverse_softplus(1.0)
        self.threshold_delta_raw = nn.Parameter(
            torch.full((num_classes - 2,), initial_step)
        )

    def ordered_thresholds(self) -> torch.Tensor:
        increments = F.softplus(self.threshold_delta_raw) + 1e-4
        return torch.cat(
            [
                self.threshold_base.reshape(1),
                self.threshold_base + torch.cumsum(increments, dim=0),
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        score = self.score(features)
        thresholds = self.ordered_thresholds().reshape(1, -1)
        return score - thresholds


class RiskAwareOrdinalCNN(nn.Module):
    def __init__(self, method: str, dropout: float = 0.5):
        super().__init__()
        self.method = method
        self.encoder = CustomCNNEncoder(dropout=dropout)
        if method == "coral":
            self.ordinal_head = RankConsistentCoralHead(32, 4)
        elif method == "corn":
            self.ordinal_head = nn.Linear(32, 3)
        else:
            raise ValueError(f"Unsupported ordinal method: {method}")
        self.risk_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        ordinal_logits = self.ordinal_head(features)
        risk_logits = self.risk_head(features).squeeze(1)
        return ordinal_logits, risk_logits


def coral_targets(labels: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(3, device=labels.device).reshape(1, -1)
    return (labels.reshape(-1, 1) > thresholds).float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, coral_targets(labels))


def corn_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    total_loss = logits.new_tensor(0.0)
    total_terms = 0
    for threshold_index in range(3):
        if threshold_index == 0:
            eligible = torch.ones_like(labels, dtype=torch.bool)
        else:
            eligible = labels > (threshold_index - 1)
        if not eligible.any():
            continue
        target = (labels[eligible] > threshold_index).float()
        total_loss = total_loss + F.binary_cross_entropy_with_logits(
            logits[eligible, threshold_index],
            target,
            reduction="sum",
        )
        total_terms += int(eligible.sum().item())
    if total_terms == 0:
        raise RuntimeError("CORN loss received no eligible terms.")
    return total_loss / total_terms


def ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    method: str,
) -> torch.Tensor:
    if method == "coral":
        return coral_loss(logits, labels)
    if method == "corn":
        return corn_loss(logits, labels)
    raise ValueError(method)


def cumulative_and_class_probabilities(
    logits: torch.Tensor,
    method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "coral":
        cumulative = torch.sigmoid(logits)
        cumulative = torch.cummin(cumulative, dim=1).values
    elif method == "corn":
        conditional = torch.sigmoid(logits)
        cumulative = torch.cumprod(conditional, dim=1)
    else:
        raise ValueError(method)

    probabilities = torch.stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2],
        ],
        dim=1,
    )
    probabilities = torch.clamp(probabilities, min=0.0)
    probabilities = probabilities / torch.clamp(
        probabilities.sum(dim=1, keepdim=True),
        min=1e-12,
    )
    return cumulative, probabilities


def ordinal_regularization(model: RiskAwareOrdinalCNN) -> torch.Tensor:
    """Regularization matched to the existing CORAL/CORN baseline."""
    value = model.encoder.fc_shared.weight.square().sum()
    if model.method == "coral":
        value = value + model.ordinal_head.score.weight.square().sum()
    else:
        value = value + model.ordinal_head.weight.square().sum()
    return value


def joint_regularization(model: RiskAwareOrdinalCNN) -> torch.Tensor:
    return ordinal_regularization(model) + model.risk_head.weight.square().sum()


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (
                (probability >= edges[index])
                & (probability <= edges[index + 1])
            )
        else:
            mask = (
                (probability >= edges[index])
                & (probability < edges[index + 1])
            )
        if not mask.any():
            continue
        result += float(mask.mean()) * abs(
            float(probability[mask].mean())
            - float(y_true[mask].mean())
        )
    return float(result)


def choose_threshold_for_recall(
    y_true: np.ndarray,
    score: np.ndarray,
    target_recall: float,
) -> Tuple[float, Dict[str, float]]:
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0], dtype=float),
                score,
                np.nextafter(score, -np.inf),
            ]
        )
    )
    best = None
    for threshold in np.sort(candidates)[::-1]:
        prediction = score >= threshold
        positive = y_true == 1
        negative = ~positive
        tp = int(np.sum(positive & prediction))
        fn = int(np.sum(positive & ~prediction))
        tn = int(np.sum(negative & ~prediction))
        fp = int(np.sum(negative & prediction))
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        flag_rate = float(prediction.mean())
        if recall + 1e-12 < target_recall:
            continue
        candidate = {
            "threshold": float(threshold),
            "recall": float(recall),
            "specificity": float(specificity),
            "flag_rate": flag_rate,
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
        }
        key = (specificity, float(threshold), -flag_rate)
        if best is None or key > best[0]:
            best = (key, candidate)
    if best is None:
        raise RuntimeError("No threshold satisfied the requested recall.")
    return best[1]["threshold"], best[1]


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grade_probability: np.ndarray,
    risk_probability: np.ndarray,
    target_risk_recall: float,
) -> Dict[str, float]:
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    y_risk = high_true.astype(int)
    class_sum = grade_probability[:, 2] + grade_probability[:, 3]
    threshold, safety = choose_threshold_for_recall(
        y_risk,
        risk_probability,
        target_risk_recall,
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
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
        "within_one_grade_accuracy": float(
            np.mean(np.abs(y_true - y_pred) <= 1)
        ),
        "undergrade_by_2plus_rate": float(
            np.mean((y_true - y_pred) >= 2)
        ),
        "overgrade_by_2plus_rate": float(
            np.mean((y_pred - y_true) >= 2)
        ),
        "high_risk_recall": float(
            np.sum(high_true & high_pred) / max(int(high_true.sum()), 1)
        ),
        "high_risk_false_clear_rate": float(
            np.sum(high_true & ~high_pred) / max(int(high_true.sum()), 1)
        ),
        "risk_auroc": float(roc_auc_score(y_risk, risk_probability)),
        "risk_auprc": float(
            average_precision_score(y_risk, risk_probability)
        ),
        "risk_brier": float(
            np.mean((risk_probability - y_risk) ** 2)
        ),
        "risk_ece": expected_calibration_error(
            y_risk,
            risk_probability,
        ),
        "ordinal_risk_auroc": float(roc_auc_score(y_risk, class_sum)),
        "ordinal_risk_auprc": float(
            average_precision_score(y_risk, class_sum)
        ),
        "ordinal_risk_brier": float(
            np.mean((class_sum - y_risk) ** 2)
        ),
        "ordinal_risk_ece": expected_calibration_error(y_risk, class_sum),
        "safety_threshold": float(threshold),
        "safety_recall": safety["recall"],
        "safety_specificity": safety["specificity"],
        "safety_flag_rate": safety["flag_rate"],
        "mean_max_probability": float(grade_probability.max(axis=1).mean()),
        "mean_entropy_norm": float(
            (
                -(
                    grade_probability
                    * np.log(np.clip(grade_probability, 1e-12, 1.0))
                ).sum(axis=1)
                / math.log(4)
            ).mean()
        ),
    }

    for grade, name in enumerate(CLASS_NAMES):
        mask = y_true == grade
        result[f"recall_{name.lower()}"] = float(
            np.sum(mask & (y_pred == grade)) / max(int(mask.sum()), 1)
        )
    return result


@torch.no_grad()
def evaluate(
    model: RiskAwareOrdinalCNN,
    loader: DataLoader,
    device: torch.device,
    method: str,
    risk_criterion: nn.Module,
    risk_loss_weight: float,
    fc_l2: float,
    target_risk_recall: float,
):
    model.eval()
    joint_sum = 0.0
    ordinal_sum = 0.0
    ordinal_data_sum = 0.0
    risk_sum = 0.0
    total = 0
    all_indices: List[int] = []
    all_true: List[int] = []
    all_grade_probability: List[np.ndarray] = []
    all_cumulative: List[np.ndarray] = []
    all_risk_probability: List[np.ndarray] = []

    for images, grades, risks, indices in loader:
        images = images.to(device, non_blocking=True)
        grades = grades.to(device, non_blocking=True)
        risks = risks.to(device, non_blocking=True)

        ordinal_logits, risk_logits = model(images)
        ordinal_data_value = ordinal_loss(ordinal_logits, grades, method)
        risk_value = risk_criterion(risk_logits, risks)
        ordinal_regularization_value = ordinal_data_value.new_tensor(0.0)
        joint_regularization_value = ordinal_data_value.new_tensor(0.0)
        if fc_l2 > 0:
            ordinal_regularization_value = fc_l2 * ordinal_regularization(model)
            joint_regularization_value = fc_l2 * joint_regularization(model)
        # The primary ordinal checkpoint exactly matches the baseline criterion:
        # encoder + ordinal-head regularization, excluding the auxiliary head.
        ordinal_value = ordinal_data_value + ordinal_regularization_value
        joint_value = (
            ordinal_data_value
            + risk_loss_weight * risk_value
            + joint_regularization_value
        )

        cumulative, grade_probability = cumulative_and_class_probabilities(
            ordinal_logits,
            method,
        )
        risk_probability = torch.sigmoid(risk_logits)

        batch_count = grades.numel()
        total += batch_count
        joint_sum += float(joint_value.detach().cpu()) * batch_count
        ordinal_sum += float(ordinal_value.detach().cpu()) * batch_count
        ordinal_data_sum += float(ordinal_data_value.detach().cpu()) * batch_count
        risk_sum += float(risk_value.detach().cpu()) * batch_count
        all_indices.extend(indices.numpy().tolist())
        all_true.extend(grades.cpu().numpy().tolist())
        all_grade_probability.append(grade_probability.cpu().numpy())
        all_cumulative.append(cumulative.cpu().numpy())
        all_risk_probability.append(risk_probability.cpu().numpy())

    grade_probability = np.concatenate(all_grade_probability, axis=0)
    cumulative = np.concatenate(all_cumulative, axis=0)
    risk_probability = np.concatenate(all_risk_probability, axis=0)
    y_true = np.asarray(all_true, dtype=int)
    y_pred = grade_probability.argmax(axis=1)

    metrics = metrics_from_predictions(
        y_true,
        y_pred,
        grade_probability,
        risk_probability,
        target_risk_recall,
    )
    metrics.update(
        {
            "joint_loss": joint_sum / max(total, 1),
            "ordinal_loss": ordinal_sum / max(total, 1),
            "ordinal_data_loss": ordinal_data_sum / max(total, 1),
            "risk_loss": risk_sum / max(total, 1),
        }
    )
    return (
        metrics,
        np.asarray(all_indices, dtype=int),
        y_true,
        y_pred,
        grade_probability,
        cumulative,
        risk_probability,
    )


def save_predictions(
    source_frame: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grade_probability: np.ndarray,
    cumulative: np.ndarray,
    risk_probability: np.ndarray,
    method: str,
    risk_loss_weight: float,
    output_path: Path,
) -> None:
    prediction = source_frame.iloc[indices].reset_index(drop=True).copy()
    prediction["y_true"] = y_true
    prediction["y_pred"] = y_pred
    prediction["correct"] = y_true == y_pred
    prediction["ordinal_method"] = method.upper()
    prediction["risk_loss_weight"] = float(risk_loss_weight)

    for index, column in enumerate(PROB_COLS):
        prediction[column] = grade_probability[:, index]
    for index, column in enumerate(CUMULATIVE_COLS):
        prediction[column] = cumulative[:, index]

    prediction["risk_prob_aux"] = risk_probability
    prediction["risk_prob_ordinal"] = (
        grade_probability[:, 2] + grade_probability[:, 3]
    )
    prediction["true_high_risk"] = (y_true >= 2).astype(int)
    prediction["dangerous_false_clear_argmax"] = (
        (y_true >= 2) & (y_pred < 2)
    )
    prediction.to_csv(output_path, index=False, encoding="utf-8-sig")


def better(
    current: float,
    best: float,
    monitor: str,
    min_delta: float,
) -> bool:
    if monitor in {"val_joint_loss", "val_ordinal_loss"}:
        return current < best - min_delta
    return current > best + min_delta


def main() -> None:
    args = parse_args()
    if args.risk_loss_weight < 0:
        raise ValueError("--risk-loss-weight must be non-negative.")
    if args.risk_pos_weight <= 0:
        raise ValueError("--risk-pos-weight must be positive.")
    if not 0 < args.target_risk_recall <= 1:
        raise ValueError("--target-risk-recall must be in (0,1].")

    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    frame = pd.read_csv(args.roi_splits)
    path_col, label_col, split_col = detect_columns(frame)
    if "protocol_v3_cv_fold" not in frame.columns:
        raise ValueError("Manifest must contain protocol_v3_cv_fold.")

    fold_series = pd.to_numeric(
        frame["protocol_v3_cv_fold"], errors="raise"
    ).astype(int)
    train_frame = frame.loc[
        fold_series != args.cv_fold
    ].copy()
    val_frame = frame.loc[
        fold_series == args.cv_fold
    ].copy()
    test_frame = frame.iloc[0:0].copy()
    if args.evaluate_test:
        raise ValueError(
            "Protocol V3 Full-468 has no separate test split. "
            "Use --no-evaluate-test."
        )

    if min(len(train_frame), len(val_frame)) == 0:
        raise ValueError("Empty train or validation fold.")
    if args.evaluate_test and len(test_frame) == 0:
        raise ValueError("Test evaluation requested but test split is empty.")

    train_transform, eval_transform = build_transforms(args.seed)
    train_dataset = ROIDataset(
        train_frame, path_col, label_col, train_transform
    )
    val_dataset = ROIDataset(val_frame, path_col, label_col, eval_transform)
    test_dataset = ROIDataset(
        test_frame, path_col, label_col, eval_transform
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = None
    shuffle = True
    class_counts = np.bincount(
        train_frame[label_col].astype(int).to_numpy(), minlength=4
    )
    if args.sampler == "weighted":
        labels = train_frame[label_col].astype(int).to_numpy()
        class_weights = np.divide(
            1.0,
            class_counts,
            out=np.zeros_like(class_counts, dtype=float),
            where=class_counts > 0,
        )
        sample_weights = torch.as_tensor(
            class_weights[labels], dtype=torch.double
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator if sampler is None else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskAwareOrdinalCNN(args.ordinal_method, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    risk_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.risk_pos_weight, device=device)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )

    monitors = {
        "val_joint_loss": (float("inf"), "best_val_joint_loss.pth"),
        "val_ordinal_loss": (float("inf"), "best_val_ordinal_loss.pth"),
        "val_qwk": (-float("inf"), "best_val_qwk.pth"),
        "val_risk_auprc": (-float("inf"), "best_val_risk_auprc.pth"),
        "val_safety_specificity": (
            -float("inf"),
            "best_val_safety_specificity.pth",
        ),
    }
    best_epochs = {monitor: None for monitor in monitors}
    history: List[Dict[str, float]] = []
    no_improvement = 0

    print("=" * 110)
    print(
        f"Risk-aware ordinal={args.ordinal_method.upper()} | device={device} | "
        f"seed={args.seed} | fold={args.cv_fold}"
    )
    print(
        f"Train/Val/Test ROI={len(train_frame)}/{len(val_frame)}/{len(test_frame)} | "
        f"test evaluation={'ENABLED' if args.evaluate_test else 'DISABLED'}"
    )
    print(
        f"lambda_risk={args.risk_loss_weight} | pos_weight={args.risk_pos_weight} | "
        f"target_recall={args.target_risk_recall}"
    )
    print(f"Train class counts={class_counts.tolist()}")
    print("=" * 110)

    for epoch in range(1, args.epochs + 1):
        model.train()
        joint_sum = 0.0
        ordinal_sum = 0.0
        ordinal_data_sum = 0.0
        risk_sum = 0.0
        seen = 0

        for images, grades, risks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            grades = grades.to(device, non_blocking=True)
            risks = risks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                ordinal_logits, risk_logits = model(images)
                ordinal_data_value = ordinal_loss(
                    ordinal_logits, grades, args.ordinal_method
                )
                risk_value = risk_criterion(risk_logits, risks)
                ordinal_regularization_value = ordinal_data_value.new_tensor(0.0)
                joint_regularization_value = ordinal_data_value.new_tensor(0.0)
                if args.fc_l2 > 0:
                    ordinal_regularization_value = (
                        args.fc_l2 * ordinal_regularization(model)
                    )
                    joint_regularization_value = (
                        args.fc_l2 * joint_regularization(model)
                    )
                ordinal_value = ordinal_data_value + ordinal_regularization_value
                joint_value = (
                    ordinal_data_value
                    + args.risk_loss_weight * risk_value
                    + joint_regularization_value
                )

            scaler.scale(joint_value).backward()
            scaler.step(optimizer)
            scaler.update()

            count = grades.numel()
            seen += count
            joint_sum += float(joint_value.detach().cpu()) * count
            ordinal_sum += float(ordinal_value.detach().cpu()) * count
            ordinal_data_sum += float(ordinal_data_value.detach().cpu()) * count
            risk_sum += float(risk_value.detach().cpu()) * count

        val_result, _, _, _, _, _, _ = evaluate(
            model,
            val_loader,
            device,
            args.ordinal_method,
            risk_criterion,
            args.risk_loss_weight,
            args.fc_l2,
            args.target_risk_recall,
        )

        history_row = {
            "epoch": epoch,
            "train_joint_loss": joint_sum / max(seen, 1),
            "train_ordinal_loss": ordinal_sum / max(seen, 1),
            "train_ordinal_data_loss": ordinal_data_sum / max(seen, 1),
            "train_risk_loss": risk_sum / max(seen, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history_row.update(
            {f"val_{key}": value for key, value in val_result.items()}
        )
        history.append(history_row)
        pd.DataFrame(history).to_csv(
            args.out / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        current_values = {
            "val_joint_loss": val_result["joint_loss"],
            "val_ordinal_loss": val_result["ordinal_loss"],
            "val_qwk": val_result["qwk"],
            "val_risk_auprc": val_result["risk_auprc"],
            "val_safety_specificity": val_result["safety_specificity"],
        }
        improved_stop_monitor = False

        for monitor, (best_value, filename) in list(monitors.items()):
            current = current_values[monitor]
            if better(current, best_value, monitor, args.min_delta):
                monitors[monitor] = (current, filename)
                best_epochs[monitor] = epoch
                torch.save(
                    {
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "epoch": epoch,
                        "monitor": monitor,
                        "monitor_value": current,
                        "ordinal_method": args.ordinal_method,
                        "risk_loss_weight": args.risk_loss_weight,
                        "args": vars(args),
                    },
                    args.out / filename,
                )
                if monitor == args.early_stop_monitor:
                    improved_stop_monitor = True

        no_improvement = 0 if improved_stop_monitor else no_improvement + 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"joint={val_result['joint_loss']:.5f} "
            f"ordinal={val_result['ordinal_loss']:.5f} "
            f"risk={val_result['risk_loss']:.5f} "
            f"acc={val_result['accuracy']:.4f} "
            f"macro_f1={val_result['macro_f1']:.4f} "
            f"qwk={val_result['qwk']:.4f} "
            f"argmax_high={val_result['high_risk_recall']:.4f} "
            f"auprc={val_result['risk_auprc']:.4f} "
            f"safety_spec={val_result['safety_specificity']:.4f}"
        )

        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs
            and no_improvement >= args.early_stop_patience
        ):
            print(
                f"Early stopping after {args.early_stop_patience} epochs "
                f"without {args.early_stop_monitor} improvement."
            )
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": history[-1]["epoch"],
            "ordinal_method": args.ordinal_method,
            "risk_loss_weight": args.risk_loss_weight,
            "args": vars(args),
        },
        args.out / "last_model.pth",
    )

    history_frame = pd.DataFrame(history)
    plt.figure(figsize=(9, 5))
    plt.plot(
        history_frame["epoch"],
        history_frame["train_joint_loss"],
        label="Train joint loss",
    )
    plt.plot(
        history_frame["epoch"],
        history_frame["val_joint_loss"],
        label="Validation joint loss",
    )
    plt.plot(
        history_frame["epoch"],
        history_frame["val_ordinal_loss"],
        label="Validation ordinal loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{args.ordinal_method.upper()}-Risk training, lambda={args.risk_loss_weight}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "training_loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history_frame["epoch"], history_frame["val_accuracy"], label="Accuracy")
    plt.plot(history_frame["epoch"], history_frame["val_macro_f1"], label="Macro F1")
    plt.plot(history_frame["epoch"], history_frame["val_qwk"], label="QWK")
    plt.plot(
        history_frame["epoch"],
        history_frame["val_high_risk_recall"],
        label="Argmax moderate-or-severe recall",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.title(
        f"{args.ordinal_method.upper()}-Risk validation metrics"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "validation_metrics_curve.png", dpi=180)
    plt.close()

    checkpoint_results: Dict[str, object] = {}
    for monitor, (_, filename) in monitors.items():
        checkpoint_path = args.out / filename
        if not checkpoint_path.exists():
            continue
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        (
            validation_metrics,
            validation_indices,
            validation_true,
            validation_pred,
            validation_probability,
            validation_cumulative,
            validation_risk_probability,
        ) = evaluate(
            model,
            val_loader,
            device,
            args.ordinal_method,
            risk_criterion,
            args.risk_loss_weight,
            args.fc_l2,
            args.target_risk_recall,
        )

        tag = monitor.replace("val_", "")
        output_path = args.out / f"val_predictions_best_{tag}.csv"
        save_predictions(
            val_frame,
            validation_indices,
            validation_true,
            validation_pred,
            validation_probability,
            validation_cumulative,
            validation_risk_probability,
            args.ordinal_method,
            args.risk_loss_weight,
            output_path,
        )

        # Compatibility alias for the primary matched ordinal checkpoint.
        if monitor == "val_ordinal_loss":
            save_predictions(
                val_frame,
                validation_indices,
                validation_true,
                validation_pred,
                validation_probability,
                validation_cumulative,
                validation_risk_probability,
                args.ordinal_method,
                args.risk_loss_weight,
                args.out / "val_predictions_best_loss.csv",
            )

        test_metrics = None
        if args.evaluate_test:
            (
                test_metrics,
                test_indices,
                test_true,
                test_pred,
                test_probability,
                test_cumulative,
                test_risk_probability,
            ) = evaluate(
                model,
                test_loader,
                device,
                args.ordinal_method,
                risk_criterion,
                args.risk_loss_weight,
                args.fc_l2,
                args.target_risk_recall,
            )
            save_predictions(
                test_frame,
                test_indices,
                test_true,
                test_pred,
                test_probability,
                test_cumulative,
                test_risk_probability,
                args.ordinal_method,
                args.risk_loss_weight,
                args.out / f"test_predictions_best_{tag}.csv",
            )

        checkpoint_results[monitor] = {
            "epoch": int(checkpoint["epoch"]),
            "validation": validation_metrics,
            "test": test_metrics,
        }

    summary = {
        "schema_version": "lss_risk_aware_ordinal_oof_v1",
        "ordinal_method": args.ordinal_method.upper(),
        "risk_loss_weight": args.risk_loss_weight,
        "risk_pos_weight": args.risk_pos_weight,
        "target_risk_recall": args.target_risk_recall,
        "active_protocol": f"development_cv_fold_{args.cv_fold}",
        "seed": args.seed,
        "cv_fold": args.cv_fold,
        "train_roi": int(len(train_frame)),
        "validation_roi": int(len(val_frame)),
        "test_roi": int(len(test_frame)),
        "test_evaluation_enabled": bool(args.evaluate_test),
        "locked_test_used": bool(args.evaluate_test),
        "device": str(device),
        "class_counts_train": class_counts.tolist(),
        "best_epochs": best_epochs,
        "checkpoint_results": checkpoint_results,
        "method_definition": {
            "encoder": "Custom CNN encoder and 32-dimensional shared feature",
            "ordinal_head": args.ordinal_method.upper(),
            "risk_head": "binary BCE head for grade >= 2",
            "loss": "ordinal loss + lambda_risk * BCE + FC regularization",
            "primary_checkpoint": "minimum validation ordinal objective loss",
            "secondary_checkpoint": (
                "maximum validation specificity at target risk recall"
            ),
        },
        "warning": (
            "Use val_predictions_best_ordinal_loss.csv for matched-checkpoint "
            "lambda selection and the default model. Keep --no-evaluate-test "
            "during OOF development."
        ),
    }
    with open(
        args.out / "risk_aware_ordinal_training_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)

    print("=" * 110)
    print("Risk-aware ordinal OOF training completed")
    print(f"Output: {args.out}")
    print(f"Best epochs: {best_epochs}")
    print(
        "Locked test evaluation: "
        + ("ENABLED" if args.evaluate_test else "DISABLED")
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
