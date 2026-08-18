# PUBLIC RELEASE NOTE
# Machine-specific paths were replaced by placeholders only. Scientific
# model logic, objectives, hyperparameters, fold logic and checkpoint rules
# are preserved from the frozen source identified in the provenance table.
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_train_multitask_cnn_risk_head.py

M1 training baseline:
    shared Custom-CNN encoder
      ├─ four-grade classification head
      └─ moderate-or-severe auxiliary risk head

The only methodological change from the Protocol-V2 Custom CNN is the
auxiliary binary head and its joint training loss.

Primary training objective
--------------------------
L_total = L_grade_focal + lambda_risk * L_risk_BCE + L_FC_regularization

The four-grade head remains directly comparable with the existing Custom CNN.
The risk head predicts y >= 2 and is evaluated explicitly rather than being
silently substituted for the four-grade probabilities.

Important
---------
- Model/checkpoint/threshold selection uses validation data only.
- The locked test set is evaluated for audit, not used for selection.
- Default sampler remains the same inverse-four-grade sampler as the strong
  Protocol-V2 Custom CNN baseline.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0007723)
    parser.add_argument("--weight-decay", type=float, default=0.002453)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fc-l2", type=float, default=0.01)
    parser.add_argument(
        "--risk-loss-weight",
        type=float,
        default=0.5,
        help="Lambda applied to the auxiliary BCE risk loss.",
    )
    parser.add_argument(
        "--risk-pos-weight",
        type=float,
        default=1.0,
        help=(
            "Positive-class multiplier for BCEWithLogitsLoss. Keep 1.0 with "
            "the default inverse-four-grade sampler to avoid double balancing."
        ),
    )
    parser.add_argument(
        "--target-risk-recall",
        type=float,
        default=0.90,
        help="Validation recall constraint used by the safety checkpoint.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampler", choices=["weighted", "none"], default="weighted")
    parser.add_argument(
        "--cv-fold",
        type=int,
        default=-1,
        help=(
            "-1 uses protocol_v2_split. Values 0..4 use the corresponding "
            "development fold while the test patients remain locked."
        ),
    )
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-monitor",
        choices=[
            "val_joint_loss",
            "val_grade_loss",
            "val_qwk",
            "val_risk_auprc",
            "val_safety_specificity",
        ],
        default="val_joint_loss",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
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


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    path_candidates = ["roi_path", "path", "image_path"]
    label_candidates = ["severity", "y_true", "label", "class_id", "grade"]
    split_candidates = ["protocol_v2_split", "split_patient", "split"]

    path_col = next((c for c in path_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)
    split_col = next((c for c in split_candidates if c in df.columns), None)

    if not path_col or not label_col or not split_col:
        raise ValueError(
            "Could not detect path/label/split columns. "
            f"Available columns: {list(df.columns)}"
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


class SharedEncoderDualHeadCNN(nn.Module):
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

        # Keep the original Custom-CNN trunk and four-grade head unchanged.
        self.fc_shared = nn.Linear(4 * 4 * 256, 32)
        self.grade_head = nn.Linear(32, 4)

        # Preserve the post-construction RNG state so adding this head does not
        # change future dropout/augmentation RNG streams merely through its
        # parameter initialization.
        cpu_rng_state = torch.get_rng_state()
        self.risk_head = nn.Linear(32, 1)
        torch.set_rng_state(cpu_rng_state)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        return self.dropout(F.relu(self.fc_shared(x)))

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        grade_logits = self.grade_head(features)
        risk_logits = self.risk_head(features).squeeze(1)
        return grade_logits, risk_logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    probability = np.asarray(probability, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]
        if index == bins - 1:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probability[mask].mean())
        observed = float(y_true[mask].mean())
        error += (count / max(total, 1)) * abs(confidence - observed)
    return float(error)


def choose_threshold_for_recall(
    y_true: np.ndarray,
    score: np.ndarray,
    target_recall: float,
) -> Tuple[float, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0], dtype=float),
                score,
                np.nextafter(score, -np.inf),
            ]
        )
    )
    candidates = np.sort(candidates)[::-1]

    best = None
    for threshold in candidates:
        prediction = (score >= threshold).astype(int)
        tp = int(((y_true == 1) & (prediction == 1)).sum())
        fn = int(((y_true == 1) & (prediction == 0)).sum())
        tn = int(((y_true == 0) & (prediction == 0)).sum())
        fp = int(((y_true == 0) & (prediction == 1)).sum())
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
        if best is None:
            best = candidate
            continue

        # Maximize specificity; ties prefer a higher threshold, then lower flag rate.
        key = (candidate["specificity"], candidate["threshold"], -candidate["flag_rate"])
        best_key = (best["specificity"], best["threshold"], -best["flag_rate"])
        if key > best_key:
            best = candidate

    if best is None:
        raise RuntimeError("No threshold satisfied the requested validation recall.")
    return float(best["threshold"]), best


def four_grade_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    high_recall = float(
        (high_true & high_pred).sum() / max(int(high_true.sum()), 1)
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_absolute_grade_error": float(np.mean(np.abs(y_true - y_pred))),
        "argmax_high_risk_recall": high_recall,
        "argmax_false_clear_rate": 1.0 - high_recall,
    }


def binary_probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(np.mean((probability - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, probability),
        "mean_probability": float(probability.mean()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    grade_criterion: nn.Module,
    risk_criterion: nn.Module,
    device: torch.device,
    risk_loss_weight: float,
    fc_l2: float,
    target_risk_recall: float,
):
    model.eval()
    joint_loss_sum = 0.0
    grade_loss_sum = 0.0
    risk_loss_sum = 0.0
    total = 0

    all_indices: List[int] = []
    all_true: List[int] = []
    all_grade_probability: List[np.ndarray] = []
    all_risk_probability: List[np.ndarray] = []

    for images, grades, risks, indices in loader:
        images = images.to(device, non_blocking=True)
        grades = grades.to(device, non_blocking=True)
        risks = risks.to(device, non_blocking=True)

        grade_logits, risk_logits = model(images)
        grade_loss = grade_criterion(grade_logits, grades)
        risk_loss = risk_criterion(risk_logits, risks)
        regularization = torch.zeros((), device=device)
        if fc_l2 > 0:
            regularization = fc_l2 * (
                model.fc_shared.weight.square().sum()
                + model.grade_head.weight.square().sum()
                + model.risk_head.weight.square().sum()
            )
        joint_loss = (
            grade_loss
            + risk_loss_weight * risk_loss
            + regularization
        )

        batch_size = grades.numel()
        total += batch_size
        joint_loss_sum += float(joint_loss.detach().cpu()) * batch_size
        grade_loss_sum += float(grade_loss.detach().cpu()) * batch_size
        risk_loss_sum += float(risk_loss.detach().cpu()) * batch_size

        grade_probability = torch.softmax(grade_logits, dim=1)
        risk_probability = torch.sigmoid(risk_logits)

        all_indices.extend(indices.numpy().tolist())
        all_true.extend(grades.cpu().numpy().tolist())
        all_grade_probability.append(grade_probability.cpu().numpy())
        all_risk_probability.append(risk_probability.cpu().numpy())

    grade_probability = np.concatenate(all_grade_probability, axis=0)
    risk_probability = np.concatenate(all_risk_probability, axis=0)
    y_true = np.asarray(all_true, dtype=int)
    y_pred = grade_probability.argmax(axis=1)
    y_risk = (y_true >= 2).astype(int)

    metrics = {
        "joint_loss": joint_loss_sum / max(total, 1),
        "grade_loss": grade_loss_sum / max(total, 1),
        "risk_loss": risk_loss_sum / max(total, 1),
    }
    metrics.update(four_grade_metrics(y_true, y_pred))

    aux_metrics = binary_probability_metrics(y_risk, risk_probability)
    class_sum_probability = grade_probability[:, 2] + grade_probability[:, 3]
    class_sum_metrics = binary_probability_metrics(y_risk, class_sum_probability)

    threshold, safety = choose_threshold_for_recall(
        y_risk,
        risk_probability,
        target_risk_recall,
    )

    metrics.update(
        {
            "risk_auroc": aux_metrics["auroc"],
            "risk_auprc": aux_metrics["auprc"],
            "risk_brier": aux_metrics["brier"],
            "risk_ece": aux_metrics["ece"],
            "risk_mean_probability": aux_metrics["mean_probability"],
            "class_sum_risk_auroc": class_sum_metrics["auroc"],
            "class_sum_risk_auprc": class_sum_metrics["auprc"],
            "safety_threshold": threshold,
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
                    / math.log(4.0)
                ).mean()
            ),
        }
    )

    return (
        metrics,
        np.asarray(all_indices, dtype=int),
        y_true,
        y_pred,
        grade_probability,
        risk_probability,
    )


def save_predictions(
    source_frame: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grade_probability: np.ndarray,
    risk_probability: np.ndarray,
    out_path: Path,
) -> None:
    output = source_frame.iloc[indices].reset_index(drop=True).copy()
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    output["correct"] = y_true == y_pred

    for class_index, column in enumerate(PROB_COLS):
        output[column] = grade_probability[:, class_index]

    output["risk_prob_aux"] = risk_probability
    output["risk_prob_class_sum"] = (
        grade_probability[:, 2] + grade_probability[:, 3]
    )
    output["true_high_risk"] = (y_true >= 2).astype(int)
    output["dangerous_false_clear_argmax"] = (
        (y_true >= 2) & (y_pred < 2)
    )
    output.to_csv(out_path, index=False, encoding="utf-8-sig")


def monitor_is_better(
    current: float,
    best: float,
    monitor: str,
    min_delta: float,
) -> bool:
    if monitor in {"val_joint_loss", "val_grade_loss"}:
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
    frame[label_col] = pd.to_numeric(
        frame[label_col],
        errors="raise",
    ).astype(int)
    if not frame[label_col].isin([0, 1, 2, 3]).all():
        raise ValueError("Four-grade labels must be 0,1,2,3.")

    split = frame[split_col].astype(str).str.lower()
    if args.cv_fold >= 0:
        if "protocol_v2_cv_fold" not in frame.columns:
            raise ValueError(
                "--cv-fold requires protocol_v2_cv_fold from Protocol V2."
            )
        fold = pd.to_numeric(
            frame["protocol_v2_cv_fold"],
            errors="raise",
        ).astype(int)
        development = split != "test"
        train_frame = frame.loc[
            development & (fold != args.cv_fold)
        ].copy()
        val_frame = frame.loc[
            development & (fold == args.cv_fold)
        ].copy()
        test_frame = frame.loc[split == "test"].copy()
        active_protocol = f"development_cv_fold_{args.cv_fold}"
    else:
        train_frame = frame.loc[split == "train"].copy()
        val_frame = frame.loc[split == "val"].copy()
        test_frame = frame.loc[split == "test"].copy()
        active_protocol = "fixed_protocol_v2_split"

    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError(
            f"Empty split: train={len(train_frame)}, "
            f"val={len(val_frame)}, test={len(test_frame)}"
        )

    train_transform, eval_transform = build_transforms(args.seed)
    train_dataset = ROIDataset(
        train_frame,
        path_col,
        label_col,
        train_transform,
    )
    val_dataset = ROIDataset(
        val_frame,
        path_col,
        label_col,
        eval_transform,
    )
    test_dataset = ROIDataset(
        test_frame,
        path_col,
        label_col,
        eval_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    sampler = None
    shuffle = True
    if args.sampler == "weighted":
        labels = train_frame[label_col].to_numpy(dtype=int)
        counts = np.bincount(labels, minlength=4)
        class_weights = np.divide(
            1.0,
            counts,
            out=np.zeros_like(counts, dtype=float),
            where=counts > 0,
        )
        sample_weights = torch.as_tensor(
            class_weights[labels],
            dtype=torch.double,
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
    model = SharedEncoderDualHeadCNN(dropout=args.dropout).to(device)
    grade_criterion = FocalLoss()
    risk_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            args.risk_pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )

    monitors = {
        "val_joint_loss": (float("inf"), "best_val_joint_loss.pth"),
        "val_grade_loss": (float("inf"), "best_val_grade_loss.pth"),
        "val_qwk": (-float("inf"), "best_val_qwk.pth"),
        "val_risk_auprc": (-float("inf"), "best_val_risk_auprc.pth"),
        "val_safety_specificity": (
            -float("inf"),
            "best_val_safety.pth",
        ),
    }
    best_epochs = {key: None for key in monitors}
    history: List[Dict[str, float]] = []
    no_improvement = 0

    print("=" * 88)
    print("M1 shared-encoder dual-head training")
    print(f"Device: {device}")
    print(f"Protocol: {active_protocol}")
    print(
        f"Train/Val/Test ROI: "
        f"{len(train_frame)}/{len(val_frame)}/{len(test_frame)}"
    )
    print(
        f"Risk loss weight={args.risk_loss_weight} | "
        f"risk pos weight={args.risk_pos_weight} | "
        f"target validation recall={args.target_risk_recall}"
    )
    print(
        f"Epochs={args.epochs} | min_epochs={args.min_epochs} | "
        f"patience={args.early_stop_patience} | "
        f"monitor={args.early_stop_monitor}"
    )
    print("=" * 88)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_joint_sum = 0.0
        train_grade_sum = 0.0
        train_risk_sum = 0.0
        train_count = 0

        for images, grades, risks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            grades = grades.to(device, non_blocking=True)
            risks = risks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                grade_logits, risk_logits = model(images)
                grade_loss = grade_criterion(grade_logits, grades)
                risk_loss = risk_criterion(risk_logits, risks)
                regularization = torch.zeros((), device=device)
                if args.fc_l2 > 0:
                    regularization = args.fc_l2 * (
                        model.fc_shared.weight.square().sum()
                        + model.grade_head.weight.square().sum()
                        + model.risk_head.weight.square().sum()
                    )
                joint_loss = (
                    grade_loss
                    + args.risk_loss_weight * risk_loss
                    + regularization
                )

            scaler.scale(joint_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = grades.numel()
            train_count += batch_size
            train_joint_sum += (
                float(joint_loss.detach().cpu()) * batch_size
            )
            train_grade_sum += (
                float(grade_loss.detach().cpu()) * batch_size
            )
            train_risk_sum += (
                float(risk_loss.detach().cpu()) * batch_size
            )

        val_metrics, _, _, _, _, _ = evaluate(
            model,
            val_loader,
            grade_criterion,
            risk_criterion,
            device,
            args.risk_loss_weight,
            args.fc_l2,
            args.target_risk_recall,
        )

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_joint_loss": train_joint_sum / max(train_count, 1),
            "train_grade_loss": train_grade_sum / max(train_count, 1),
            "train_risk_loss": train_risk_sum / max(train_count, 1),
        }
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(
            args.out / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        current_values = {
            "val_joint_loss": val_metrics["joint_loss"],
            "val_grade_loss": val_metrics["grade_loss"],
            "val_qwk": val_metrics["qwk"],
            "val_risk_auprc": val_metrics["risk_auprc"],
            "val_safety_specificity": val_metrics["safety_specificity"],
        }

        stop_monitor_improved = False
        for monitor, (best_value, filename) in list(monitors.items()):
            current = current_values[monitor]
            if monitor_is_better(
                current,
                best_value,
                monitor,
                args.min_delta,
            ):
                monitors[monitor] = (current, filename)
                best_epochs[monitor] = epoch
                torch.save(
                    {
                        "model_state_dict": copy.deepcopy(
                            model.state_dict()
                        ),
                        "epoch": epoch,
                        "monitor": monitor,
                        "monitor_value": current,
                        "validation_metrics": val_metrics,
                        "args": vars(args),
                    },
                    args.out / filename,
                )
                if monitor == args.early_stop_monitor:
                    stop_monitor_improved = True

        if stop_monitor_improved:
            no_improvement = 0
        else:
            no_improvement += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"joint={row['train_joint_loss']:.5f}/"
            f"{val_metrics['joint_loss']:.5f} "
            f"grade={val_metrics['grade_loss']:.5f} "
            f"risk={val_metrics['risk_loss']:.5f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"qwk={val_metrics['qwk']:.4f} "
            f"risk_auprc={val_metrics['risk_auprc']:.4f} "
            f"risk_auc={val_metrics['risk_auroc']:.4f} "
            f"spec@R{args.target_risk_recall:.2f}="
            f"{val_metrics['safety_specificity']:.4f} "
            f"flag={val_metrics['safety_flag_rate']:.4f}"
        )

        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs
            and no_improvement >= args.early_stop_patience
        ):
            print(
                f"Early stopping: no {args.early_stop_monitor} "
                f"improvement for {args.early_stop_patience} epochs."
            )
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": history[-1]["epoch"],
            "args": vars(args),
        },
        args.out / "last_model.pth",
    )

    history_frame = pd.DataFrame(history)

    plt.figure(figsize=(10, 5))
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
        history_frame["val_grade_loss"],
        label="Validation grade loss",
    )
    plt.plot(
        history_frame["epoch"],
        history_frame["val_risk_loss"],
        label="Validation risk loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("M1 dual-head loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "training_loss_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for column, label in [
        ("val_accuracy", "Accuracy"),
        ("val_macro_f1", "Macro F1"),
        ("val_qwk", "QWK"),
        ("val_risk_auprc", "Aux risk AUPRC"),
        ("val_risk_auroc", "Aux risk AUROC"),
        ("val_safety_specificity", "Specificity at target recall"),
    ]:
        plt.plot(history_frame["epoch"], history_frame[column], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.title("M1 validation metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "validation_metrics_curves.png", dpi=180)
    plt.close()

    checkpoint_results = {}
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
            val_metrics,
            val_indices,
            val_true,
            val_pred,
            val_grade_probability,
            val_risk_probability,
        ) = evaluate(
            model,
            val_loader,
            grade_criterion,
            risk_criterion,
            device,
            args.risk_loss_weight,
            args.fc_l2,
            args.target_risk_recall,
        )
        (
            test_metrics,
            test_indices,
            test_true,
            test_pred,
            test_grade_probability,
            test_risk_probability,
        ) = evaluate(
            model,
            test_loader,
            grade_criterion,
            risk_criterion,
            device,
            args.risk_loss_weight,
            args.fc_l2,
            args.target_risk_recall,
        )

        tag = monitor.replace("val_", "")
        save_predictions(
            val_frame,
            val_indices,
            val_true,
            val_pred,
            val_grade_probability,
            val_risk_probability,
            args.out / f"val_predictions_best_{tag}.csv",
        )
        save_predictions(
            test_frame,
            test_indices,
            test_true,
            test_pred,
            test_grade_probability,
            test_risk_probability,
            args.out / f"test_predictions_best_{tag}.csv",
        )

        # Apply the validation-selected auxiliary-risk threshold unchanged.
        threshold = float(val_metrics["safety_threshold"])
        test_aux_prediction = (
            test_risk_probability >= threshold
        ).astype(int)
        test_true_risk = (test_true >= 2).astype(int)
        tp = int(
            ((test_true_risk == 1) & (test_aux_prediction == 1)).sum()
        )
        fn = int(
            ((test_true_risk == 1) & (test_aux_prediction == 0)).sum()
        )
        tn = int(
            ((test_true_risk == 0) & (test_aux_prediction == 0)).sum()
        )
        fp = int(
            ((test_true_risk == 0) & (test_aux_prediction == 1)).sum()
        )
        test_threshold_metrics = {
            "threshold_selected_on_validation": threshold,
            "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "flag_rate": float(test_aux_prediction.mean()),
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
        }

        checkpoint_results[monitor] = {
            "epoch": int(checkpoint["epoch"]),
            "validation": val_metrics,
            "test": test_metrics,
            "test_at_validation_selected_aux_threshold": (
                test_threshold_metrics
            ),
        }

    summary = {
        "schema_version": "lss_m1_dual_head_v1",
        "method": {
            "name": "M1 shared encoder + four-grade head + high-risk auxiliary head",
            "encoder": "Protocol-V2 Custom CNN",
            "grade_loss": "four-class focal loss",
            "risk_loss": "binary BCEWithLogitsLoss for grade >= 2",
            "risk_loss_weight": args.risk_loss_weight,
            "risk_pos_weight": args.risk_pos_weight,
            "target_validation_recall": args.target_risk_recall,
        },
        "active_protocol": active_protocol,
        "cv_fold": args.cv_fold,
        "roi_counts": {
            "train": len(train_frame),
            "validation": len(val_frame),
            "test": len(test_frame),
        },
        "best_epochs": best_epochs,
        "checkpoint_results": checkpoint_results,
        "primary_checkpoint_rule": (
            "best_val_safety maximizes validation specificity while the "
            "auxiliary risk threshold satisfies the requested validation recall."
        ),
        "comparison_rule": (
            "Compare the four-grade head with the original CNN and compare "
            "risk_prob_aux with the R2 class-sum risk score. Do not select a "
            "checkpoint using locked-test results."
        ),
    }
    with open(
        args.out / "multitask_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("=" * 88)
    print("M1 dual-head training completed")
    print(f"Output: {args.out}")
    print(f"Best epochs: {best_epochs}")
    print("=" * 88)


if __name__ == "__main__":
    main()
