# PUBLIC RELEASE NOTE
# Machine-specific paths were replaced by placeholders only. Scientific
# model logic, objectives, hyperparameters, fold logic and checkpoint rules
# are preserved from the frozen source identified in the provenance table.
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
53_train_coral_protocol_v3_full468_oof.py

Patient-level Protocol V3 Full-468 OOF training for ordinal CNN baselines.

Supported methods
-----------------
CORAL
    Rank-consistent cumulative ordinal regression. The network predicts
    P(y > 0), P(y > 1), and P(y > 2) through one shared latent score and
    ordered thresholds.

CORN
    Conditional ordinal regression. The network predicts the conditional
    probabilities P(y > 0), P(y > 1 | y > 0), and
    P(y > 2 | y > 1); cumulative probabilities are formed by products.

Fair-comparison design
----------------------
- Same 64x64 grayscale ROI input as C0/M1.
- Same four-block Custom CNN encoder and 32-dimensional feature.
- Same augmentation, inverse-four-grade sampler, AdamW hyperparameters,
  dropout, FC regularization, epochs, and early-stopping settings.
- Main checkpoint: minimum validation ordinal loss.
- Development OOF only by default; separate test evaluation is disabled.

Outputs
-------
- val_predictions_best_loss.csv (primary ordinal baseline checkpoint)
- val_predictions_best_macro_f1.csv
- val_predictions_best_qwk.csv
- training_history.csv
- best_val_loss.pth / best_val_macro_f1.pth / best_val_qwk.pth
- ordinal_training_summary.json
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
        required=True,
    )
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
        help="Held-out Protocol V3 Full-468 full-cohort fold.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-monitor",
        choices=["val_loss", "val_macro_f1", "val_qwk"],
        default="val_loss",
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
        help=(
            "Keep disabled for full-cohort OOF. The separate test set must not "
            "be repeatedly evaluated."
        ),
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
        return tensor, grade, index


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
    """One latent score plus explicitly ordered thresholds."""

    def __init__(self, in_features: int = 32, num_classes: int = 4):
        super().__init__()
        if num_classes < 3:
            raise ValueError("CORAL requires at least three classes here.")
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


class OrdinalCNN(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ordinal_head(self.encoder(x))


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
        # The model construction already guarantees monotonicity. cummin is a
        # numerical guard and does not alter a valid rank-consistent output.
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


def fc_regularization(model: OrdinalCNN) -> torch.Tensor:
    value = model.encoder.fc_shared.weight.square().sum()
    if model.method == "coral":
        value = value + model.ordinal_head.score.weight.square().sum()
    else:
        value = value + model.ordinal_head.weight.square().sum()
    return value


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


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    risk_score = probability[:, 2] + probability[:, 3]
    high_binary = high_true.astype(int)

    return {
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
        "risk_auroc": float(roc_auc_score(high_binary, risk_score)),
        "risk_auprc": float(
            average_precision_score(high_binary, risk_score)
        ),
        "risk_brier": float(np.mean((risk_score - high_binary) ** 2)),
        "risk_ece": expected_calibration_error(high_binary, risk_score),
        "mean_max_probability": float(probability.max(axis=1).mean()),
        "mean_entropy_norm": float(
            (
                -(
                    probability
                    * np.log(np.clip(probability, 1e-12, 1.0))
                ).sum(axis=1)
                / math.log(4)
            ).mean()
        ),
    }


@torch.no_grad()
def evaluate(
    model: OrdinalCNN,
    loader: DataLoader,
    device: torch.device,
    method: str,
    fc_l2: float,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_indices: List[int] = []
    all_true: List[int] = []
    all_probability: List[np.ndarray] = []
    all_cumulative: List[np.ndarray] = []

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = ordinal_loss(logits, labels, method)
        if fc_l2 > 0:
            loss = loss + fc_l2 * fc_regularization(model)

        cumulative, probability = cumulative_and_class_probabilities(
            logits,
            method,
        )
        batch_count = labels.numel()
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count
        all_indices.extend(indices.numpy().tolist())
        all_true.extend(labels.cpu().numpy().tolist())
        all_probability.append(probability.cpu().numpy())
        all_cumulative.append(cumulative.cpu().numpy())

    probability = np.concatenate(all_probability, axis=0)
    cumulative = np.concatenate(all_cumulative, axis=0)
    y_true = np.asarray(all_true, dtype=int)
    y_pred = probability.argmax(axis=1)
    metrics = metrics_from_predictions(y_true, y_pred, probability)
    metrics["loss"] = total_loss / max(total_count, 1)
    return (
        metrics,
        np.asarray(all_indices, dtype=int),
        y_true,
        y_pred,
        probability,
        cumulative,
    )


def save_predictions(
    source_frame: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
    cumulative: np.ndarray,
    method: str,
    output_path: Path,
) -> None:
    prediction = source_frame.iloc[indices].reset_index(drop=True).copy()
    prediction["y_true"] = y_true
    prediction["y_pred"] = y_pred
    prediction["correct"] = y_true == y_pred
    prediction["ordinal_method"] = method.upper()

    for index, column in enumerate(PROB_COLS):
        prediction[column] = probability[:, index]
    for index, column in enumerate(CUMULATIVE_COLS):
        prediction[column] = cumulative[:, index]

    prediction["risk_prob_ordinal"] = (
        probability[:, 2] + probability[:, 3]
    )
    prediction.to_csv(output_path, index=False, encoding="utf-8-sig")


def better(
    current: float,
    best: float,
    monitor: str,
    min_delta: float,
) -> bool:
    if monitor == "val_loss":
        return current < best - min_delta
    return current > best + min_delta


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    frame = pd.read_csv(args.roi_splits)
    path_col, label_col, split_col = detect_columns(frame)
    if "protocol_v3_cv_fold" not in frame.columns:
        raise ValueError(
            "The manifest must contain protocol_v3_cv_fold generated by "
            "50_build_protocol_v3_full468_patient_cv.py."
        )

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
        raise ValueError(
            f"Empty split: train={len(train_frame)}, val={len(val_frame)}"
        )
    if args.evaluate_test and len(test_frame) == 0:
        raise ValueError("Test evaluation requested but test split is empty.")

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
    class_counts = np.bincount(
        train_frame[label_col].astype(int).to_numpy(),
        minlength=4,
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
    model = OrdinalCNN(args.ordinal_method, args.dropout).to(device)
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
        "val_loss": (float("inf"), "best_val_loss.pth"),
        "val_macro_f1": (-float("inf"), "best_val_macro_f1.pth"),
        "val_qwk": (-float("inf"), "best_val_qwk.pth"),
    }
    best_epochs = {monitor: None for monitor in monitors}
    history: List[Dict[str, float]] = []
    no_improvement = 0

    print("=" * 100)
    print(
        f"Ordinal baseline={args.ordinal_method.upper()} | device={device} | "
        f"seed={args.seed} | fold={args.cv_fold}"
    )
    print(
        f"Train/Val/Test ROI={len(train_frame)}/{len(val_frame)}/{len(test_frame)} | "
        f"test evaluation={'ENABLED' if args.evaluate_test else 'DISABLED'}"
    )
    print(f"Train class counts={class_counts.tolist()}")
    print(
        f"Epochs={args.epochs}, min_epochs={args.min_epochs}, "
        f"patience={args.early_stop_patience}, monitor={args.early_stop_monitor}"
    )
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = ordinal_loss(logits, labels, args.ordinal_method)
                if args.fc_l2 > 0:
                    loss = loss + args.fc_l2 * fc_regularization(model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = labels.numel()
            running_loss += float(loss.detach().cpu()) * batch_count
            seen += batch_count

        train_loss = running_loss / max(seen, 1)
        val_result, _, _, _, _, _ = evaluate(
            model,
            val_loader,
            device,
            args.ordinal_method,
            args.fc_l2,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
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
            "val_loss": val_result["loss"],
            "val_macro_f1": val_result["macro_f1"],
            "val_qwk": val_result["qwk"],
        }
        improved_stop_monitor = False

        for monitor, (best_value, filename) in list(monitors.items()):
            current = current_values[monitor]
            if better(current, best_value, monitor, args.min_delta):
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
                        "ordinal_method": args.ordinal_method,
                        "args": vars(args),
                    },
                    args.out / filename,
                )
                if monitor == args.early_stop_monitor:
                    improved_stop_monitor = True

        no_improvement = 0 if improved_stop_monitor else no_improvement + 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_result['loss']:.5f} "
            f"acc={val_result['accuracy']:.4f} "
            f"macro_f1={val_result['macro_f1']:.4f} "
            f"qwk={val_result['qwk']:.4f} "
            f"high_recall={val_result['high_risk_recall']:.4f} "
            f"auprc={val_result['risk_auprc']:.4f}"
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
            "args": vars(args),
        },
        args.out / "last_model.pth",
    )

    history_frame = pd.DataFrame(history)
    plt.figure(figsize=(9, 5))
    plt.plot(history_frame["epoch"], history_frame["train_loss"], label="Train loss")
    plt.plot(history_frame["epoch"], history_frame["val_loss"], label="Validation loss")
    if best_epochs["val_loss"] is not None:
        plt.axvline(
            best_epochs["val_loss"],
            linestyle="--",
            label="Best validation loss",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{args.ordinal_method.upper()} training loss")
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
        label="Moderate-or-severe recall",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.title(f"{args.ordinal_method.upper()} validation metrics")
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
        ) = evaluate(
            model,
            val_loader,
            device,
            args.ordinal_method,
            args.fc_l2,
        )

        tag = monitor.replace("val_", "")
        save_predictions(
            val_frame,
            validation_indices,
            validation_true,
            validation_pred,
            validation_probability,
            validation_cumulative,
            args.ordinal_method,
            args.out / f"val_predictions_best_{tag}.csv",
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
            ) = evaluate(
                model,
                test_loader,
                device,
                args.ordinal_method,
                args.fc_l2,
            )
            save_predictions(
                test_frame,
                test_indices,
                test_true,
                test_pred,
                test_probability,
                test_cumulative,
                args.ordinal_method,
                args.out / f"test_predictions_best_{tag}.csv",
            )

        checkpoint_results[monitor] = {
            "epoch": int(checkpoint["epoch"]),
            "validation": validation_metrics,
            "test": test_metrics,
        }

    summary = {
        "schema_version": "lss_ordinal_cnn_oof_v1",
        "ordinal_method": args.ordinal_method.upper(),
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
        "fair_comparison": {
            "encoder": "Same Custom CNN encoder and 32-dimensional feature as C0/M1",
            "input": "64x64 grayscale equalized ROI",
            "sampler": args.sampler,
            "optimizer": "AdamW",
            "main_checkpoint": "minimum validation ordinal loss",
        },
        "warning": (
            "For OOF baseline development, keep --no-evaluate-test. "
            "Use val_predictions_best_loss.csv as the primary ordinal "
            "baseline prediction."
        ),
    }
    with open(
        args.out / "ordinal_training_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)

    print("=" * 100)
    print(f"{args.ordinal_method.upper()} OOF training completed")
    print(f"Output: {args.out}")
    print(f"Best epochs: {best_epochs}")
    print("Locked test evaluation: " + ("ENABLED" if args.evaluate_test else "DISABLED"))
    print("=" * 100)


if __name__ == "__main__":
    main()
