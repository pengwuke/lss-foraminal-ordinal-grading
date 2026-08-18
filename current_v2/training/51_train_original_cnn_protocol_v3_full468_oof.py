# PUBLIC RELEASE NOTE
# Machine-specific paths were replaced by placeholders only. Scientific
# model logic, objectives, hyperparameters, fold logic and checkpoint rules
# are preserved from the frozen source identified in the provenance table.
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
51_train_original_cnn_protocol_v3_full468_oof.py

Purpose
-------
Run a patient-level Custom CNN training audit without test-set model selection.
The script saves full 150-epoch curves, confidence curves, and validation-selected
checkpoints, so early stopping can be judged from evidence rather than intuition.

Default protocol
----------------
- Patient split from roi_splits.csv / split_patient
- 64x64 grayscale ROI
- Official-like shallow CNN: 16,64,128,256 channels; FC=32
- Focal loss + optional FC L2
- AdamW
- WeightedRandomSampler by default
- 200 epochs by default
- Early stopping patience 50 by default
- Best checkpoints selected ONLY on validation data
"""

from __future__ import annotations

import argparse
import copy
import os
import json
import math
import random
import sys
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
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
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
    p = argparse.ArgumentParser()
    p.add_argument("--roi-splits", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.0007723)
    p.add_argument("--weight-decay", type=float, default=0.002453)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--fc-l2", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sampler", choices=["weighted", "none"], default="weighted")
    p.add_argument(
        "--cv-fold",
        type=int,
        default=-1,
        help=(
            "-1 uses protocol_v3_split train/val/test. "
            "0..4 uses protocol_v3_cv_fold for full-cohort CV while keeping test locked."
        ),
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="50 is recommended for the noisy patient-level validation set; 10 reproduces the official rule; 0 disables early stopping.",
    )
    p.add_argument("--min-epochs", type=int, default=60)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument(
        "--early-stop-monitor",
        choices=["val_loss", "val_macro_f1", "val_qwk"],
        default="val_loss",
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable during full-cohort CV so the separate test set is not "
            "repeatedly evaluated."
        ),
    )
    return p.parse_args()


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
    label_candidates = ["severity", "y_true", "label", "class_id"]
    split_candidates = ["protocol_v3_split", "protocol_v2_split", "split_patient", "split"]

    path_col = next((c for c in path_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)
    split_col = next((c for c in split_candidates if c in df.columns), None)
    if not path_col or not label_col or not split_col:
        raise ValueError(
            f"Could not detect path/label/split columns. Columns: {list(df.columns)}"
        )
    return path_col, label_col, split_col


def build_transforms(seed: int):
    train_tf = A.Compose(
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
                brightness_limit=0.2, contrast_limit=0.2, p=0.7
            ),
            A.GaussNoise(p=0.3),
            A.Resize(64, 64),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ],
        seed=seed,
    )
    eval_tf = A.Compose(
        [
            A.Resize(64, 64),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ],
        seed=seed,
    )
    return train_tf, eval_tf


class ROIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, path_col: str, label_col: str, transform):
        self.df = df.reset_index(drop=True).copy()
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read ROI: {path}")
        image = cv2.equalizeHist(image)
        tensor = self.transform(image=image)["image"]
        label = int(row[self.label_col])
        return tensor, label, idx


class FlexibleCNN(nn.Module):
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
        self.fc1 = nn.Linear(4 * 4 * 256, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    high_support = int(high_true.sum())
    high_recall = (
        float((high_true & high_pred).sum() / high_support)
        if high_support > 0
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "high_risk_recall": high_recall,
        "high_risk_false_clear_rate": 1.0 - high_recall,
        "mean_absolute_grade_error": float(np.mean(np.abs(y_true - y_pred))),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    fc_l2: float,
):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_idx, all_true, all_prob = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        if fc_l2 > 0:
            loss = loss + fc_l2 * (
                model.fc1.weight.square().sum() + model.fc2.weight.square().sum()
            )
        prob = torch.softmax(logits, dim=1)
        n = y.numel()
        total_loss += float(loss.detach().cpu()) * n
        total_n += n
        all_idx.extend(idx.numpy().tolist())
        all_true.extend(y.cpu().numpy().tolist())
        all_prob.append(prob.cpu().numpy())

    prob = np.concatenate(all_prob, axis=0)
    y_true = np.asarray(all_true, dtype=int)
    y_pred = prob.argmax(axis=1)
    metrics = ordinal_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_n, 1)
    max_prob = prob.max(axis=1)
    entropy = -(prob * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1) / math.log(4)
    metrics["mean_max_probability"] = float(max_prob.mean())
    metrics["mean_entropy_norm"] = float(entropy.mean())
    return metrics, np.asarray(all_idx, dtype=int), y_true, y_pred, prob


def save_predictions(
    source_df: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray,
    out_path: Path,
) -> None:
    pred = source_df.iloc[indices].reset_index(drop=True).copy()
    pred["y_true"] = y_true
    pred["y_pred"] = y_pred
    pred["correct"] = y_true == y_pred
    for i, col in enumerate(PROB_COLS):
        pred[col] = prob[:, i]
    pred.to_csv(out_path, index=False, encoding="utf-8-sig")


def better(value: float, best: float, monitor: str, min_delta: float) -> bool:
    if monitor == "val_loss":
        return value < best - min_delta
    return value > best + min_delta


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.roi_splits)
    path_col, label_col, split_col = detect_columns(df)
    if args.cv_fold < 0:
        raise ValueError("Protocol V3 requires --cv-fold 0..4.")
    if "protocol_v3_cv_fold" not in df.columns:
        raise ValueError(
            "--cv-fold requires protocol_v3_cv_fold generated by "
            "50_build_protocol_v3_full468_patient_cv.py."
        )
    fold_series = pd.to_numeric(
        df["protocol_v3_cv_fold"], errors="raise"
    ).astype(int)
    train_df = df.loc[fold_series != args.cv_fold].copy()
    val_df = df.loc[fold_series == args.cv_fold].copy()
    test_df = df.iloc[0:0].copy()
    active_protocol = f"protocol_v3_full468_cv_fold_{args.cv_fold}"

    if min(len(train_df), len(val_df)) == 0:
        raise ValueError(
            f"Empty split detected: train={len(train_df)}, val={len(val_df)}"
        )
    if args.evaluate_test:
        raise ValueError(
            "Protocol V3 Full-468 has no separate test split. "
            "Use --no-evaluate-test."
        )

    train_tf, eval_tf = build_transforms(args.seed)
    train_ds = ROIDataset(train_df, path_col, label_col, train_tf)
    val_ds = ROIDataset(val_df, path_col, label_col, eval_tf)
    test_ds = ROIDataset(test_df, path_col, label_col, eval_tf)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    sampler = None
    shuffle = True
    if args.sampler == "weighted":
        labels = train_df[label_col].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=4)
        class_w = np.divide(
            1.0, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0
        )
        sample_w = torch.as_tensor(class_w[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(
            sample_w,
            num_samples=len(sample_w),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin,
        generator=generator if sampler is None else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexibleCNN(dropout=args.dropout).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    monitors = {
        "val_loss": (float("inf"), "best_val_loss.pth"),
        "val_macro_f1": (-float("inf"), "best_val_macro_f1.pth"),
        "val_qwk": (-float("inf"), "best_val_qwk.pth"),
    }
    best_epochs = {k: None for k in monitors}
    history: List[Dict[str, float]] = []
    no_improve = 0

    print("=" * 78)
    print(f"Device: {device}")
    print(f"Protocol: {active_protocol}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(
        f"Epochs={args.epochs}, early_stop_patience={args.early_stop_patience}, "
        f"monitor={args.early_stop_monitor}"
    )
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(x)
                loss = criterion(logits, y)
                if args.fc_l2 > 0:
                    loss = loss + args.fc_l2 * (
                        model.fc1.weight.square().sum()
                        + model.fc2.weight.square().sum()
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_n = y.numel()
            running += float(loss.detach().cpu()) * batch_n
            n_seen += batch_n

        train_loss = running / max(n_seen, 1)
        val_metrics, _, _, _, _ = evaluate(
            model, val_loader, criterion, device, args.fc_l2
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(
            args.out / "training_history.csv", index=False, encoding="utf-8-sig"
        )

        current_values = {
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["qwk"],
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
                        "args": vars(args),
                    },
                    args.out / filename,
                )
                if monitor == args.early_stop_monitor:
                    improved_stop_monitor = True

        if improved_stop_monitor:
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"qwk={val_metrics['qwk']:.4f} "
            f"high_recall={val_metrics['high_risk_recall']:.4f} "
            f"maxprob={val_metrics['mean_max_probability']:.4f}"
        )

        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs
            and no_improve >= args.early_stop_patience
        ):
            print(
                f"Early stopping: no {args.early_stop_monitor} improvement for "
                f"{args.early_stop_patience} epochs."
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

    hist = pd.DataFrame(history)

    plt.figure(figsize=(9, 5))
    plt.plot(hist["epoch"], hist["train_loss"], label="Train loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
    if best_epochs["val_loss"] is not None:
        plt.axvline(best_epochs["val_loss"], linestyle="--", label="Best val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Custom CNN training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "training_loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(hist["epoch"], hist["val_accuracy"], label="Validation accuracy")
    plt.plot(hist["epoch"], hist["val_macro_f1"], label="Validation macro F1")
    plt.plot(hist["epoch"], hist["val_qwk"], label="Validation QWK")
    plt.plot(
        hist["epoch"],
        hist["val_high_risk_recall"],
        label="Validation moderate-or-severe recall",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.title("Validation metrics by epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "validation_metrics_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        hist["epoch"],
        hist["val_mean_max_probability"],
        label="Mean maximum probability",
    )
    plt.plot(
        hist["epoch"],
        hist["val_mean_entropy_norm"],
        label="Mean normalized entropy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.title("Validation confidence diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "validation_confidence_curve.png", dpi=180)
    plt.close()

    checkpoint_results = {}
    for monitor, (_, filename) in monitors.items():
        ckpt_path = args.out / filename
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        val_m, val_idx, val_true, val_pred, val_prob = evaluate(
            model, val_loader, criterion, device, args.fc_l2
        )
        tag = monitor.replace("val_", "")
        save_predictions(
            val_df,
            val_idx,
            val_true,
            val_pred,
            val_prob,
            args.out / f"val_predictions_best_{tag}.csv",
        )

        test_m = None
        if args.evaluate_test:
            test_m, test_idx, test_true, test_pred, test_prob = evaluate(
                model, test_loader, criterion, device, args.fc_l2
            )
            save_predictions(
                test_df,
                test_idx,
                test_true,
                test_pred,
                test_prob,
                args.out / f"test_predictions_best_{tag}.csv",
            )

        checkpoint_results[monitor] = {
            "epoch": int(ckpt["epoch"]),
            "validation": val_m,
            "test": test_m,
        }

    summary = {
        "purpose": "Protocol-v2 training; final model choice must remain validation-based.",
        "active_protocol": active_protocol,
        "cv_fold": args.cv_fold,
        "train_roi": len(train_df),
        "val_roi": len(val_df),
        "test_roi": len(test_df),
        "test_evaluation_enabled": args.evaluate_test,
        "device": str(device),
        "best_epochs": best_epochs,
        "checkpoint_results": checkpoint_results,
        "warning": (
            "During OOF development runs, use --no-evaluate-test. "
            "Checkpoint selection must remain validation-based."
        ),
    }
    with open(args.out / "curve_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 78)
    print("Curve audit completed")
    print(f"Output: {args.out}")
    print(f"Best epochs: {best_epochs}")
    print("=" * 78)


if __name__ == "__main__":
    main()
