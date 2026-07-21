#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
52_train_official_deit_protocol_v3_full468_oof.py

Protocol-V3 Full-468 patient-level reproduction of the official DeiT classifier component:
- DeiT-Tiny patch16 224, ImageNet pretrained
- 224x224, grayscale repeated to 3 channels
- batch size 16
- 20 epochs
- learning rate 2e-5
- inverse-frequency weighted cross entropy
- one-cycle learning-rate schedule
- full training curves and validation-selected checkpoint
"""

from __future__ import annotations

import argparse
import copy
import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:
    raise SystemExit(
        "Install dependencies: python -m pip install albumentations opencv-python"
    ) from exc

try:
    import timm
except ImportError as exc:
    raise SystemExit("Install timm: python -m pip install timm") from exc


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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=0)
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
        "--model-name",
        default="deit_tiny_patch16_224.fb_in1k",
        help="timm model name",
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep disabled for full-cohort OOF training. The separate test set "
            "is not evaluated unless explicitly enabled."
        ),
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic convolution and matrix-multiplication settings.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # PyTorch may otherwise select memory-efficient or Flash SDPA kernels whose
    # backward pass is non-deterministic on some CUDA/PyTorch combinations.
    # The math SDPA kernel preserves the DeiT architecture and equations while
    # avoiding that non-deterministic kernel warning. It may be slightly slower.
    if torch.cuda.is_available():
        cuda_backend = torch.backends.cuda
        if hasattr(cuda_backend, "enable_flash_sdp"):
            cuda_backend.enable_flash_sdp(False)
        if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
            cuda_backend.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backend, "enable_math_sdp"):
            cuda_backend.enable_math_sdp(True)

    torch.use_deterministic_algorithms(True, warn_only=False)
    cv2.setNumThreads(0)


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    path_col = next((c for c in ["roi_path", "path", "image_path"] if c in df), None)
    label_col = next(
        (c for c in ["severity", "y_true", "label", "class_id"] if c in df), None
    )
    split_col = next((c for c in ["protocol_v3_split", "protocol_v2_split", "split_patient", "split"] if c in df), None)
    if not path_col or not label_col or not split_col:
        raise ValueError(f"Required columns not found. Columns: {list(df.columns)}")
    return path_col, label_col, split_col


def build_transforms(seed: int):
    train_tf = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        seed=seed,
    )
    eval_tf = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ],
        seed=seed,
    )
    return train_tf, eval_tf


class ROIDataset(Dataset):
    def __init__(self, df, path_col, label_col, transform):
        self.df = df.reset_index(drop=True).copy()
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read ROI: {path}")
        image = cv2.equalizeHist(image)
        image = np.repeat(image[..., None], 3, axis=2)
        x = self.transform(image=image)["image"]
        y = int(row[self.label_col])
        return x, y, idx


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    high_true = y_true >= 2
    high_pred = y_pred >= 2
    high_recall = float(
        (high_true & high_pred).sum() / max(int(high_true.sum()), 1)
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
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n_total = 0.0, 0
    indices, ys, probs = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        p = torch.softmax(logits, dim=1)
        n = y.numel()
        loss_sum += float(loss.detach().cpu()) * n
        n_total += n
        indices.extend(idx.numpy().tolist())
        ys.extend(y.cpu().numpy().tolist())
        probs.append(p.cpu().numpy())
    prob = np.concatenate(probs, axis=0)
    y_true = np.asarray(ys, dtype=int)
    y_pred = prob.argmax(axis=1)
    result = metrics(y_true, y_pred)
    result["loss"] = loss_sum / max(n_total, 1)
    result["mean_max_probability"] = float(prob.max(axis=1).mean())
    entropy = -(prob * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1) / math.log(4)
    result["mean_entropy_norm"] = float(entropy.mean())
    return result, np.asarray(indices), y_true, y_pred, prob


def save_predictions(df, indices, y_true, y_pred, prob, path):
    out = df.iloc[indices].reset_index(drop=True).copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["correct"] = y_true == y_pred
    for i, col in enumerate(PROB_COLS):
        out[col] = prob[:, i]
    out.to_csv(path, index=False, encoding="utf-8-sig")


def main():
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
    fold = pd.to_numeric(
        df["protocol_v3_cv_fold"], errors="raise"
    ).astype(int)
    train_df = df.loc[fold != args.cv_fold].copy()
    val_df = df.loc[fold == args.cv_fold].copy()
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

    g = torch.Generator().manual_seed(args.seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        generator=g,
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

    labels = train_df[label_col].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=4)
    class_weights = np.divide(
        1.0, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0
    )
    class_weights = class_weights / class_weights.sum() * 4.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        args.model_name,
        pretrained=args.pretrained,
        num_classes=4,
        drop_rate=0.25,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_loss = float("inf")
    best_epoch = None
    history: List[Dict[str, float]] = []

    print("=" * 78)
    print(f"Official DeiT component | model={args.model_name}")
    print(f"Protocol={active_protocol}")
    print(f"Device={device} | train/val/test={len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Locked test evaluation enabled: {args.evaluate_test}")
    print(f"Class counts={counts.tolist()} | class weights={class_weights.tolist()}")
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_total = 0.0, 0
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n = y.numel()
            loss_sum += float(loss.detach().cpu()) * n
            n_total += n

        train_loss = loss_sum / max(n_total, 1)
        val_m, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        row.update({f"val_{k}": v for k, v in val_m.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(
            args.out / "training_history.csv", index=False, encoding="utf-8-sig"
        )

        if val_m["loss"] < best_loss:
            best_loss = val_m["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "model_name": args.model_name,
                    "args": vars(args),
                },
                args.out / "best_val_loss.pth",
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.5f} "
            f"val_loss={val_m['loss']:.5f} acc={val_m['accuracy']:.4f} "
            f"macro_f1={val_m['macro_f1']:.4f} qwk={val_m['qwk']:.4f} "
            f"high_recall={val_m['high_risk_recall']:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "model_name": args.model_name,
            "args": vars(args),
        },
        args.out / "last_model.pth",
    )

    hist = pd.DataFrame(history)
    plt.figure(figsize=(9, 5))
    plt.plot(hist["epoch"], hist["train_loss"], label="Train loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label="Best validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted cross-entropy loss")
    plt.title("DeiT training and validation loss")
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
    plt.title("DeiT validation metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out / "validation_metrics_curve.png", dpi=180)
    plt.close()

    ckpt = torch.load(args.out / "best_val_loss.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_m, val_idx, val_true, val_pred, val_prob = evaluate(
        model, val_loader, criterion, device
    )
    save_predictions(
        val_df,
        val_idx,
        val_true,
        val_pred,
        val_prob,
        args.out / "val_predictions.csv",
    )

    test_m = None
    if args.evaluate_test:
        test_m, test_idx, test_true, test_pred, test_prob = evaluate(
            model, test_loader, criterion, device
        )
        save_predictions(
            test_df,
            test_idx,
            test_true,
            test_pred,
            test_prob,
            args.out / "test_predictions.csv",
        )

    summary = {
        "active_protocol": active_protocol,
        "cv_fold": args.cv_fold,
        "locked_test_evaluated": args.evaluate_test,
        "protocol": {
            "component": "DeiT",
            "official_notebook_reference": {
                "architecture": "deit_tiny_patch16_224",
                "pretrained": True,
                "image_size": 224,
                "batch_size": 16,
                "epochs": 20,
                "learning_rate": 2e-5,
                "loss": "inverse-frequency weighted cross entropy",
                "schedule": "one cycle",
            },
            "implementation_note": (
                "Pure-PyTorch/timm implementation of the official component protocol; "
                "not bitwise-identical to fastai."
            ),
        },
        "best_epoch_by_validation_loss": best_epoch,
        "validation": val_m,
        "test": test_m,
        "class_counts": counts.tolist(),
        "class_weights": class_weights.tolist(),
        "determinism": {
            "training_seed": int(args.seed),
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
            "flash_sdp_enabled": False,
            "memory_efficient_sdp_enabled": False,
            "math_sdp_enabled": True,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
    }
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("DeiT component reproduction completed")
    print(f"Best epoch: {best_epoch}")
    print(f"Output: {args.out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
