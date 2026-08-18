# PUBLIC RELEASE NOTE
# Machine-specific paths were replaced by placeholders only. Scientific
# model logic, objectives, hyperparameters, fold logic and checkpoint rules
# are preserved from the frozen source identified in the provenance table.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step06A1 - formal modern ConvNeXt-CORAL / ConvNeXt-CORAL-MSaux unified five-fold training.

Scientific contract was frozen in Step06A0-Fix1 BEFORE these new outer-fold results.

New trainings:
  - 5 x ConvNeXt-CORAL
  - 5 x ConvNeXt-CORAL-MSaux

For each outer fold:
  TEST = outer fold
  VAL  = fixed mapping {0:1, 1:2, 2:3, 3:4, 4:0}
  TRAIN = remaining three folds

Hard rules:
  - TEST is never used in training / checkpoint / hyperparameter selection.
  - Exactly 200 epochs; no early stopping.
  - Exact formal ConvNeXt-CE preprocessing and augmentation imported from authoritative source.
  - Exact formal deterministic epoch-wise inverse-frequency weighted sampler.
  - AdamW; pretrained backbone LR 1e-5; new head LR 1e-4; WD 0.05.
  - Matched cosine decay: backbone min 1e-7; head min 1e-6.
  - CORAL checkpoint = minimum full fixed-VAL unweighted CORAL loss.
  - CORAL-MSaux total = CORAL + 0.5 * BCE(y>=2).
  - CORAL-MSaux checkpoint = minimum homologous full fixed-VAL TOTAL loss.
  - Outer fold inference happens only AFTER checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, Sampler

# -----------------------------------------------------------------------------
# Frozen identities
# -----------------------------------------------------------------------------
CE_SOURCE = Path(
    "/path/to/LSS-MRI-AISSLab-Dataset/experiments/baseline_v1/scripts/"
    "393_train_step14u25d_convnext_ce_differential_lr.py"
)
EXPECTED_CE_SHA = "5eef55010d4c019d26601a270a78be5e51b264bb396ba3a8b70871b7f4dfae04"

MANIFEST = Path(
    "/path/to/LSS/baseline_v1/server_runs/"
    "step02a0_fix1_resolve_roi_paths_20260813/"
    "unified_5fold_manifest_server_resolved.csv"
)
EXPECTED_MANIFEST_SHA = "5e64c21b9a7d70cb9eae5c336a08ffc0a317c904b1be36a7d01f660dc4c56d22"

OUT_ROOT = Path(
    "/path/to/LSS/baseline_v1/server_runs/"
    "step06a1_convnext_coral_msaux_unified5fold_png_seed42"
)

SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
IMAGE_SIZE = 224
HEAD_LR = 1e-4
BACKBONE_LR = 1e-5
HEAD_MIN_LR = 1e-6
BACKBONE_MIN_LR = 1e-7
WEIGHT_DECAY = 0.05
LAMBDA_MSAUX = 0.5
FIXED_VAL = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}

PRIMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "qwk",
    "mae",
]

# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def hard(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(0)


def load_authoritative_ce():
    hard(CE_SOURCE.is_file(), f"MISSING_CE_SOURCE: {CE_SOURCE}")
    actual = sha256_file(CE_SOURCE)
    hard(actual == EXPECTED_CE_SHA, f"CE_SOURCE_SHA_MISMATCH actual={actual}")
    spec = importlib.util.spec_from_file_location("formal_convnext_ce", CE_SOURCE)
    hard(spec is not None and spec.loader is not None, "IMPORT_SPEC_FAIL")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_transform_pair(src):
    """Call authoritative CE build_transforms without reimplementing augmentation."""
    fn = getattr(src, "build_transforms", None)
    hard(fn is not None, "AUTHORITATIVE_CE_HAS_NO_build_transforms")
    sig = inspect.signature(fn)
    kwargs = {}
    args = []
    for name, p in sig.parameters.items():
        lname = name.lower()
        if lname in ("image_size", "size", "img_size"):
            kwargs[name] = IMAGE_SIZE
        elif lname == "seed":
            kwargs[name] = SEED
        elif p.default is inspect._empty:
            # Conservative positional fallback for common historical signature.
            if not args:
                args.append(IMAGE_SIZE)
            elif len(args) == 1:
                args.append(SEED)
            else:
                raise RuntimeError(f"UNSUPPORTED_build_transforms_SIGNATURE: {sig}")
    try:
        pair = fn(*args, **kwargs)
    except TypeError:
        # Two common historical signatures.
        try:
            pair = fn(IMAGE_SIZE)
        except TypeError:
            pair = fn(IMAGE_SIZE, SEED)
    hard(isinstance(pair, (tuple, list)) and len(pair) == 2,
         f"build_transforms did not return (train, eval): {type(pair)}")
    return pair[0], pair[1], str(sig)


class ROIDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        path = str(row["roi_path"])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read ROI: {path}")
        # Exact formal ConvNeXt preprocessing.
        image = cv2.equalizeHist(image)
        image = np.repeat(image[..., None], 3, axis=2)
        out = self.transform(image=image)
        x = out["image"]
        y = int(row["severity"])
        return x, y, i


class FixedIndexSampler(Sampler[int]):
    def __init__(self, indices):
        self.indices = [int(x) for x in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def epoch_weighted_indices(labels: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    """Exact formal CE deterministic epoch-wise inverse-frequency sampling."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    hard(np.all(counts > 0), f"EMPTY_CLASS_IN_TRAIN: {counts.tolist()}")
    class_weight = 1.0 / counts
    sample_weight = torch.as_tensor(class_weight[labels], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1_000_003 * int(epoch))
    sampled = torch.multinomial(
        sample_weight,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
    return sampled.cpu().numpy().astype(np.int64)


# -----------------------------------------------------------------------------
# Standard rank-consistent CORAL (NOT BB-CORAL)
# -----------------------------------------------------------------------------
def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class RankConsistentCoralHead(nn.Module):
    """Historical standard CORAL head: one latent score + ordered thresholds."""
    def __init__(self, in_features: int = 768, num_classes: int = 4):
        super().__init__()
        hard(num_classes == 4, "THIS_FROZEN_EXPERIMENT_REQUIRES_4_CLASSES")
        self.score = nn.Linear(in_features, 1, bias=False)
        self.threshold_base = nn.Parameter(torch.tensor(-1.0))
        initial_step = inverse_softplus(1.0)
        self.threshold_delta_raw = nn.Parameter(
            torch.full((num_classes - 2,), float(initial_step))
        )

    def ordered_thresholds(self):
        first = self.threshold_base.reshape(1)
        deltas = F.softplus(self.threshold_delta_raw)
        rest = self.threshold_base + torch.cumsum(deltas, dim=0)
        return torch.cat([first, rest], dim=0)

    def forward(self, features):
        score = self.score(features)
        thresholds = self.ordered_thresholds().reshape(1, -1)
        return score - thresholds


class ConvNeXtCoralHead(nn.Module):
    def __init__(self, in_features: int, use_msaux: bool):
        super().__init__()
        self.use_msaux = bool(use_msaux)
        self.coral = RankConsistentCoralHead(in_features, 4)
        self.msaux = nn.Linear(in_features, 1) if self.use_msaux else None

    def forward(self, features):
        coral_logits = self.coral(features)
        aux_logits = self.msaux(features).reshape(-1) if self.msaux is not None else None
        return coral_logits, aux_logits


def coral_targets(labels: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(3, device=labels.device).reshape(1, -1)
    return (labels.reshape(-1, 1) > thresholds).float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Standard unweighted CORAL cumulative BCE.
    return F.binary_cross_entropy_with_logits(logits, coral_targets(labels))


def probabilities_from_coral_logits(logits: torch.Tensor):
    cumulative = torch.sigmoid(logits)
    # Numerical guard only; rank-consistent head should already be monotone.
    cumulative = torch.cummin(cumulative, dim=1).values
    prob = torch.stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2],
        ],
        dim=1,
    )
    prob = torch.clamp(prob, min=1e-12)
    prob = prob / torch.clamp(prob.sum(dim=1, keepdim=True), min=1e-12)
    return cumulative, prob


def metrics_from_arrays(y_true: np.ndarray, prob: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(prob.argmax(axis=1), dtype=np.int64)
    recalls = recall_score(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "normal_recall": float(recalls[0]),
        "mild_recall": float(recalls[1]),
        "moderate_recall": float(recalls[2]),
        "severe_recall": float(recalls[3]),
        "ms_recall": float(recall_score(
            (y_true >= 2).astype(int), (y_pred >= 2).astype(int), zero_division=0
        )),
    }


# -----------------------------------------------------------------------------
# Model / optimizer
# -----------------------------------------------------------------------------
def build_model(src, use_msaux: bool):
    weights = src.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = src.convnext_tiny(weights=weights)
    in_features = int(model.classifier[-1].in_features)
    hard(in_features == 768, f"UNEXPECTED_CONVNEXT_FEATURE_DIM: {in_features}")
    model.classifier[-1] = ConvNeXtCoralHead(in_features, use_msaux)
    return model


def build_optimizer(model):
    head_parameters = list(model.classifier[-1].parameters())
    head_ids = {id(p) for p in head_parameters}
    backbone_parameters = [p for p in model.parameters() if id(p) not in head_ids]
    hard(bool(head_parameters) and bool(backbone_parameters), "PARAM_GROUP_BUILD_FAIL")
    opt = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR,
                "weight_decay": WEIGHT_DECAY,
                "group_name": "pretrained_backbone",
            },
            {
                "params": head_parameters,
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
                "group_name": "new_coral_msaux_head",
            },
        ]
    )
    min_ratio = HEAD_MIN_LR / HEAD_LR
    hard(abs(min_ratio - BACKBONE_MIN_LR / BACKBONE_LR) < 1e-12,
         "COSINE_MIN_RATIO_MISMATCH")

    def cosine_factor(epoch_index):
        progress = min(max(float(epoch_index) / float(EPOCHS), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=[cosine_factor, cosine_factor]
    )
    return opt, sched


# -----------------------------------------------------------------------------
# Train / evaluate
# -----------------------------------------------------------------------------
@dataclass
class EvalOutput:
    total_loss: float
    coral_loss: float
    aux_loss: float | None
    metrics: dict
    indices: np.ndarray
    y_true: np.ndarray
    probability: np.ndarray
    cumulative: np.ndarray
    aux_probability: np.ndarray | None


def compute_total(coral, aux, use_msaux):
    if use_msaux:
        return coral + LAMBDA_MSAUX * aux
    return coral


def train_one_epoch(model, loader, optimizer, device, use_msaux):
    model.train()
    total_sum = 0.0
    coral_sum = 0.0
    aux_sum = 0.0
    n_sum = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        coral_logits, aux_logits = model(images)
        l_coral = coral_loss(coral_logits, labels)
        if use_msaux:
            target = (labels >= 2).float()
            l_aux = F.binary_cross_entropy_with_logits(aux_logits, target)
            loss = l_coral + LAMBDA_MSAUX * l_aux
        else:
            l_aux = None
            loss = l_coral
        loss.backward()
        optimizer.step()
        n = int(labels.numel())
        total_sum += float(loss.detach().cpu()) * n
        coral_sum += float(l_coral.detach().cpu()) * n
        if l_aux is not None:
            aux_sum += float(l_aux.detach().cpu()) * n
        n_sum += n
    return {
        "train_total_loss": total_sum / max(n_sum, 1),
        "train_coral_loss": coral_sum / max(n_sum, 1),
        "train_aux_loss": aux_sum / max(n_sum, 1) if use_msaux else None,
    }


@torch.no_grad()
def evaluate(model, loader, device, use_msaux) -> EvalOutput:
    model.eval()
    total_sum = coral_sum = aux_sum = 0.0
    n_sum = 0
    idx_all, y_all, p_all, c_all, a_all = [], [], [], [], []
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        coral_logits, aux_logits = model(images)
        l_coral = coral_loss(coral_logits, labels)
        if use_msaux:
            target = (labels >= 2).float()
            l_aux = F.binary_cross_entropy_with_logits(aux_logits, target)
            loss = l_coral + LAMBDA_MSAUX * l_aux
        else:
            l_aux = None
            loss = l_coral
        cumulative, prob = probabilities_from_coral_logits(coral_logits)

        n = int(labels.numel())
        total_sum += float(loss.detach().cpu()) * n
        coral_sum += float(l_coral.detach().cpu()) * n
        if l_aux is not None:
            aux_sum += float(l_aux.detach().cpu()) * n
        n_sum += n

        idx_all.extend(indices.cpu().numpy().tolist())
        y_all.extend(labels.cpu().numpy().tolist())
        p_all.append(prob.cpu().numpy())
        c_all.append(cumulative.cpu().numpy())
        if use_msaux:
            a_all.append(torch.sigmoid(aux_logits).cpu().numpy())

    idx = np.asarray(idx_all, dtype=np.int64)
    y = np.asarray(y_all, dtype=np.int64)
    prob = np.concatenate(p_all, axis=0)
    cum = np.concatenate(c_all, axis=0)
    aux_prob = np.concatenate(a_all, axis=0) if use_msaux else None
    return EvalOutput(
        total_loss=total_sum / max(n_sum, 1),
        coral_loss=coral_sum / max(n_sum, 1),
        aux_loss=(aux_sum / max(n_sum, 1)) if use_msaux else None,
        metrics=metrics_from_arrays(y, prob),
        indices=idx,
        y_true=y,
        probability=prob,
        cumulative=cum,
        aux_probability=aux_prob,
    )


def save_predictions(source_frame, ev: EvalOutput, path: Path, arm: str, outer: int, val_fold: int):
    pred = source_frame.iloc[ev.indices].reset_index(drop=True).copy()
    pred["y_true"] = ev.y_true
    pred["y_pred"] = ev.probability.argmax(axis=1).astype(int)
    pred["correct"] = pred["y_true"].to_numpy() == pred["y_pred"].to_numpy()
    pred["arm"] = arm
    pred["outer_fold"] = outer
    pred["fixed_val_fold"] = val_fold
    for i, c in enumerate(["prob_0_normal","prob_1_mild","prob_2_moderate","prob_3_severe"]):
        pred[c] = ev.probability[:, i]
    for i, c in enumerate(["prob_gt_0","prob_gt_1","prob_gt_2"]):
        pred[c] = ev.cumulative[:, i]
    if ev.aux_probability is not None:
        pred["msaux_prob_y_ge_2"] = ev.aux_probability
    pred.to_csv(path, index=False, encoding="utf-8-sig")


def fold_complete(fold_dir: Path) -> bool:
    marker = fold_dir / "PASS_FOLD_COMPLETE.json"
    ckpt = fold_dir / "best_val_total_loss.pth"
    pred = fold_dir / "outer_predictions_frozen_checkpoint.csv"
    return marker.is_file() and ckpt.is_file() and pred.is_file()


def run_fold(src, manifest, arm, outer, device, train_tf, eval_tf):
    use_msaux = arm == "convnext_coral_msaux"
    val_fold = FIXED_VAL[outer]
    fold_dir = OUT_ROOT / arm / f"outer_{outer}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    if fold_complete(fold_dir):
        print(f"[SKIP] {arm} outer={outer}: PASS_FOLD_COMPLETE already present.")
        return

    fold_series = pd.to_numeric(manifest["protocol_v3_cv_fold"], errors="raise").astype(int)
    test_mask = fold_series == outer
    val_mask = fold_series == val_fold
    train_mask = ~(test_mask | val_mask)

    train_frame = manifest.loc[train_mask].reset_index(drop=True).copy()
    val_frame = manifest.loc[val_mask].reset_index(drop=True).copy()
    test_frame = manifest.loc[test_mask].reset_index(drop=True).copy()

    hard(len(train_frame)+len(val_frame)+len(test_frame)==2978, "SPLIT_ROW_SUM_FAIL")
    train_pat = set(train_frame["patient_id"].astype(str))
    val_pat = set(val_frame["patient_id"].astype(str))
    test_pat = set(test_frame["patient_id"].astype(str))
    hard(train_pat.isdisjoint(val_pat) and train_pat.isdisjoint(test_pat) and val_pat.isdisjoint(test_pat),
         "PATIENT_LEAKAGE_SPLIT_FAIL")

    split_audit = {
        "arm": arm,
        "outer_fold": outer,
        "val_fold": val_fold,
        "train_roi": len(train_frame),
        "val_roi": len(val_frame),
        "outer_roi": len(test_frame),
        "train_patients": len(train_pat),
        "val_patients": len(val_pat),
        "outer_patients": len(test_pat),
        "outer_used_for_training": False,
        "outer_used_for_checkpoint": False,
    }
    (fold_dir/"split_audit.json").write_text(json.dumps(split_audit,indent=2),encoding="utf-8")

    train_ds = ROIDataset(train_frame, train_tf)
    val_ds = ROIDataset(val_frame, eval_tf)
    test_ds = ROIDataset(test_frame, eval_tf)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    # IMPORTANT: outer loader is constructed here, but NOT iterated until the checkpoint
    # has been completely frozen. No TEST statistic enters training/selection.
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    set_seed(SEED)
    model = build_model(src, use_msaux).to(device)
    optimizer, scheduler = build_optimizer(model)

    history = []
    best_val = float("inf")
    best_epoch = None
    best_path = fold_dir / "best_val_total_loss.pth"

    print("="*118)
    print(f"TRAIN {arm} | outer={outer} | fixed VAL={val_fold} | device={device}")
    print(f"TRAIN/VAL/SEALED-OUTER ROI = {len(train_frame)}/{len(val_frame)}/{len(test_frame)}")
    print(f"TRAIN/VAL/SEALED-OUTER patients = {len(train_pat)}/{len(val_pat)}/{len(test_pat)}")
    print(f"MSaux={use_msaux} lambda={LAMBDA_MSAUX if use_msaux else 'NA'}")
    print("Checkpoint monitor = homologous full fixed-VAL TOTAL loss")
    print("Outer inference = FORBIDDEN until checkpoint freeze.")
    print("="*118)

    labels_np = train_frame["severity"].astype(int).to_numpy()

    for epoch in range(1, EPOCHS+1):
        sampled = epoch_weighted_indices(labels_np, SEED, epoch)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=FixedIndexSampler(sampled),
            num_workers=0,
            pin_memory=True,
        )
        bb_lr = float(optimizer.param_groups[0]["lr"])
        hd_lr = float(optimizer.param_groups[1]["lr"])

        tr = train_one_epoch(model, train_loader, optimizer, device, use_msaux)
        ev = evaluate(model, val_loader, device, use_msaux)

        row = {
            "epoch": epoch,
            "backbone_lr": bb_lr,
            "head_lr": hd_lr,
            **tr,
            "validation_total_loss": ev.total_loss,
            "validation_coral_loss": ev.coral_loss,
            "validation_aux_loss": ev.aux_loss,
            **{f"validation_{k}": v for k,v in ev.metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            fold_dir/"training_history.csv", index=False, encoding="utf-8-sig"
        )

        if ev.total_loss < best_val:
            best_val = float(ev.total_loss)
            best_epoch = int(epoch)
            cpu_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(
                {
                    "step":"Step06A1",
                    "arm":arm,
                    "outer_fold":outer,
                    "fixed_val_fold":val_fold,
                    "epoch":epoch,
                    "validation_total_loss":ev.total_loss,
                    "validation_coral_loss":ev.coral_loss,
                    "validation_aux_loss":ev.aux_loss,
                    "model_state_dict":cpu_state,
                    "protocol":{
                        "epochs":EPOCHS,
                        "batch_size":BATCH_SIZE,
                        "backbone_lr":BACKBONE_LR,
                        "head_lr":HEAD_LR,
                        "weight_decay":WEIGHT_DECAY,
                        "lambda_msaux":LAMBDA_MSAUX if use_msaux else None,
                        "checkpoint_monitor":"homologous full fixed-VAL total loss",
                    },
                },
                best_path,
            )

        print(
            f"{arm} outer={outer} "
            f"ep={epoch:03d}/{EPOCHS} "
            f"bb_lr={bb_lr:.8f} head_lr={hd_lr:.8f} "
            f"train_total={tr['train_total_loss']:.6f} "
            f"val_total={ev.total_loss:.6f} "
            f"val_coral={ev.coral_loss:.6f} "
            + (f"val_aux={ev.aux_loss:.6f} " if use_msaux else "")
            + f"BA={ev.metrics['balanced_accuracy']:.4f} "
              f"MF1={ev.metrics['macro_f1']:.4f} "
              f"QWK={ev.metrics['qwk']:.4f} "
              f"MAE={ev.metrics['mae']:.4f} "
              f"best_ep={best_epoch}"
        )
        scheduler.step()

    hard(best_path.is_file() and best_epoch is not None, "NO_BEST_CHECKPOINT")

    # Freeze checkpoint identity BEFORE any outer inference.
    best_sha = sha256_file(best_path)
    freeze = {
        "status":"PASS_CHECKPOINT_FROZEN_BEFORE_OUTER_INFERENCE",
        "arm":arm,
        "outer_fold":outer,
        "fixed_val_fold":val_fold,
        "best_epoch":best_epoch,
        "best_validation_total_loss":best_val,
        "checkpoint":str(best_path),
        "checkpoint_sha256":best_sha,
        "outer_inference_has_run":False,
    }
    (fold_dir/"checkpoint_freeze_before_outer.json").write_text(
        json.dumps(freeze,indent=2),encoding="utf-8"
    )
    print(f"CHECKPOINT FROZEN | {arm} outer={outer} | epoch={best_epoch} | SHA={best_sha}")

    # Only now load frozen checkpoint and run outer inference once.
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    outer_ev = evaluate(model, test_loader, device, use_msaux)
    pred_path = fold_dir/"outer_predictions_frozen_checkpoint.csv"
    save_predictions(test_frame, outer_ev, pred_path, arm, outer, val_fold)

    marker = {
        "status":"PASS_FOLD_COMPLETE",
        "arm":arm,
        "outer_fold":outer,
        "fixed_val_fold":val_fold,
        "best_epoch":best_epoch,
        "best_validation_total_loss":best_val,
        "checkpoint_sha256":best_sha,
        "outer_prediction_sha256":sha256_file(pred_path),
        "outer_metrics":outer_ev.metrics,
        "outer_used_for_training":False,
        "outer_used_for_checkpoint_selection":False,
    }
    (fold_dir/"PASS_FOLD_COMPLETE.json").write_text(
        json.dumps(marker,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(f"PASS_FOLD_COMPLETE | {arm} outer={outer} | OUTER metrics={outer_ev.metrics}")


def aggregate_arm(manifest, arm):
    frames=[]
    audits=[]
    for outer in range(5):
        d=OUT_ROOT/arm/f"outer_{outer}"
        marker=d/"PASS_FOLD_COMPLETE.json"
        pred=d/"outer_predictions_frozen_checkpoint.csv"
        hard(marker.is_file() and pred.is_file(), f"INCOMPLETE_ARM {arm} outer={outer}")
        f=pd.read_csv(pred)
        hard(len(f)>0, f"EMPTY_PRED {arm} outer={outer}")
        frames.append(f)
        audits.append(json.loads(marker.read_text(encoding="utf-8")))
    allf=pd.concat(frames,ignore_index=True)
    hard(len(allf)==2978, f"OOF_ROW_COUNT_FAIL {arm}: {len(allf)}")
    hard(allf["patient_id"].astype(str).nunique()==468, f"OOF_PATIENT_COUNT_FAIL {arm}")
    hard((allf.groupby("patient_id")["outer_fold"].nunique()==1).all(),
         f"OOF_PATIENT_FOLD_LEAK {arm}")
    hard(not allf["protocol_v4_row_id"].duplicated().any(),
         f"OOF_DUPLICATED_ROW_ID {arm}")

    # Sort back to frozen identity.
    allf=allf.sort_values("protocol_v4_row_id").reset_index(drop=True)
    y=allf["y_true"].to_numpy(int)
    prob=allf[["prob_0_normal","prob_1_mild","prob_2_moderate","prob_3_severe"]].to_numpy(float)
    metrics=metrics_from_arrays(y,prob)
    oof=OUT_ROOT/f"{arm}_OOF_predictions.csv"
    allf.to_csv(oof,index=False,encoding="utf-8-sig")
    summary={
        "status":"PASS_ARM_OOF_AGGREGATED",
        "arm":arm,
        "patients":468,
        "rois":2978,
        "metrics":metrics,
        "oof_path":str(oof),
        "oof_sha256":sha256_file(oof),
        "folds":audits,
    }
    (OUT_ROOT/f"{arm}_OOF_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print("="*118)
    print(f"PASS_ARM_OOF_AGGREGATED | {arm}")
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print(f"OOF: {oof}")
    print(f"SHA: {summary['oof_sha256']}")
    print("="*118)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=["coral","msaux","all"],default="all")
    ap.add_argument("--fold",type=int,choices=[0,1,2,3,4],default=None,
                    help="Optional single fold; omit to run all five.")
    args=ap.parse_args()

    OUT_ROOT.mkdir(parents=True,exist_ok=True)
    hard(MANIFEST.is_file(),f"MISSING_MANIFEST: {MANIFEST}")
    hard(sha256_file(MANIFEST)==EXPECTED_MANIFEST_SHA,"MANIFEST_SHA_MISMATCH")
    manifest=pd.read_csv(MANIFEST)
    hard(len(manifest)==2978,"MANIFEST_ROW_COUNT_FAIL")
    hard(manifest["patient_id"].astype(str).nunique()==468,"MANIFEST_PATIENT_COUNT_FAIL")
    hard(manifest["protocol_v4_row_id"].nunique()==2978,"ROW_ID_UNIQUENESS_FAIL")
    hard(manifest["roi_path"].map(lambda p: Path(str(p)).is_file()).all(),
         "AT_LEAST_ONE_RESOLVED_ROI_PATH_MISSING")

    src=load_authoritative_ce()
    train_tf,eval_tf,tf_sig=build_transform_pair(src)

    set_seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hard(device.type=="cuda","FORMAL_TRAINING_REQUIRES_CUDA")

    protocol={
        "status":"STEP06A1_FORMAL_TRAINING_CONTRACT",
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest":str(MANIFEST),
        "manifest_sha256":sha256_file(MANIFEST),
        "authoritative_ce_source":str(CE_SOURCE),
        "authoritative_ce_sha256":sha256_file(CE_SOURCE),
        "build_transforms_signature":tf_sig,
        "device":str(device),
        "gpu":torch.cuda.get_device_name(0),
        "seed":SEED,
        "epochs":EPOCHS,
        "batch_size":BATCH_SIZE,
        "image_size":IMAGE_SIZE,
        "backbone_lr":BACKBONE_LR,
        "head_lr":HEAD_LR,
        "backbone_min_lr":BACKBONE_MIN_LR,
        "head_min_lr":HEAD_MIN_LR,
        "weight_decay":WEIGHT_DECAY,
        "coral":"standard unweighted cumulative BCE with historical rank-consistent head",
        "bb_coral_used":False,
        "msaux":"BCE[1(y>=2)]",
        "lambda_msaux":LAMBDA_MSAUX,
        "fixed_val_mapping":FIXED_VAL,
        "checkpoint_coral":"minimum full fixed-VAL CORAL loss",
        "checkpoint_msaux":"minimum full fixed-VAL (CORAL + 0.5*MSaux) total loss",
        "outer_used_for_training":False,
        "outer_used_for_selection":False,
        "fusion_weight_search":False,
    }
    (OUT_ROOT/"STEP06A1_FORMAL_TRAINING_CONTRACT.json").write_text(
        json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    print("="*118)
    print("STEP06A1 | FORMAL MODERN CONVNEXT-CORAL / MSAUX UNIFIED FIVE-FOLD")
    print(f"GPU            : {protocol['gpu']}")
    print(f"Manifest SHA   : {protocol['manifest_sha256']}")
    print(f"CE source SHA  : {protocol['authoritative_ce_sha256']}")
    print(f"Transform sig  : {tf_sig}")
    print("CORAL          : standard unweighted cumulative BCE; rank-consistent ordered thresholds")
    print("BB-CORAL       : NO")
    print("MSaux lambda   : 0.5 fixed")
    print("Outer TEST     : sealed until checkpoint freeze per fold")
    print("="*118)

    arms=[]
    if args.arm in ("coral","all"): arms.append("convnext_coral")
    if args.arm in ("msaux","all"): arms.append("convnext_coral_msaux")
    folds=[args.fold] if args.fold is not None else list(range(5))

    for arm in arms:
        for outer in folds:
            run_fold(src,manifest,arm,outer,device,train_tf,eval_tf)

    # Aggregate only if all 5 folds are complete for an arm.
    for arm in arms:
        if all(fold_complete(OUT_ROOT/arm/f"outer_{f}") for f in range(5)):
            aggregate_arm(manifest,arm)
        else:
            print(f"[INFO] {arm}: not all five folds complete yet; aggregate deferred.")

    if args.fold is None and args.arm=="all":
        hard(
            (OUT_ROOT/"convnext_coral_OOF_summary.json").is_file()
            and (OUT_ROOT/"convnext_coral_msaux_OOF_summary.json").is_file(),
            "FULL_RUN_FINISHED_WITHOUT_BOTH_OOF_SUMMARIES"
        )
        print("="*118)
        print("PASS_STEP06A1_CONVNEXT_CORAL_MSAUX_UNIFIED5FOLD")
        print(f"Result root: {OUT_ROOT}")
        print("New trained models: 10 = 5 CORAL + 5 CORAL-MSaux")
        print("NEXT: Step06A2 fixed DeiT fusion + 20,000 patient-cluster bootstrap.")
        print("="*118)


if __name__=="__main__":
    main()
