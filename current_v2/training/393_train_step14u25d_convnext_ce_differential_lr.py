# PUBLIC RELEASE NOTE
# Machine-specific paths were replaced by placeholders only. Scientific
# model logic, objectives, hyperparameters, fold logic and checkpoint rules
# are preserved from the frozen source identified in the provenance table.
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step14U25D
==========

T2-native Strong Nominal Baseline Gate.

Model:
    ImageNet-pretrained torchvision ConvNeXt-Tiny
    end-to-end fine-tuning
    native 4-class CE

Frozen cohort:
    TRAIN = 2040 ROI / 327 patients
    VAL   = 319 ROI / 47 patients
    TEST  = 619 ROI / 94 patients

TEST policy:
    TEST count may be read from the manifest.
    TEST paths MUST NOT be mapped/resolved.
    TEST dataset MUST NOT be constructed.
    TEST inference MUST NOT be performed.

Input:
    official first-generation ROI PNG
    -> cv2.equalizeHist
    -> first-generation augmentation
    -> resize 224x224
    -> grayscale repeated to RGB
    -> ImageNet normalization

Training:
    seed 42
    200 fixed epochs
    no early stopping
    deterministic epoch-wise inverse-frequency weighted sampling
    AdamW
    LR = 1e-4
    WD = 0.05
    cosine decay to 1e-6

Selection:
    strict minimum full-validation unweighted CE loss only.
    Metrics do not participate in checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
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
    mean_absolute_error,
    recall_score,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
)

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:
    raise SystemExit(
        "Missing albumentations / albumentations.pytorch"
    ) from exc


VERSION = "1.0.0"
STEP = "14U25D"

CLASS_NAMES = [
    "Normal",
    "Mild",
    "Moderate",
    "Severe",
]

EXPECTED_TRAIN_ROI = 2040
EXPECTED_VAL_ROI = 319
EXPECTED_TEST_ROI = 619

EXPECTED_TRAIN_PATIENTS = 327
EXPECTED_VAL_PATIENTS = 47
EXPECTED_TEST_PATIENTS = 94

EXPECTED_TRAIN_COUNTS = [1376, 341, 178, 145]
EXPECTED_VAL_COUNTS = [213, 57, 28, 21]

EXPECTED_TRAIN_HASH = (
    "ff08d959986915c2ad0e0bfabd8b521c9acbc9cda4bd774a5518a5c4cdaf5e07"
)

EXPECTED_VAL_HASH = (
    "3c46e8579128a4d745208f60b7cb88790ce0a0d68daf6e71d671ccc15f777fbf"
)

WINDOWS_ROOT = r"C:\path\to\LSS"
LINUX_ROOT = "/path/to/LSS"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--roi-splits",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Classifier-head learning rate.",
    )

    p.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1e-5,
        help="ConvNeXt pretrained-backbone learning rate.",
    )

    p.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-6,
        help="Minimum classifier-head learning rate.",
    )

    p.add_argument(
        "--min-backbone-learning-rate",
        type=float,
        default=1e-7,
        help="Minimum pretrained-backbone learning rate.",
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    return p.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    cv2.setNumThreads(0)


def normalize_split(value: Any) -> str:
    text = str(value).strip().lower()

    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }

    if text not in aliases:
        raise ValueError(
            f"Unexpected split value: {value!r}"
        )

    return aliases[text]


def detect_columns(
    frame: pd.DataFrame,
) -> Dict[str, str]:

    candidates = {
        "path": [
            "roi_path",
            "path",
            "image_path",
        ],
        "label": [
            "severity",
            "label",
            "class_id",
            "grade",
        ],
        "split": [
            "split_patient",
            "protocol_v3_split",
            "protocol_v2_split",
            "split",
        ],
        "patient": [
            "patient_id",
            "patient",
            "case_id",
            "patient_key",
        ],
    }

    selected = {}

    for role, options in candidates.items():
        col = next(
            (
                x
                for x in options
                if x in frame.columns
            ),
            None,
        )

        if col is None:
            raise ValueError(
                f"Could not identify {role} column. "
                f"Available={list(frame.columns)}"
            )

        selected[role] = col

    return selected


def sample_hash(
    frame: pd.DataFrame,
    cols: Mapping[str, str],
) -> str:

    rows: List[str] = []

    for _, row in frame.iterrows():

        patient_raw = str(
            row[cols["patient"]]
        ).strip()

        try:
            numeric = float(patient_raw)

            if numeric.is_integer():
                patient = str(int(numeric))
            else:
                patient = patient_raw

        except Exception:
            patient = patient_raw

        roi_path = str(
            row[cols["path"]]
        ).strip()

        label = str(
            int(row[cols["label"]])
        )

        rows.append(
            f"{patient}|{roi_path}|{label}"
        )

    payload = "\n".join(
        sorted(rows)
    ).encode("utf-8")

    return sha256_bytes(payload)


def map_runtime_path(value: Any) -> str:
    original = str(value).strip()

    normalized = original.replace(
        "/",
        "\\",
    )

    root = WINDOWS_ROOT.replace(
        "/",
        "\\",
    )

    if normalized.lower().startswith(
        root.lower()
    ):
        suffix = normalized[
            len(root):
        ].lstrip("\\")

        parts = PureWindowsPath(
            suffix
        ).parts

        return str(
            Path(
                LINUX_ROOT,
                *parts,
            )
        )

    if original.startswith(
        LINUX_ROOT
    ):
        return original

    raise ValueError(
        f"Path outside frozen root: {original}"
    )


def load_frozen_development_cohort(
    roi_splits: Path,
):
    frame = pd.read_csv(
        roi_splits
    )

    cols = detect_columns(frame)

    split = frame[
        cols["split"]
    ].map(normalize_split)

    train = frame.loc[
        split == "train"
    ].copy()

    validation = frame.loc[
        split == "val"
    ].copy()

    # Important:
    # TEST is not copied into a dataframe used downstream.
    test_count = int(
        (split == "test").sum()
    )

    train[
        cols["label"]
    ] = pd.to_numeric(
        train[cols["label"]],
        errors="raise",
    ).astype(int)

    validation[
        cols["label"]
    ] = pd.to_numeric(
        validation[cols["label"]],
        errors="raise",
    ).astype(int)

    if len(train) != EXPECTED_TRAIN_ROI:
        raise RuntimeError(
            f"TRAIN count mismatch: {len(train)}"
        )

    if len(validation) != EXPECTED_VAL_ROI:
        raise RuntimeError(
            f"VAL count mismatch: {len(validation)}"
        )

    if test_count != EXPECTED_TEST_ROI:
        raise RuntimeError(
            f"TEST count mismatch: {test_count}"
        )

    train_counts = (
        np.bincount(
            train[
                cols["label"]
            ].to_numpy(),
            minlength=4,
        )
        .astype(int)
        .tolist()
    )

    val_counts = (
        np.bincount(
            validation[
                cols["label"]
            ].to_numpy(),
            minlength=4,
        )
        .astype(int)
        .tolist()
    )

    if train_counts != EXPECTED_TRAIN_COUNTS:
        raise RuntimeError(
            f"TRAIN class-count mismatch: "
            f"{train_counts}"
        )

    if val_counts != EXPECTED_VAL_COUNTS:
        raise RuntimeError(
            f"VAL class-count mismatch: "
            f"{val_counts}"
        )

    train_patients = set(
        train[
            cols["patient"]
        ].astype(str).str.strip()
    )

    val_patients = set(
        validation[
            cols["patient"]
        ].astype(str).str.strip()
    )

    if len(train_patients) != EXPECTED_TRAIN_PATIENTS:
        raise RuntimeError(
            "TRAIN patient-count mismatch"
        )

    if len(val_patients) != EXPECTED_VAL_PATIENTS:
        raise RuntimeError(
            "VAL patient-count mismatch"
        )

    overlap = (
        train_patients
        & val_patients
    )

    if overlap:
        raise RuntimeError(
            f"TRAIN/VAL patient overlap: "
            f"{sorted(overlap)}"
        )

    train_hash = sample_hash(
        train,
        cols,
    )

    val_hash = sample_hash(
        validation,
        cols,
    )

    if train_hash != EXPECTED_TRAIN_HASH:
        raise RuntimeError(
            "Frozen TRAIN hash mismatch: "
            f"{train_hash}"
        )

    if val_hash != EXPECTED_VAL_HASH:
        raise RuntimeError(
            "Frozen VAL hash mismatch: "
            f"{val_hash}"
        )

    # Hash is calculated BEFORE path mapping.
    train[
        "_original_manifest_roi_path"
    ] = train[
        cols["path"]
    ].astype(str)

    validation[
        "_original_manifest_roi_path"
    ] = validation[
        cols["path"]
    ].astype(str)

    train[
        cols["path"]
    ] = train[
        cols["path"]
    ].map(map_runtime_path)

    validation[
        cols["path"]
    ] = validation[
        cols["path"]
    ].map(map_runtime_path)

    missing_train = [
        p
        for p in train[
            cols["path"]
        ].astype(str)
        if not Path(p).is_file()
    ]

    missing_val = [
        p
        for p in validation[
            cols["path"]
        ].astype(str)
        if not Path(p).is_file()
    ]

    if missing_train:
        raise FileNotFoundError(
            f"Missing TRAIN ROI PNGs: "
            f"{missing_train[:10]}"
        )

    if missing_val:
        raise FileNotFoundError(
            f"Missing VAL ROI PNGs: "
            f"{missing_val[:10]}"
        )

    audit = {
        "train_roi": len(train),
        "validation_roi": len(validation),
        "sealed_test_roi_count_only": test_count,

        "train_patients": len(
            train_patients
        ),

        "validation_patients": len(
            val_patients
        ),

        "train_class_counts": train_counts,
        "validation_class_counts": val_counts,

        "train_hash": train_hash,
        "validation_hash": val_hash,

        "train_hash_matches_frozen": True,
        "validation_hash_matches_frozen": True,

        "path_mapping": {
            "windows_root": WINDOWS_ROOT,
            "linux_root": LINUX_ROOT,
            "mapping_applied_after_hash": True,
            "runtime_missing": 0,
        },

        "test_paths_resolved": False,
        "test_dataset_constructed": False,
        "test_dataloader_constructed": False,
        "test_inference_performed": False,
        "test_metrics_computed": False,
    }

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        cols,
        audit,
    )


def build_transforms(
    seed: int,
    image_size: int,
):
    train_transform = A.Compose(
        [
            A.HorizontalFlip(
                p=0.5
            ),

            A.VerticalFlip(
                p=0.5
            ),

            A.RandomRotate90(
                p=0.5
            ),

            A.Affine(
                translate_percent=(
                    -0.05,
                    0.05,
                ),
                scale=(
                    0.9,
                    1.1,
                ),
                rotate=(
                    -30,
                    30,
                ),
                p=0.7,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7,
            ),

            A.GaussNoise(
                p=0.3
            ),

            A.Resize(
                image_size,
                image_size,
            ),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ],
        seed=seed,
    )

    validation_transform = A.Compose(
        [
            A.Resize(
                image_size,
                image_size,
            ),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ],
        seed=seed,
    )

    return (
        train_transform,
        validation_transform,
    )


class ROIDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        path_col: str,
        label_col: str,
        transform,
    ):
        self.frame = (
            frame
            .reset_index(drop=True)
            .copy()
        )

        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(
        self,
        index: int,
    ):
        row = self.frame.iloc[
            index
        ]

        path = str(
            row[self.path_col]
        )

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read ROI: {path}"
            )

        # Exact historical preprocessing.
        image = cv2.equalizeHist(
            image
        )

        # ConvNeXt pretrained weights expect RGB.
        image = np.repeat(
            image[:, :, None],
            3,
            axis=2,
        )

        tensor = self.transform(
            image=image
        )["image"]

        label = int(
            row[self.label_col]
        )

        return (
            tensor,
            label,
            index,
        )


class FixedIndexSampler(
    Sampler[int]
):
    def __init__(
        self,
        indices: Sequence[int],
    ):
        self.indices = [
            int(x)
            for x in indices
        ]

    def __iter__(self):
        return iter(
            self.indices
        )

    def __len__(self):
        return len(
            self.indices
        )


def weighted_epoch_indices(
    labels: np.ndarray,
    seed: int,
    epoch: int,
) -> np.ndarray:

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        labels,
        minlength=4,
    ).astype(np.float64)

    class_weight = (
        1.0
        / counts
    )

    sample_weight = torch.as_tensor(
        class_weight[labels],
        dtype=torch.double,
    )

    generator = torch.Generator()

    generator.manual_seed(
        int(seed)
        + 1_000_003 * int(epoch)
    )

    sampled = torch.multinomial(
        sample_weight,
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )

    return (
        sampled
        .cpu()
        .numpy()
        .astype(np.int64)
    )


def index_hash(
    indices: np.ndarray,
) -> str:

    text = ",".join(
        str(int(x))
        for x in indices.tolist()
    )

    return sha256_bytes(
        text.encode("utf-8")
    )


def build_model() -> nn.Module:
    weights = (
        ConvNeXt_Tiny_Weights
        .IMAGENET1K_V1
    )

    model = convnext_tiny(
        weights=weights
    )

    in_features = (
        model.classifier[-1]
        .in_features
    )

    model.classifier[-1] = (
        nn.Linear(
            in_features,
            4,
        )
    )

    return model


def classwise_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[float]:

    return (
        recall_score(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
                3,
            ],
            average=None,
            zero_division=0,
        )
        .astype(float)
        .tolist()
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:

    recalls = classwise_recall(
        y_true,
        y_pred,
    )

    high_true = (
        y_true >= 2
    )

    high_pred = (
        y_pred >= 2
    )

    high_support = int(
        high_true.sum()
    )

    high_recall = (
        float(
            (
                high_true
                & high_pred
            ).sum()
            / high_support
        )
        if high_support > 0
        else float("nan")
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
            3,
        ],
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "qwk": float(
            cohen_kappa_score(
                y_true,
                y_pred,
                weights="quadratic",
            )
        ),

        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),

        "classwise_recall": {
            CLASS_NAMES[i]: float(
                recalls[i]
            )
            for i in range(4)
        },

        "moderate_to_severe_recall": (
            high_recall
        ),

        "moderate_to_severe_support": (
            high_support
        ),

        "moderate_to_severe_false_clear_rate": (
            1.0 - high_recall
            if math.isfinite(
                high_recall
            )
            else float("nan")
        ),

        "confusion_matrix": (
            cm.astype(int).tolist()
        ),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):

    model.eval()

    loss_sum = 0.0
    seen = 0

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    row_indices: List[int] = []

    for (
        images,
        labels,
        indices,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(
                device.type == "cuda"
            ),
        ):
            logits = model(
                images
            )

            loss = F.cross_entropy(
                logits,
                labels,
                reduction="mean",
            )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite validation CE loss"
            )

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        preds = probs.argmax(
            dim=1
        )

        n = int(
            labels.numel()
        )

        loss_sum += (
            float(
                loss.detach().cpu()
            )
            * n
        )

        seen += n

        y_true.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        y_pred.extend(
            preds.detach()
            .cpu()
            .tolist()
        )

        y_prob.extend(
            probs.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        row_indices.extend(
            indices.detach()
            .cpu()
            .tolist()
        )

    y_true_arr = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred_arr = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    y_prob_arr = np.asarray(
        y_prob,
        dtype=np.float32,
    )

    metrics = compute_metrics(
        y_true_arr,
        y_pred_arr,
    )

    metrics[
        "validation_ce_loss"
    ] = float(
        loss_sum / seen
    )

    return (
        metrics,
        np.asarray(
            row_indices,
            dtype=np.int64,
        ),
        y_true_arr,
        y_pred_arr,
        y_prob_arr,
    )


def save_predictions(
    frame: pd.DataFrame,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    cols: Mapping[str, str],
    path: Path,
):

    rows = frame.iloc[
        indices
    ].copy()

    output = pd.DataFrame(
        {
            "row_index": indices,
            "patient_id": rows[
                cols["patient"]
            ].to_numpy(),

            "roi_path_original": rows[
                "_original_manifest_roi_path"
            ].to_numpy(),

            "y_true": y_true,
            "y_pred": y_pred,

            "prob_0_normal": probs[:, 0],
            "prob_1_mild": probs[:, 1],
            "prob_2_moderate": probs[:, 2],
            "prob_3_severe": probs[:, 3],
        }
    )

    output.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def state_dict_sha256(
    state: Mapping[str, torch.Tensor],
) -> str:

    h = hashlib.sha256()

    for key in sorted(
        state.keys()
    ):
        tensor = (
            state[key]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(
            key.encode("utf-8")
        )

        h.update(
            str(
                tuple(
                    tensor.shape
                )
            ).encode("utf-8")
        )

        h.update(
            str(
                tensor.dtype
            ).encode("utf-8")
        )

        h.update(
            tensor.numpy()
            .tobytes()
        )

    return h.hexdigest()


def main():
    args = parse_args()

    if args.seed != 42:
        raise RuntimeError(
            "Frozen protocol requires seed=42"
        )

    if args.epochs != 200:
        raise RuntimeError(
            "Frozen protocol requires 200 epochs"
        )

    if args.batch_size != 32:
        raise RuntimeError(
            "Frozen protocol requires batch size 32"
        )

    if abs(
        args.learning_rate
        - 1e-4
    ) > 1e-12:
        raise RuntimeError(
            "Frozen protocol requires LR=1e-4"
        )

    if abs(
        args.backbone_learning_rate
        - 1e-5
    ) > 1e-12:
        raise RuntimeError(
            "Frozen correction protocol requires backbone LR=1e-5"
        )

    if abs(
        args.min_learning_rate
        - 1e-6
    ) > 1e-12:
        raise RuntimeError(
            "Frozen correction protocol requires head min LR=1e-6"
        )

    if abs(
        args.min_backbone_learning_rate
        - 1e-7
    ) > 1e-12:
        raise RuntimeError(
            "Frozen correction protocol requires backbone min LR=1e-7"
        )

    if abs(
        args.weight_decay
        - 0.05
    ) > 1e-12:
        raise RuntimeError(
            "Frozen protocol requires WD=0.05"
        )

    if args.image_size != 224:
        raise RuntimeError(
            "Frozen protocol requires 224x224"
        )

    if args.num_workers != 0:
        raise RuntimeError(
            "Frozen deterministic protocol "
            "requires num_workers=0"
        )

    if args.output.exists():
        if any(
            args.output.iterdir()
        ):
            raise RuntimeError(
                f"Output directory already exists "
                f"and is not empty: {args.output}"
            )
    else:
        args.output.mkdir(
            parents=True,
            exist_ok=False,
        )

    set_seed(
        args.seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required"
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Frozen protocol requires "
            "CUDA bfloat16 support"
        )

    device = torch.device(
        "cuda"
    )

    (
        train_frame,
        val_frame,
        cols,
        cohort_audit,
    ) = load_frozen_development_cohort(
        args.roi_splits.resolve()
    )

    write_json(
        args.output
        / "cohort_audit.json",
        cohort_audit,
    )

    train_transform, val_transform = (
        build_transforms(
            args.seed,
            args.image_size,
        )
    )

    train_dataset = ROIDataset(
        train_frame,
        cols["path"],
        cols["label"],
        train_transform,
    )

    val_dataset = ROIDataset(
        val_frame,
        cols["path"],
        cols["label"],
        val_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(
        "Loading ImageNet pretrained "
        "ConvNeXt-Tiny..."
    )

    model = build_model()

    total_parameters = int(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    trainable_parameters = int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    if (
        trainable_parameters
        != total_parameters
    ):
        raise RuntimeError(
            "ConvNeXt backbone is not fully trainable"
        )

    initial_state_sha = (
        state_dict_sha256(
            model.state_dict()
        )
    )

    model.to(
        device
    )

    # Step14U25D is the single authorized correction:
    # preserve pretrained features with a 10x lower backbone LR,
    # while allowing the newly initialized classifier to learn faster.
    head_parameters = list(
        model.classifier[-1].parameters()
    )

    head_parameter_ids = {
        id(p) for p in head_parameters
    }

    backbone_parameters = [
        p
        for p in model.parameters()
        if id(p) not in head_parameter_ids
    ]

    if not backbone_parameters or not head_parameters:
        raise RuntimeError(
            "Could not construct backbone/head parameter groups."
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
                "group_name": "pretrained_backbone",
            },
            {
                "params": head_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "group_name": "new_classifier",
            },
        ]
    )

    backbone_min_ratio = (
        args.min_backbone_learning_rate
        / args.backbone_learning_rate
    )

    head_min_ratio = (
        args.min_learning_rate
        / args.learning_rate
    )

    if abs(backbone_min_ratio - head_min_ratio) > 1e-12:
        raise RuntimeError(
            "Backbone/head cosine minimum ratios must match."
        )

    minimum_ratio = head_min_ratio

    def cosine_factor(epoch_index: int) -> float:
        progress = min(
            max(float(epoch_index) / float(args.epochs), 0.0),
            1.0,
        )

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            minimum_ratio
            + (1.0 - minimum_ratio) * cosine
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            cosine_factor,
            cosine_factor,
        ],
    )

    labels = (
        train_frame[
            cols["label"]
        ]
        .astype(int)
        .to_numpy()
    )

    protocol = {
        "version": VERSION,
        "step": STEP,

        "purpose": (
            "T2-native modern strong nominal "
            "baseline gate"
        ),

        "candidate_host": (
            "torchvision ConvNeXt-Tiny"
        ),

        "initialization": (
            "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
        ),

        "initial_state_sha256": (
            initial_state_sha
        ),

        "backbone_fully_trainable": True,

        "total_parameters": (
            total_parameters
        ),

        "trainable_parameters": (
            trainable_parameters
        ),

        "task": (
            "native 4-class nominal classification"
        ),

        "training_objective": (
            "unweighted CrossEntropyLoss"
        ),

        "input": (
            "official first-generation ROI PNG "
            "-> cv2.equalizeHist "
            "-> first-generation augmentation "
            "-> resize 224x224 "
            "-> grayscale repeated to RGB "
            "-> ImageNet normalization"
        ),

        "augmentation": {
            "HorizontalFlip": 0.5,
            "VerticalFlip": 0.5,
            "RandomRotate90": 0.5,
            "Affine": {
                "translate_percent": [
                    -0.05,
                    0.05,
                ],
                "scale": [
                    0.9,
                    1.1,
                ],
                "rotate": [
                    -30,
                    30,
                ],
                "p": 0.7,
            },
            "RandomBrightnessContrast": {
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "p": 0.7,
            },
            "GaussNoise_p": 0.3,
        },

        "optimizer": {
            "name": "AdamW",
            "classifier_learning_rate": (
                args.learning_rate
            ),
            "backbone_learning_rate": (
                args.backbone_learning_rate
            ),
            "learning_rate_ratio_head_to_backbone": 10.0,
            "weight_decay": (
                args.weight_decay
            ),
        },

        "scheduler": {
            "name": "groupwise cosine via LambdaLR",
            "T_max": args.epochs,
            "classifier_min_learning_rate": (
                args.min_learning_rate
            ),
            "backbone_min_learning_rate": (
                args.min_backbone_learning_rate
            ),
            "head_to_backbone_ratio_preserved": True,
        },

        "single_authorized_correction": (
            "reduce pretrained ConvNeXt backbone LR from 1e-4 "
            "to 1e-5 while retaining 1e-4 for the newly "
            "initialized four-class classifier"
        ),

        "batch_size": args.batch_size,
        "fixed_epochs": args.epochs,
        "early_stopping": False,

        "sampling": (
            "deterministic epoch-wise "
            "inverse-frequency weighted sampling "
            "with replacement"
        ),

        "checkpoint_selection": (
            "strict minimum full-validation "
            "unweighted CE loss only"
        ),

        "metric_used_for_selection": False,

        "mixed_precision": (
            "CUDA bfloat16 autocast"
        ),

        "num_workers": 0,

        "test_paths_resolved": False,
        "test_dataset_constructed": False,
        "test_inference_performed": False,
    }

    write_json(
        args.output
        / "frozen_protocol.json",
        protocol,
    )

    print("=" * 100)
    print(
        "STEP14U25D ? CONVNEXT-TINY CE STRONG BASELINE"
    )
    print("=" * 100)

    print(
        f"Device             : {device}"
    )

    print(
        f"GPU                : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"TRAIN / VAL        : "
        f"{len(train_frame)} / {len(val_frame)}"
    )

    print(
        "TEST inference     : NO"
    )

    print(
        f"Parameters         : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable          : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Initial state SHA  : "
        f"{initial_state_sha}"
    )

    print(
        "Objective          : native 4-class CE"
    )

    print(
        "Sampler            : deterministic weighted"
    )

    print(
        f"Head LR            : "
        f"{args.learning_rate}"
    )

    print(
        f"Backbone LR        : "
        f"{args.backbone_learning_rate}"
    )

    print(
        f"WD                 : "
        f"{args.weight_decay}"
    )

    print(
        "Scheduler          : groupwise cosine; "
        "10x LR ratio preserved"
    )

    print(
        f"Batch              : {args.batch_size}"
    )

    print(
        f"Epochs             : {args.epochs}"
    )

    print(
        "Early stopping     : NO"
    )

    print(
        "Checkpoint         : minimum validation CE ONLY"
    )

    print("=" * 100)

    history: List[
        Dict[str, Any]
    ] = []

    sampler_hashes: List[str] = []

    best_val_loss = float(
        "inf"
    )

    best_epoch = -1

    checkpoint_path = (
        args.output
        / "best_checkpoint_min_validation_ce.pth"
    )

    started = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        sampled_indices = (
            weighted_epoch_indices(
                labels,
                args.seed,
                epoch,
            )
        )

        epoch_sampler_hash = (
            index_hash(
                sampled_indices
            )
        )

        sampler_hashes.append(
            epoch_sampler_hash
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=FixedIndexSampler(
                sampled_indices
            ),
            num_workers=0,
            pin_memory=True,
        )

        model.train()

        train_loss_sum = 0.0
        train_seen = 0

        backbone_epoch_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        head_epoch_lr = float(
            optimizer.param_groups[1]["lr"]
        )

        for (
            images,
            batch_labels,
            _,
        ) in train_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            batch_labels = (
                batch_labels.to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):

                logits = model(
                    images
                )

                loss = F.cross_entropy(
                    logits,
                    batch_labels,
                )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"Non-finite training loss "
                    f"at epoch {epoch}"
                )

            loss.backward()

            optimizer.step()

            batch_n = int(
                batch_labels.numel()
            )

            train_loss_sum += (
                float(
                    loss.detach().cpu()
                )
                * batch_n
            )

            train_seen += (
                batch_n
            )

        train_ce = float(
            train_loss_sum
            / train_seen
        )

        (
            val_metrics,
            _,
            _,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        val_ce = float(
            val_metrics[
                "validation_ce_loss"
            ]
        )

        row = {
            "epoch": epoch,
            "backbone_learning_rate": backbone_epoch_lr,
            "head_learning_rate": head_epoch_lr,
            "train_ce_loss": train_ce,
            "validation_ce_loss": val_ce,
            "validation_accuracy": (
                val_metrics[
                    "accuracy"
                ]
            ),
            "validation_balanced_accuracy": (
                val_metrics[
                    "balanced_accuracy"
                ]
            ),
            "validation_macro_f1": (
                val_metrics[
                    "macro_f1"
                ]
            ),
            "validation_weighted_f1": (
                val_metrics[
                    "weighted_f1"
                ]
            ),
            "validation_qwk": (
                val_metrics[
                    "qwk"
                ]
            ),
            "validation_mae": (
                val_metrics[
                    "mae"
                ]
            ),
            "validation_moderate_to_severe_recall": (
                val_metrics[
                    "moderate_to_severe_recall"
                ]
            ),
            "sampler_sha256": (
                epoch_sampler_hash
            ),
        }

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            args.output
            / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if (
            val_ce
            < best_val_loss
        ):

            best_val_loss = (
                val_ce
            )

            best_epoch = (
                epoch
            )

            cpu_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model
                .state_dict()
                .items()
            }

            torch.save(
                {
                    "version": VERSION,
                    "step": STEP,
                    "epoch": epoch,
                    "validation_ce_loss": (
                        val_ce
                    ),
                    "model_state_dict": (
                        cpu_state
                    ),
                    "initial_state_sha256": (
                        initial_state_sha
                    ),
                    "protocol": protocol,
                },
                checkpoint_path,
            )

        scheduler.step()

        print(
            f"Epoch "
            f"{epoch:03d}/{args.epochs} "
            f"| bb_lr={backbone_epoch_lr:.8f} "
            f"| head_lr={head_epoch_lr:.8f} "
            f"| train_ce={train_ce:.6f} "
            f"| val_ce={val_ce:.6f} "
            f"| acc={val_metrics['accuracy']:.4f} "
            f"| bal={val_metrics['balanced_accuracy']:.4f} "
            f"| macroF1={val_metrics['macro_f1']:.4f} "
            f"| qwk={val_metrics['qwk']:.4f} "
            f"| mae={val_metrics['mae']:.4f} "
            f"| MSrec={val_metrics['moderate_to_severe_recall']:.4f} "
            f"| best_epoch={best_epoch}",
            flush=True,
        )

    if len(history) != args.epochs:
        raise RuntimeError(
            "Fixed 200-epoch protocol violated"
        )

    if (
        best_epoch < 1
        or not checkpoint_path.is_file()
    ):
        raise RuntimeError(
            "No selected checkpoint"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(
        device
    )

    (
        selected_metrics,
        selected_indices,
        selected_true,
        selected_pred,
        selected_prob,
    ) = evaluate(
        model,
        val_loader,
        device,
    )

    selected_loss = float(
        selected_metrics[
            "validation_ce_loss"
        ]
    )

    if abs(
        selected_loss
        - best_val_loss
    ) > 1e-7:

        raise RuntimeError(
            "Reloaded selected validation "
            f"loss mismatch: "
            f"{selected_loss} != "
            f"{best_val_loss}"
        )

    save_predictions(
        val_frame,
        selected_indices,
        selected_true,
        selected_pred,
        selected_prob,
        cols,
        args.output
        / "selected_validation_predictions.csv",
    )

    cm = np.asarray(
        selected_metrics[
            "confusion_matrix"
        ],
        dtype=int,
    )

    pd.DataFrame(
        cm,
        index=[
            f"true_{x}"
            for x in CLASS_NAMES
        ],
        columns=[
            f"pred_{x}"
            for x in CLASS_NAMES
        ],
    ).to_csv(
        args.output
        / "selected_validation_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    history_df = pd.DataFrame(
        history
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_ce_loss"],
        label="Training CE",
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "validation_ce_loss"
        ],
        label="Validation CE",
    )

    plt.scatter(
        [best_epoch],
        [best_val_loss],
        s=60,
        label=(
            f"Selected epoch "
            f"{best_epoch}"
        ),
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-entropy loss"
    )

    plt.title(
        "Step14U25C ConvNeXt-Tiny CE"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        args.output
        / "loss_curves.png",
        dpi=180,
    )

    plt.close()

    sampler_combined_sha = (
        sha256_bytes(
            "\n".join(
                sampler_hashes
            ).encode("utf-8")
        )
    )

    selected_state_sha = (
        state_dict_sha256(
            checkpoint[
                "model_state_dict"
            ]
        )
    )

    elapsed_seconds = float(
        time.time()
        - started
    )

    summary = {
        "version": VERSION,
        "step": STEP,

        "status": "completed",

        "model": (
            "ConvNeXt-Tiny"
        ),

        "task": (
            "native four-class CE"
        ),

        "cohort": cohort_audit,

        "protocol": protocol,

        "completed_epochs": (
            len(history)
        ),

        "early_stopping": False,

        "selected_checkpoint": {
            "criterion": (
                "minimum validation CE loss"
            ),
            "epoch": (
                best_epoch
            ),
            "validation_ce_loss": (
                best_val_loss
            ),
            "path": str(
                checkpoint_path
            ),
            "state_dict_sha256": (
                selected_state_sha
            ),
        },

        "selected_validation_metrics": (
            selected_metrics
        ),

        "sampler": {
            "epochs": args.epochs,
            "epoch_index_sha256": (
                sampler_hashes
            ),
            "combined_sha256": (
                sampler_combined_sha
            ),
            "sampling_with_replacement": True,
            "num_samples_per_epoch": (
                len(train_frame)
            ),
        },

        "runtime": {
            "device": str(
                device
            ),
            "gpu": (
                torch.cuda
                .get_device_name(0)
            ),
            "torch_version": (
                torch.__version__
            ),
            "elapsed_seconds": (
                elapsed_seconds
            ),
        },

        "test_safety": {
            "test_paths_resolved": False,
            "test_dataset_constructed": False,
            "test_dataloader_constructed": False,
            "test_inference_performed": False,
            "test_metrics_computed": False,
        },
    }

    write_json(
        args.output
        / "step14u25d_summary.json",
        summary,
    )

    report = f"""
# Step14U25C ? ConvNeXt-Tiny CE Strong Nominal Baseline

## Status

COMPLETED

## Cohort

- TRAIN ROI: {len(train_frame)}
- VAL ROI: {len(val_frame)}
- TEST ROI count only: {EXPECTED_TEST_ROI}
- TEST paths resolved: NO
- TEST inference: NO

## Model

- ConvNeXt-Tiny
- ImageNet1K V1 pretrained initialization
- End-to-end fine-tuning
- Native four-class CE
- Parameters: {total_parameters:,}
- Trainable: {trainable_parameters:,}

## Training

- Seed: {args.seed}
- Fixed epochs: {args.epochs}
- Early stopping: NO
- Batch size: {args.batch_size}
- AdamW classifier LR: {args.learning_rate}
- AdamW pretrained-backbone LR: {args.backbone_learning_rate}
- Head/backbone LR ratio: 10?
- AdamW WD: {args.weight_decay}
- Classifier minimum LR: {args.min_learning_rate}
- Backbone minimum LR: {args.min_backbone_learning_rate}
- Weighted sampling: YES
- Checkpoint selection: minimum full-validation CE only

## Selected checkpoint

- Epoch: {best_epoch}
- Validation CE: {best_val_loss:.8f}

## Selected validation metrics

- Accuracy: {selected_metrics['accuracy']:.6f}
- Balanced accuracy: {selected_metrics['balanced_accuracy']:.6f}
- Macro-F1: {selected_metrics['macro_f1']:.6f}
- Weighted-F1: {selected_metrics['weighted_f1']:.6f}
- QWK: {selected_metrics['qwk']:.6f}
- MAE: {selected_metrics['mae']:.6f}
- Moderate/severe recall: {selected_metrics['moderate_to_severe_recall']:.6f}

## Class-wise recall

- Normal: {selected_metrics['classwise_recall']['Normal']:.6f}
- Mild: {selected_metrics['classwise_recall']['Mild']:.6f}
- Moderate: {selected_metrics['classwise_recall']['Moderate']:.6f}
- Severe: {selected_metrics['classwise_recall']['Severe']:.6f}

## Test safety

- TEST paths resolved: NO
- TEST dataset constructed: NO
- TEST dataloader constructed: NO
- TEST inference performed: NO
- TEST metrics computed: NO
"""

    (
        args.output
        / "step14u25d_report.md"
    ).write_text(
        report.strip()
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "STEP14U25C COMPLETED"
    )
    print("=" * 100)

    print(
        f"Selected epoch     : "
        f"{best_epoch}"
    )

    print(
        f"Validation CE      : "
        f"{best_val_loss:.8f}"
    )

    print(
        f"Accuracy           : "
        f"{selected_metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced accuracy  : "
        f"{selected_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro-F1           : "
        f"{selected_metrics['macro_f1']:.6f}"
    )

    print(
        f"QWK                : "
        f"{selected_metrics['qwk']:.6f}"
    )

    print(
        f"MAE                : "
        f"{selected_metrics['mae']:.6f}"
    )

    print(
        f"M/S recall         : "
        f"{selected_metrics['moderate_to_severe_recall']:.6f}"
    )

    print(
        f"Normal recall      : "
        f"{selected_metrics['classwise_recall']['Normal']:.6f}"
    )

    print(
        f"Mild recall        : "
        f"{selected_metrics['classwise_recall']['Mild']:.6f}"
    )

    print(
        f"Moderate recall    : "
        f"{selected_metrics['classwise_recall']['Moderate']:.6f}"
    )

    print(
        f"Severe recall      : "
        f"{selected_metrics['classwise_recall']['Severe']:.6f}"
    )

    print(
        f"Elapsed seconds    : "
        f"{elapsed_seconds:.1f}"
    )

    print()
    print(
        "EARLY STOPPING     : NO"
    )

    print(
        "CHECKPOINT METRIC  : VALIDATION CE ONLY"
    )

    print(
        "TEST INFERENCE     : NO"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
