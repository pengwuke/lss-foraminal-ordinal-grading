# Lumbar foraminal stenosis grading on sagittal T2-weighted MRI

Reproducibility repository for:

**Four-grade lumbar foraminal stenosis grading on sagittal T2-weighted MRI: a patient-disjoint evaluation of ordinal modelling, moderate-to-severe auxiliary supervision, and fixed DeiT fusion**

Author: Wuke Peng  
ORCID: 0009-0003-5791-3783

## Current paper release: v2.0.0

`current_v2/` contains the current-paper code and non-image reproducibility material.

- `training/`: **7 current-paper training programs covering 8 independently trained model families**. The unified Step06A1 trainer covers both ConvNeXt-CORAL and ConvNeXt-CORAL-MSaux.
- `historical_training_sources/`: retained predecessor code, including the pre-OOF CNN-MSaux trainer.
- `analysis_scripts/`: final journal-uploaded analysis/reproducibility scripts.
- `source_data/`: final journal-uploaded source-data package.
- `oof_predictions_and_folds/`: final sanitised OOF predictions and fold assignments.
- `provenance/`: family-to-source and checkpoint manifests.

### CNN-MSaux source identity
The current OOF CNN-MSaux source is `08b_train_multitask_cnn_risk_head_oof.py`. The earlier `08_train_multitask_cnn_risk_head.py` is retained as a historical predecessor and is not represented as the current OOF source.

### Models
Current checkpoints: https://huggingface.co/wuke2024/lss-foraminal-ordinal-grading-models

There are 8 branch families × 5 outer folds = 40 independent checkpoints. Fixed fusion has no separate checkpoint.

### Dataset
MRI data are not redistributed. Dataset DOI: 10.17632/rgb77xm3jf.4

Research/reproducibility use only; not a clinical device.

## Environment and setup

The frozen `v2.0.0` release includes an `environment.yml` that defines the
top-level software environment used for the public reproducibility package.

### Create the Conda environment

```bash
conda env create -f environment.yml
conda activate lss-protocol-v3
```

The frozen environment file specifies **Python 3.10** and includes NumPy,
pandas, scikit-learn, Matplotlib, OpenCV, PyTorch, torchvision, timm,
albumentations, and Pillow.

> **Version note.** The released `environment.yml` pins Python to 3.10 but does
> not pin exact versions for all Python packages, including PyTorch and
> torchvision. It should therefore be treated as the released top-level
> dependency specification rather than a fully locked byte-for-byte software
> environment.

### Verify the installation

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import timm, albumentations, cv2, pandas, sklearn; print('imports OK')"
```

GPU execution requires a locally compatible NVIDIA driver / CUDA runtime for the
installed PyTorch build. CPU-only execution remains possible for inspection and
many analysis utilities, but model training and checkpoint inference are
intended for GPU-enabled environments.

### Reproducibility layout

The current-paper materials are organised under `current_v2/`:

- `training/` — seven current-paper training programs covering eight
  independently trained model families; the unified Step06A1 trainer covers
  both ConvNeXt-CORAL and ConvNeXt-CORAL-MSaux.
- `analysis_scripts/` — journal-uploaded analysis and reproducibility scripts.
- `oof_predictions_and_folds/` — sanitised patient-disjoint OOF predictions and
  fold assignments.
- `source_data/` — source tables used for the manuscript analyses and figures.
- `provenance/` — family-to-training-source and checkpoint manifests.
- `historical_training_sources/` — retained predecessor code for provenance;
  these files are not additional current-paper model families.

The 40 current-paper model checkpoints are hosted in the companion Hugging Face
repository under the frozen `v2.0.0/` release path cited in the manuscript.

### Recommended starting point

For code inspection or reproduction from the current public landing page, start
with:

```bash
git clone https://github.com/pengwuke/lss-foraminal-ordinal-grading.git
cd lss-foraminal-ordinal-grading
conda env create -f environment.yml
conda activate lss-protocol-v3
```

The immutable `v2.0.0` tag remains the frozen public release identity used by
the manuscript; later `main` commits are documentation-only maintenance unless
explicitly stated otherwise.
