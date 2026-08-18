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
