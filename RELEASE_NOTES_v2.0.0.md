# Release notes — v2.0.0

Current-paper reproducibility release.

- 14 reader-facing systems.
- 8 independently trained branch families.
- 40 model checkpoints (5 folds per branch family).
- 7 current-paper training programs; one unified modern trainer covers two ConvNeXt families.
- CNN-MSaux current OOF source is `08b_train_multitask_cnn_risk_head_oof.py`.
- Earlier `08_train_multitask_cnn_risk_head.py` retained as historical predecessor.
- Final G4D-B source-data, OOF/fold and analysis-script packages reused.
- 40/40 public model state dictionaries tensor-exact to frozen sources.
