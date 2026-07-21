# Protocol V3 runbook

Use the frozen patient folds. Main seed: 42. Risk-loss weight: 0.50.
Primary fusion: 60/40. Bootstrap: 5,000 patient replicates.

```powershell
& ".\scripts\59_run_final_main_pipeline.ps1" `
  -SkipFoldBuild `
  -MainSeed 42 `
  -BootstrapReps 5000 `
  -ContinueOnError
```

Expected cohort: 468 patients and 2,978 ROIs.
