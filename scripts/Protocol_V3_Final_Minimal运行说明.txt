Protocol V3 portable usage
===========================

Repository root:
  <REPOSITORY_ROOT>

Raw MRI and ROI files are downloaded separately and are not stored in the
public repository.

1. Create or supply a local ROI manifest
----------------------------------------

The local manifest must contain valid paths on the current machine. Keep it
outside Git, or place it below the ignored `outputs/` directory.

2. Build folds and run the main pipeline
----------------------------------------

From the repository root:

& ".\scripts\59_run_final_main_pipeline.ps1" `
  -InputManifest "<LOCAL_ROI_MANIFEST.csv>" `
  -MainSeed 42 `
  -BootstrapReps 5000 `
  -ContinueOnError

Default generated locations:

  outputs\protocol_v3_full468
  outputs\protocol_v3_final

3. Reuse an already generated local Protocol V3 manifest
--------------------------------------------------------

& ".\scripts\59_run_final_main_pipeline.ps1" `
  -SkipFoldBuild `
  -MainSeed 42 `
  -BootstrapReps 5000 `
  -ContinueOnError

4. Run only the main training jobs
----------------------------------

& ".\scripts\55_run_main_experiment_seed42.ps1" `
  -Manifest "<LOCAL_PROTOCOL_V3_MANIFEST.csv>" `
  -ResultsRoot "<LOCAL_RESULTS_ROOT>" `
  -MainSeed 42 `
  -ContinueOnError

5. Optional supplementary seed branch
-------------------------------------

& ".\scripts\60_run_optional_seed_sensitivity_branch.ps1" `
  -Manifest "<LOCAL_PROTOCOL_V3_MANIFEST.csv>" `
  -ResultsRoot "<LOCAL_RESULTS_ROOT>"

This branch is optional and is not required for the primary manuscript result.

Frozen values
-------------

- Patients: 468
- ROIs: 2,978
- Patient-level folds: 5
- Main seed: 42
- Risk-loss weight: 0.50
- Primary fusion: 60% DeiT + 40% CNN
- Bootstrap replicates: 5,000
