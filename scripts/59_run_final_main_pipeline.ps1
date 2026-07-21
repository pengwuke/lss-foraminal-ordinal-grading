param(
    [string]$ScriptsDir = $PSScriptRoot,
    [string]$InputManifest = "",
    [string]$ProtocolRoot = (
        Join-Path (Split-Path -Parent $PSScriptRoot) `
          "outputs\protocol_v3_full468"
    ),
    [string]$ResultsRoot = (
        Join-Path (Split-Path -Parent $PSScriptRoot) `
          "outputs\protocol_v3_final"
    ),
    [int]$MainSeed = 42,
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [int]$BootstrapReps = 5000,
    [switch]$ContinueOnError,
    [switch]$SkipFoldBuild,
    [switch]$SkipTraining,
    [switch]$SkipAggregation,
    [switch]$SkipBootstrap,
    [switch]$SkipDiagnostics
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ScriptsDir)) {
    throw "Scripts directory not found: $ScriptsDir"
}
if ((-not $SkipFoldBuild) -and [string]::IsNullOrWhiteSpace($InputManifest)) {
    throw @"
-InputManifest is required when fold construction is enabled.

Pass a local ROI manifest that points to your separately downloaded data:
  -InputManifest "<LOCAL_ROI_MANIFEST.csv>"

The public repository intentionally contains no personal dataset path.
"@
}
if ((-not $SkipFoldBuild) -and (-not (Test-Path -LiteralPath $InputManifest))) {
    throw "Input ROI manifest not found: $InputManifest"
}

New-Item -ItemType Directory -Force -Path $ProtocolRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

$manifest = Join-Path $ProtocolRoot "roi_splits_protocol_v3_full468.csv"
$seedRoot = Join-Path $ResultsRoot ("main_seed_{0}" -f $MainSeed)
$analysisRoot = Join-Path $ResultsRoot ("main_analysis_seed_{0}" -f $MainSeed)

if (-not $SkipFoldBuild) {
    Write-Host ("=" * 120)
    Write-Host "Stage 1/5: create Full-468 fixed patient folds" -ForegroundColor Cyan
    Write-Host ("=" * 120)
    & python `
      (Join-Path $ScriptsDir "50_build_protocol_v3_full468_patient_cv.py") `
      --roi-manifest $InputManifest `
      --out $ProtocolRoot `
      --folds 5 `
      --candidate-seeds 20000 `
      --fold-search-base-seed 20260719 `
      --expected-patients 468 `
      --expected-roi 2978
    if ($LASTEXITCODE -ne 0) {
        throw "Protocol V3 fold construction failed."
    }
}

if (-not (Test-Path -LiteralPath $manifest)) {
    throw @"
Protocol V3 manifest missing:
  $manifest

Use -InputManifest to build it, or copy your already generated local fixed
manifest into the repository-relative ProtocolRoot and run with -SkipFoldBuild.
"@
}

if (-not $SkipTraining) {
    Write-Host ("=" * 120)
    Write-Host "Stage 2/5: main training only (one fixed seed)" -ForegroundColor Cyan
    Write-Host ("=" * 120)
    & (Join-Path $ScriptsDir "55_run_main_experiment_seed42.ps1") `
      -ScriptsDir $ScriptsDir `
      -Manifest $manifest `
      -ResultsRoot $ResultsRoot `
      -MainSeed $MainSeed `
      -Folds $Folds `
      -ContinueOnError:$ContinueOnError
    if ($LASTEXITCODE -ne 0) {
        throw "Main training failed."
    }
}

if (-not $SkipAggregation) {
    Write-Host ("=" * 120)
    Write-Host "Stage 3/5: aggregate main-seed Full-468 OOF" -ForegroundColor Cyan
    Write-Host ("=" * 120)
    & python `
      (Join-Path $ScriptsDir "56_aggregate_main_seed42.py") `
      --seed-root $seedRoot `
      --out $analysisRoot `
      --main-seed $MainSeed `
      --folds 0 1 2 3 4 `
      --target-recall 0.90 `
      --expected-patients 468 `
      --expected-roi 2978 `
      --original-deit-weight 0.60 `
      --original-cnn-weight 0.40 `
      --sensitivity-deit-weight 0.25 `
      --sensitivity-cnn-weight 0.75
    if ($LASTEXITCODE -ne 0) {
        throw "Main aggregation failed."
    }
}

if (-not $SkipBootstrap) {
    Write-Host ("=" * 120)
    Write-Host "Stage 4/5: patient-cluster paired bootstrap" -ForegroundColor Cyan
    Write-Host ("=" * 120)

    foreach ($riskMode in @("native", "matched")) {
        & python `
          (Join-Path $ScriptsDir "57_patient_cluster_bootstrap_main_seed42.py") `
          --analysis-root $analysisRoot `
          --out (Join-Path $analysisRoot ("bootstrap_{0}" -f $riskMode)) `
          --main-seed $MainSeed `
          --folds 0 1 2 3 4 `
          --risk-mode $riskMode `
          --target-recall 0.90 `
          --bootstrap-reps $BootstrapReps `
          --bootstrap-seed 2026

        if ($LASTEXITCODE -ne 0) {
            throw ("{0}-risk bootstrap failed." -f $riskMode)
        }
    }
}

if (-not $SkipDiagnostics) {
    Write-Host ("=" * 120)
    Write-Host "Stage 5/5: manuscript diagnostics" -ForegroundColor Cyan
    Write-Host ("=" * 120)
    & python `
      (Join-Path $ScriptsDir "58_generate_main_diagnostics.py") `
      --analysis-root $analysisRoot `
      --out (Join-Path $analysisRoot "diagnostics_native") `
      --main-seed $MainSeed `
      --risk-mode native
    if ($LASTEXITCODE -ne 0) {
        throw "Main diagnostics failed."
    }
}

Write-Host ("=" * 120)
Write-Host "Protocol V3 FINAL MAIN pipeline completed" -ForegroundColor Green
Write-Host "Main seed: $MainSeed"
Write-Host "Manifest: $manifest"
Write-Host "Analysis: $analysisRoot"
Write-Host "Repeated seed training: NOT REQUIRED"
Write-Host "Lambda search: NOT REPEATED"
Write-Host "Separate test used: NO"
Write-Host ("=" * 120)
