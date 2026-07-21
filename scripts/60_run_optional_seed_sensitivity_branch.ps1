param(
    [string]$ScriptsDir = $PSScriptRoot,
    [string]$Manifest = (
        Join-Path (Split-Path -Parent $PSScriptRoot) `
          "outputs\protocol_v3_full468\roi_splits_protocol_v3_full468.csv"
    ),
    [string]$ResultsRoot = (
        Join-Path (Split-Path -Parent $PSScriptRoot) `
          "outputs\protocol_v3_final"
    ),
    [int[]]$AdditionalSeeds = @(2026, 3407),
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

function Invoke-Run {
    param(
        [string]$Title,
        [string]$Marker,
        [string[]]$CommandArgs,
        [string]$LogPath
    )

    if (Test-Path -LiteralPath $Marker) {
        Write-Host "SKIP: $Title" -ForegroundColor Yellow
        return [pscustomobject]@{ title = $Title; status = "skipped" }
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativePreference = $null -ne (
        Get-Variable -Name PSNativeCommandUseErrorActionPreference `
          -ErrorAction SilentlyContinue
    )
    if ($hasNativePreference) {
        $previousNativePreference = $PSNativeCommandUseErrorActionPreference
    }

    try {
        $ErrorActionPreference = "Continue"
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $false
        }
        & python @CommandArgs 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }

    if ($exitCode -ne 0) {
        $result = [pscustomobject]@{
            title = $Title
            status = "failed"
            exit_code = $exitCode
        }
        if (-not $ContinueOnError) {
            throw "Training failed: $Title"
        }
        return $result
    }

    if (-not (Test-Path -LiteralPath $Marker)) {
        throw "Completion marker missing: $Marker"
    }

    return [pscustomobject]@{
        title = $Title
        status = "completed"
        exit_code = 0
    }
}

if (-not (Test-Path -LiteralPath $ScriptsDir)) {
    throw "Scripts directory not found: $ScriptsDir"
}
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw @"
Protocol V3 manifest not found:
  $Manifest

Pass -Manifest explicitly or create the repository-relative local manifest.
The optional seed branch is supplementary and is not required for the primary
manuscript result.
"@
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$statuses = @()

foreach ($seed in $AdditionalSeeds) {
    $seedRoot = Join-Path $ResultsRoot ("supplement_seed_{0}" -f $seed)
    foreach ($fold in $Folds) {
        $coralOut = Join-Path $seedRoot "CORAL\fold_$fold"
        $statuses += Invoke-Run `
          -Title "Supplement seed ${seed}: CORAL fold=$fold" `
          -Marker (Join-Path $coralOut "val_predictions_best_loss.csv") `
          -LogPath (Join-Path $coralOut "console.log") `
          -CommandArgs @(
              (Join-Path $ScriptsDir "53_train_coral_protocol_v3_full468_oof.py"),
              "--roi-splits", $Manifest, "--out", $coralOut,
              "--ordinal-method", "coral", "--seed", "$seed",
              "--cv-fold", "$fold", "--epochs", "200",
              "--early-stop-patience", "50", "--min-epochs", "60",
              "--early-stop-monitor", "val_loss", "--no-evaluate-test"
          )

        $riskOut = Join-Path $seedRoot "CORAL_RISK_lambda0p50\fold_$fold"
        $statuses += Invoke-Run `
          -Title "Supplement seed ${seed}: CORAL-Risk fold=$fold" `
          -Marker (Join-Path $riskOut "val_predictions_best_ordinal_loss.csv") `
          -LogPath (Join-Path $riskOut "console.log") `
          -CommandArgs @(
              (Join-Path $ScriptsDir "54_train_coral_risk_protocol_v3_full468_oof.py"),
              "--roi-splits", $Manifest, "--out", $riskOut,
              "--ordinal-method", "coral", "--risk-loss-weight", "0.50",
              "--risk-pos-weight", "1.0", "--target-risk-recall", "0.90",
              "--seed", "$seed", "--cv-fold", "$fold",
              "--epochs", "200", "--early-stop-patience", "50",
              "--min-epochs", "60", "--early-stop-monitor", "val_joint_loss",
              "--no-evaluate-test"
          )
    }
}

$statusPath = Join-Path $ResultsRoot "supplement_seed_sensitivity_training_status.csv"
$statuses | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $statusPath

Write-Host ("=" * 112)
Write-Host "OPTIONAL branch seed-sensitivity training finished" -ForegroundColor Green
Write-Host "This output belongs in supplementary material, not the primary result."
Write-Host "Status: $statusPath"
Write-Host ("=" * 112)
