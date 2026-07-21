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
    [int]$MainSeed = 42,
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [switch]$ContinueOnError,
    [switch]$SkipOriginalCNN,
    [switch]$SkipOriginalDeiT,
    [switch]$SkipCORAL,
    [switch]$SkipCORALRisk
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
        return [pscustomobject]@{
            title = $Title
            status = "skipped"
            marker = $Marker
            log = $LogPath
        }
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    Write-Host ("=" * 112)
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * 112)

    $watch = [System.Diagnostics.Stopwatch]::StartNew()
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

        & python @CommandArgs 2>&1 |
          Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
        $watch.Stop()
    }

    if ($exitCode -ne 0) {
        $result = [pscustomobject]@{
            title = $Title
            status = "failed"
            exit_code = $exitCode
            minutes = [math]::Round($watch.Elapsed.TotalMinutes, 3)
            marker = $Marker
            log = $LogPath
        }
        Write-Host "FAILED: $Title" -ForegroundColor Red
        if (-not $ContinueOnError) {
            throw "Training failed: $Title"
        }
        return $result
    }

    if (-not (Test-Path -LiteralPath $Marker)) {
        throw "Completion marker missing after successful process: $Marker"
    }

    Write-Host ("DONE: {0} ({1:N2} min)" -f $Title, $watch.Elapsed.TotalMinutes) -ForegroundColor Green
    return [pscustomobject]@{
        title = $Title
        status = "completed"
        exit_code = 0
        minutes = [math]::Round($watch.Elapsed.TotalMinutes, 3)
        marker = $Marker
        log = $LogPath
    }
}

if (-not (Test-Path -LiteralPath $ScriptsDir)) {
    throw "Scripts directory not found: $ScriptsDir"
}
if (-not (Test-Path -LiteralPath $Manifest)) {
    throw @"
Protocol V3 manifest not found:
  $Manifest

Create a local manifest that points to your separately downloaded ROI data, or
pass -Manifest explicitly. Personal dataset paths are intentionally not stored
in this public repository.
"@
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$seedRoot = Join-Path $ResultsRoot ("main_seed_{0}" -f $MainSeed)
$statuses = @()

foreach ($fold in $Folds) {
    if (-not $SkipOriginalCNN) {
        $runOut = Join-Path $seedRoot "Original_CNN\fold_$fold"
        $statuses += Invoke-Run `
            -Title "Main seed ${MainSeed}: original CNN fold=$fold" `
            -Marker (Join-Path $runOut "val_predictions_best_loss.csv") `
            -LogPath (Join-Path $runOut "console.log") `
            -CommandArgs @(
                (Join-Path $ScriptsDir "51_train_original_cnn_protocol_v3_full468_oof.py"),
                "--roi-splits", $Manifest, "--out", $runOut,
                "--seed", "$MainSeed", "--cv-fold", "$fold",
                "--epochs", "200", "--early-stop-patience", "50",
                "--min-epochs", "60", "--early-stop-monitor", "val_loss",
                "--no-evaluate-test"
            )
    }

    if (-not $SkipOriginalDeiT) {
        $runOut = Join-Path $seedRoot "Original_DeiT\fold_$fold"
        $statuses += Invoke-Run `
            -Title "Main seed ${MainSeed}: original DeiT fold=$fold" `
            -Marker (Join-Path $runOut "val_predictions.csv") `
            -LogPath (Join-Path $runOut "console.log") `
            -CommandArgs @(
                (Join-Path $ScriptsDir "52_train_official_deit_protocol_v3_full468_oof.py"),
                "--roi-splits", $Manifest, "--out", $runOut,
                "--seed", "$MainSeed", "--cv-fold", "$fold",
                "--epochs", "20", "--batch-size", "16",
                "--learning-rate", "2e-5", "--no-evaluate-test"
            )
    }

    if (-not $SkipCORAL) {
        $runOut = Join-Path $seedRoot "CORAL\fold_$fold"
        $statuses += Invoke-Run `
            -Title "Main seed ${MainSeed}: CORAL fold=$fold" `
            -Marker (Join-Path $runOut "val_predictions_best_loss.csv") `
            -LogPath (Join-Path $runOut "console.log") `
            -CommandArgs @(
                (Join-Path $ScriptsDir "53_train_coral_protocol_v3_full468_oof.py"),
                "--roi-splits", $Manifest, "--out", $runOut,
                "--ordinal-method", "coral", "--seed", "$MainSeed",
                "--cv-fold", "$fold", "--epochs", "200",
                "--early-stop-patience", "50", "--min-epochs", "60",
                "--early-stop-monitor", "val_loss", "--no-evaluate-test"
            )
    }

    if (-not $SkipCORALRisk) {
        $runOut = Join-Path $seedRoot "CORAL_RISK_lambda0p50\fold_$fold"
        $statuses += Invoke-Run `
            -Title "Main seed ${MainSeed}: CORAL-Risk lambda=0.50 fold=$fold" `
            -Marker (Join-Path $runOut "val_predictions_best_ordinal_loss.csv") `
            -LogPath (Join-Path $runOut "console.log") `
            -CommandArgs @(
                (Join-Path $ScriptsDir "54_train_coral_risk_protocol_v3_full468_oof.py"),
                "--roi-splits", $Manifest, "--out", $runOut,
                "--ordinal-method", "coral", "--risk-loss-weight", "0.50",
                "--risk-pos-weight", "1.0", "--target-risk-recall", "0.90",
                "--seed", "$MainSeed", "--cv-fold", "$fold",
                "--epochs", "200", "--early-stop-patience", "50",
                "--min-epochs", "60", "--early-stop-monitor", "val_joint_loss",
                "--no-evaluate-test"
            )
    }
}

$statusPath = Join-Path $ResultsRoot ("main_seed_{0}_training_status.csv" -f $MainSeed)
$statuses | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $statusPath

$completed = @($statuses | Where-Object { $_.status -eq "completed" }).Count
$skipped = @($statuses | Where-Object { $_.status -eq "skipped" }).Count
$failed = @($statuses | Where-Object { $_.status -eq "failed" }).Count

Write-Host ("=" * 112)
Write-Host "Protocol V3 MAIN training finished" -ForegroundColor Green
Write-Host "Main seed: $MainSeed"
Write-Host "Runs: $($statuses.Count) | Completed: $completed | Skipped: $skipped | Failed: $failed"
Write-Host "Status: $statusPath"
Write-Host "Lambda search: NOT RUN (lambda fixed from development analysis)"
Write-Host "Additional seed training: NOT RUN"
Write-Host "Separate test evaluation: DISABLED"
Write-Host ("=" * 112)

if ($failed -gt 0) {
    exit 1
}
