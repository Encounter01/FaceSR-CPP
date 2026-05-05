param(
    [string]$Experiment = "all",
    [switch]$Run,
    [switch]$Cpu,
    [string]$Binary = "build/bin/Release/facesr_train.exe"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path

$experiments = [ordered]@{
    "a1_l1_only" = "config/ablations/a1_l1_only.ini"
    "a2_l1_perceptual" = "config/ablations/a2_l1_perceptual.ini"
    "a3_full_nonprogressive" = "config/ablations/a3_full_nonprogressive.ini"
    "a4_three_stage" = "config/ablations/a4_three_stage.ini"
    "a5_three_stage_attention" = "config/ablations/a5_three_stage_attention.ini"
}

Push-Location $repoRoot
try {
    if (-not (Test-Path $Binary)) {
        throw "Missing training binary: $Binary. Build the Release target first."
    }

    $binaryPath = (Resolve-Path $Binary).Path

    if ($Experiment -eq "all") {
        $selected = @($experiments.Keys)
    }
    elseif ($experiments.Contains($Experiment)) {
        $selected = @($Experiment)
    }
    else {
        throw "Unknown experiment '$Experiment'. Valid values: all, $($experiments.Keys -join ', ')"
    }

    foreach ($name in $selected) {
        $configPath = $experiments[$name]
        if (-not (Test-Path $configPath)) {
            throw "Missing config for $name`: $configPath"
        }

        $argsList = @("--config", $configPath)
        if ($Cpu) {
            $argsList += "--cpu"
        }

        $quotedArgs = $argsList | ForEach-Object {
            if ($_ -match "\s") { "`"$_`"" } else { $_ }
        }
        Write-Host ""
        Write-Host "[$name]"
        Write-Host "& `"$binaryPath`" $($quotedArgs -join ' ')"

        if ($Run) {
            $logDir = "results/ablation_runs/_logs"
            New-Item -ItemType Directory -Force -Path $logDir | Out-Null
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $logPath = Join-Path $logDir "${name}_${timestamp}.log"

            & $binaryPath @argsList 2>&1 | Tee-Object -FilePath $logPath
            if ($LASTEXITCODE -ne 0) {
                throw "Experiment $name failed with exit code $LASTEXITCODE. See $logPath"
            }
        }
    }

    if (-not $Run) {
        Write-Host ""
        Write-Host "Dry run only. Add -Run to execute training."
    }
}
finally {
    Pop-Location
}
