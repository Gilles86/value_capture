# run_session.ps1 — run a full value_capture scanning session
#
# Usage: .\run_session.ps1
# The script activates the value_capture_psychopy conda environment automatically.

$ErrorActionPreference = "Stop"

$ExperimentDir = $PSScriptRoot
$DeepMREyeDir  = Join-Path $PSScriptRoot "deepmreye_calibration"

# ── Activate conda environment ────────────────────────────────────────────────

Write-Host "Activating value_capture_psychopy environment..."
(& conda "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate value_capture_psychopy

# ── Prompts ───────────────────────────────────────────────────────────────────

$Subject = Read-Host "Subject ID"
$Session = Read-Host "Session number"

do {
    $NRuns = Read-Host "Number of runs [8/10]"
} while ($NRuns -ne "8" -and $NRuns -ne "10")
$NRuns = [int]$NRuns

$DoCalib = $false
if (Test-Path $DeepMREyeDir) {
    $CalibInput = Read-Host "Run DeepMREye calibration before task runs? [y/N]"
    $DoCalib = $CalibInput -match '^[Yy]$'
}

Write-Host ""
Write-Host "════════════════════════════════════"
Write-Host "  Subject:  $Subject"
Write-Host "  Session:  $Session"
Write-Host "  Runs:     $NRuns"
Write-Host "  DeepMREye calibration: $DoCalib"
Write-Host "════════════════════════════════════"
Write-Host ""

# ── DeepMREye calibration (start) ────────────────────────────────────────────

if ($DoCalib) {
    Write-Host ">>> Starting DeepMREye calibration (run 1)..."
    Set-Location $DeepMREyeDir
    python deepmreye_calib.py --subject $Subject --run 1
    Write-Host ""
    Write-Host ">>> Calibration complete."
    Read-Host "Press Enter to start the task runs"
}

# ── Task runs ─────────────────────────────────────────────────────────────────

Set-Location $ExperimentDir

for ($Run = 1; $Run -le $NRuns; $Run++) {
    Write-Host ""
    Write-Host "════════════════════════════════════"
    Write-Host "  Run $Run / $NRuns"
    Write-Host "════════════════════════════════════"
    $Confirm = Read-Host "Press Enter to start run $Run (or type q to abort)"
    if ($Confirm -eq "q") {
        Write-Host "Aborted before run $Run."
        exit 0
    }

    python main.py $Subject $Session $Run

    Write-Host ">>> Run $Run complete."
}

# ── DeepMREye calibration (end) ──────────────────────────────────────────────

if ($DoCalib) {
    Write-Host ""
    $CalibEnd = Read-Host "Run DeepMREye calibration again at end of session? [y/N]"
    if ($CalibEnd -match '^[Yy]$') {
        Write-Host ">>> Starting DeepMREye calibration (run 2)..."
        Set-Location $DeepMREyeDir
        python deepmreye_calib.py --subject $Subject --run 2
        Write-Host ">>> End-of-session calibration complete."
    }
}

Write-Host ""
Write-Host "Session complete: subject $Subject, session $Session, $NRuns runs."
