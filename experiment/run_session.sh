#!/usr/bin/env bash
# run_session.sh — run a full value_capture scanning session
#
# Usage: bash run_session.sh

set -e

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEEPMREYE_DIR="$EXPERIMENT_DIR/deepmreye_calibration"

# ── Activate conda environment ─────────────────────────────────────────────────

echo "Activating value_capture_psychopy environment..."
eval "$(conda shell.bash hook)"
conda activate value_capture_psychopy

# ── Prompts ────────────────────────────────────────────────────────────────────

read -rp "Subject ID: " Subject
read -rp "Session number: " Session

while true; do
    read -rp "Number of runs [8/10]: " NRuns
    [[ "$NRuns" == "8" || "$NRuns" == "10" ]] && break
done

DoCalib=false
if [[ -d "$DEEPMREYE_DIR" ]]; then
    read -rp "Run DeepMREye calibration before task runs? [y/N]: " CalibInput
    [[ "$CalibInput" =~ ^[Yy]$ ]] && DoCalib=true
fi

echo ""
echo "════════════════════════════════════"
echo "  Subject:  $Subject"
echo "  Session:  $Session"
echo "  Runs:     $NRuns"
echo "  DeepMREye calibration: $DoCalib"
echo "════════════════════════════════════"
echo ""

# ── DeepMREye calibration (start) ─────────────────────────────────────────────

if $DoCalib; then
    echo ">>> Starting DeepMREye calibration (run 1)..."
    cd "$DEEPMREYE_DIR"
    python deepmreye_calib.py --subject "$Subject" --run 1
    echo ""
    echo ">>> Calibration complete."
    read -rp "Press Enter to start the task runs"
fi

# ── Task runs ──────────────────────────────────────────────────────────────────

cd "$EXPERIMENT_DIR"

for ((Run = 1; Run <= NRuns; Run++)); do
    echo ""
    echo "════════════════════════════════════"
    echo "  Run $Run / $NRuns"
    echo "════════════════════════════════════"
    read -rp "Press Enter to start run $Run (or type q to abort): " Confirm
    if [[ "$Confirm" == "q" ]]; then
        echo "Aborted before run $Run."
        exit 0
    fi

    python main.py "$Subject" "$Session" "$Run"

    echo ">>> Run $Run complete."
done

# ── DeepMREye calibration (end) ───────────────────────────────────────────────

if $DoCalib; then
    echo ""
    read -rp "Run DeepMREye calibration again at end of session? [y/N]: " CalibEnd
    if [[ "$CalibEnd" =~ ^[Yy]$ ]]; then
        echo ">>> Starting DeepMREye calibration (run 2)..."
        cd "$DEEPMREYE_DIR"
        python deepmreye_calib.py --subject "$Subject" --run 2
        echo ">>> End-of-session calibration complete."
    fi
fi

echo ""
echo "Session complete: subject $Subject, session $Session, $NRuns runs."
