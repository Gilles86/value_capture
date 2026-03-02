#!/usr/bin/env bash
# Creates both conda environments for the value_capture project:
#   value_capture_psychopy  — running the experiment
#   value_capture_analysis  — data analysis / notebooks
#
# Usage (from repo root):
#   bash create_env/setup.sh
#
# To create only one environment:
#   bash create_env/setup.sh psychopy
#   bash create_env/setup.sh analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPTOOLS_PATH="$(cd "$SCRIPT_DIR/../../abstract_values/libs/exptools2" 2>/dev/null && pwd)" || EXPTOOLS_PATH=""

create_psychopy() {
    echo "=== Creating value_capture_psychopy ==="
    conda env create -f "$SCRIPT_DIR/environment_psychopy.yml" --yes

    echo ""
    echo "--- Installing exptools2 ---"
    if [ -n "$EXPTOOLS_PATH" ] && [ -d "$EXPTOOLS_PATH" ]; then
        conda run -n value_capture_psychopy pip install -e "$EXPTOOLS_PATH"
        echo "    Installed from $EXPTOOLS_PATH"
    else
        echo "    WARNING: exptools2 not found. Install it manually:"
        echo "    conda activate value_capture_psychopy"
        echo "    pip install -e /path/to/abstract_values/libs/exptools2"
    fi

    echo ""
    echo "value_capture_psychopy ready."
    echo "  conda activate value_capture_psychopy"
    echo "  python experiment/main.py <subject> <session> <run>"
}

create_analysis() {
    echo "=== Creating value_capture_analysis ==="
    conda env create -f "$SCRIPT_DIR/environment_analysis.yml" --yes

    echo ""
    echo "value_capture_analysis ready."
    echo "  conda activate value_capture_analysis"
    echo "  jupyter lab"
}

TARGET="${1:-both}"

case "$TARGET" in
    psychopy) create_psychopy ;;
    analysis) create_analysis ;;
    both)
        create_psychopy
        echo ""
        create_analysis
        ;;
    *)
        echo "Unknown target '$TARGET'. Use: psychopy | analysis | both"
        exit 1
        ;;
esac

echo ""
echo "=== All done ==="
