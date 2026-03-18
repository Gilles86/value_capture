#!/usr/bin/env bash
# Creates conda environments for the value_capture project.
#
# Usage (from repo root):
#   bash create_env/setup.sh [target]
#
# Targets:
#   psychopy       — experiment runtime (PsychoPy + exptools2)
#   apple_silicon  — analysis env for macOS arm64 (default for 'both')
#   linux          — analysis env for Linux / cluster
#   both           — psychopy + apple_silicon (default)

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

create_apple_silicon() {
    echo "=== Creating value_capture (Apple Silicon / macOS arm64) ==="
    conda env create -f "$SCRIPT_DIR/environment_apple_silicon.yml" --yes

    echo ""
    echo "value_capture ready."
    echo "  conda activate value_capture"
    echo "  jupyter lab"
}

create_linux() {
    echo "=== Creating value_capture (Linux / cluster) ==="
    conda env create -f "$SCRIPT_DIR/environment_linux.yml" --yes

    echo ""
    echo "value_capture ready."
    echo "  conda activate value_capture"
    echo "  jupyter lab"
}

TARGET="${1:-both}"

case "$TARGET" in
    psychopy)      create_psychopy ;;
    apple_silicon) create_apple_silicon ;;
    linux)         create_linux ;;
    both)
        create_psychopy
        echo ""
        create_apple_silicon
        ;;
    *)
        echo "Unknown target '$TARGET'. Use: psychopy | apple_silicon | linux | hssm | both"
        exit 1
        ;;
esac

echo ""
echo "=== All done ==="
