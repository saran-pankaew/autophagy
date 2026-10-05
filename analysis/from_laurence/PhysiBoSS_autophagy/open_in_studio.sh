#!/bin/bash
# Open this project in PhysiCell Studio.
#
# Usage:
#   ./open_in_studio.sh                                    # amino-acid starvation (default)
#   ./open_in_studio.sh 1_growth_factor_stimulation.xml
#   ./open_in_studio.sh 3_ER_glucose_starvation.xml
#   ./open_in_studio.sh 4_oxidative_stress.xml
#
# Run from anywhere; paths below are resolved relative to this script's location.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_DIR="/Users/laurence/Desktop/PhysiCell-Studio"
CONFIG_NAME="${1:-PhysiCell_settings.xml}"
CONFIG_PATH="$PROJECT_DIR/config/$CONFIG_NAME"
EXEC_PATH="$PROJECT_DIR/autophagy_population"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    echo "Available configs in $PROJECT_DIR/config:" >&2
    ls "$PROJECT_DIR/config"/*.xml >&2
    exit 1
fi

if [ ! -x "$EXEC_PATH" ]; then
    echo "Executable not found or not executable: $EXEC_PATH" >&2
    echo "Build it first, e.g.:" >&2
    echo "  cd \"$PROJECT_DIR\" && PHYSICELL_CPP=/usr/local/Cellar/gcc/16.1.0/bin/g++-16 make -j4" >&2
    exit 1
fi

# Pick whichever conda env has PyQt5 (Studio's dependency); prefer physicell-studio.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/opt/miniconda3")"
PYTHON_BIN=""
for env_name in physicell-studio studio; do
    candidate="$CONDA_BASE/envs/$env_name/bin/python"
    if [ -x "$candidate" ]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Could not find a 'physicell-studio' or 'studio' conda environment under $CONDA_BASE/envs." >&2
    echo "Activate the right environment yourself, then run:" >&2
    echo "  cd \"$STUDIO_DIR\" && python bin/studio.py -c \"$CONFIG_PATH\" -e \"$EXEC_PATH\"" >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
echo "Config:       $CONFIG_PATH"
echo "Executable:   $EXEC_PATH"

cd "$STUDIO_DIR"
"$PYTHON_BIN" bin/studio.py -c "$CONFIG_PATH" -e "$EXEC_PATH"
