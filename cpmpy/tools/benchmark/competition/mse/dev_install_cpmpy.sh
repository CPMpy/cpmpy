#!/usr/bin/env bash
set -euo pipefail

# Local development: install CPMpy + extras into a venv (not used in competition zips).
# Reuses install_solver.sh (same pip steps as bin/install_solver.sh in submissions).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/install_solver.sh"

# Defaults. Override here, with environment variables, or with CLI flags below.
CPMPY_SRC="${CPMPY_SRC:-$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$CPMPY_SRC/.venv-maxsat}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPMPY_EXTRAS="${CPMPY_EXTRAS:-highs,io.wcnf}"
SOLVER_REQUIREMENTS="${SOLVER_REQUIREMENTS:-}"
RECREATE_VENV="${RECREATE_VENV:-0}"
UPGRADE_PACKAGES="${UPGRADE_PACKAGES:-0}"

usage() {
    cat <<EOF
Usage: $0 [options]

Create a virtualenv and install CPMpy for local MaxSAT / MSE development.
For competition bundles, use submission_setup.sh (which copies install_solver.sh into ./bin/).

Options:
  --venv DIR          Virtualenv directory (default: $VENV_DIR)
  --python BIN        Python executable used by virtualenv (default: $PYTHON_BIN)
  --cpmpy-src DIR     Local CPMpy source checkout (default: $CPMPY_SRC)
  --extras LIST       CPMpy extras to install, comma-separated (default: $CPMPY_EXTRAS)
  --solver-requirements PATH
                      Extra pip requirements with exact pins (installed after editable cpmpy)
  --no-extras         Install CPMpy without extras
  --recreate          Remove the virtualenv before creating it
  --upgrade           Upgrade installed packages
  -h, --help          Show this help

Examples:
  $0
  $0 --venv .venv --extras highs
  $0 --solver-requirements ./solver_requirements.txt --upgrade
  CPMPY_EXTRAS=highs $0 --upgrade
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --cpmpy-src)
            CPMPY_SRC="$2"
            shift 2
            ;;
        --extras)
            CPMPY_EXTRAS="$2"
            shift 2
            ;;
        --solver-requirements)
            SOLVER_REQUIREMENTS="$2"
            shift 2
            ;;
        --no-extras)
            CPMPY_EXTRAS=""
            shift
            ;;
        --recreate)
            RECREATE_VENV=1
            shift
            ;;
        --upgrade)
            UPGRADE_PACKAGES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

CPMPY_SRC="$(cd -- "$CPMPY_SRC" && pwd)"
VENV_DIR="$(mkdir -p -- "$(dirname -- "$VENV_DIR")" && cd -- "$(dirname -- "$VENV_DIR")" && pwd)/$(basename -- "$VENV_DIR")"

if [[ ! -f "$CPMPY_SRC/setup.py" ]]; then
    echo "CPMPY_SRC does not look like a CPMpy source checkout: $CPMPY_SRC" >&2
    exit 1
fi

if [[ -n "$SOLVER_REQUIREMENTS" ]]; then
    if [[ ! -f "$SOLVER_REQUIREMENTS" ]]; then
        echo "Solver requirements file not found: $SOLVER_REQUIREMENTS" >&2
        exit 1
    fi
    SOLVER_REQUIREMENTS="$(cd -- "$(dirname -- "$SOLVER_REQUIREMENTS")" && pwd)/$(basename -- "$SOLVER_REQUIREMENTS")"
    export SOLVER_REQUIREMENTS
fi

if [[ "$RECREATE_VENV" == "1" && -d "$VENV_DIR" ]]; then
    rm -rf -- "$VENV_DIR"
fi

if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    MSE_SUBMISSION_REQUIREMENTS="$(cd -- "$SCRIPT_DIR" && pwd)/requirements.txt"
    export MSE_SUBMISSION_REQUIREMENTS
fi
mse_install_cpmpy "$CPMPY_SRC" "$VENV_DIR" "$CPMPY_EXTRAS" "$PYTHON_BIN" "$UPGRADE_PACKAGES"
unset MSE_SUBMISSION_REQUIREMENTS

cat <<EOF

MaxSAT CPMpy environment ready.
  venv:      $VENV_DIR
  cpmpy:     $CPMPY_SRC
  extras:    ${CPMPY_EXTRAS:-none}
  pins:      ${SOLVER_REQUIREMENTS:-none}

Activate it with:
  source "$VENV_DIR/bin/activate"
EOF
