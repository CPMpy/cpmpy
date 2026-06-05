#!/usr/bin/env bash
# MaxSAT submission: lives in ./bin/ next to run_<TRACK> scripts.
# Compute node has Python 3.13.x and virtualenv only. Run this once after unpack
# (before experiments or before any run_<TRACK>): it creates ./bin/.venv by default
# and pip-installs from ./code/ (cpmpy checkout). run_<TRACK> invokes the solver launcher
# using ./bin/.venv/bin/python for Python launchers so imports match that venv.
#
# Venv creation tries, in order: the `virtualenv` CLI (competition), then
# PYTHON_BOOTSTRAP or /usr/bin/python3 with -m virtualenv / -m venv, then PYTHON.
# If your shell's `python3` is another venv without the virtualenv module, either
# put `virtualenv` on PATH or set PYTHON_BOOTSTRAP=/usr/bin/python3.
#
# After editable CPMpy: optional ./bin/mse_submission_requirements.txt (MSE defaults: psutil,
# solver pins, … from submission_setup), then optional ./bin/solver_requirements.txt or
# SOLVER_REQUIREMENTS (also build_solver.env). Later -r wins for duplicate packages.
#
# Can be sourced by dev_install_cpmpy.sh for local dev (same pip steps).
set -euo pipefail

# Defaults when run standalone from ./bin/ (optional overrides via build_solver.env)
: "${PYTHON:=python3}"
: "${CPMPY_EXTRAS:=highs,io.wcnf}"
: "${VENV_NAME:=.venv}"
: "${SOLVER_REQUIREMENTS:=}"
: "${MSE_SUBMISSION_REQUIREMENTS:=}"
# Interpreter used only to *run* "python -m virtualenv" / "python -m venv" when the
# standalone `virtualenv` command is missing. Prefer system Python so an active
# project .venv (without the virtualenv module) does not break bootstrap.
: "${PYTHON_BOOTSTRAP:=}"

# Create venv_dir with Python py_target as the environment's python (virtualenv -p).
mse_create_venv() {
    local venv_dir="$1"
    local py_target="$2"
    local boot="${PYTHON_BOOTSTRAP:-}"
    if [[ -z "$boot" && -x /usr/bin/python3 ]]; then
        boot=/usr/bin/python3
    fi
    [[ -n "$boot" ]] || boot="$py_target"

    if [[ -d "$venv_dir" ]]; then
        return 0
    fi

    if command -v virtualenv >/dev/null 2>&1; then
        virtualenv -p "$py_target" "$venv_dir"
        return
    fi
    if "$boot" -m virtualenv "$venv_dir" 2>/dev/null; then
        return
    fi
    if "$py_target" -m virtualenv "$venv_dir" 2>/dev/null; then
        return
    fi
    if "$boot" -m venv "$venv_dir" 2>/dev/null; then
        return
    fi
    if "$py_target" -m venv "$venv_dir" 2>/dev/null; then
        return
    fi
    echo "mse_create_venv: could not create $venv_dir." >&2
    echo "  Tried: virtualenv CLI, then ${boot} and ${py_target} with -m virtualenv / -m venv." >&2
    echo "  Tip: ensure \`virtualenv\` is on PATH (competition), or set PYTHON_BOOTSTRAP=/usr/bin/python3." >&2
    return 1
}

# Install CPMpy from a source tree into a virtualenv.
# Uses env MSE_SUBMISSION_REQUIREMENTS and SOLVER_REQUIREMENTS when set to file paths;
# otherwise optional ./bin/mse_submission_requirements.txt and ./bin/solver_requirements.txt.
# Args: code_dir venv_dir extras_csv python_bin upgrade_flag
mse_install_cpmpy() {
    local code_dir="$1"
    local venv_dir="$2"
    local extras="$3"
    local python_bin="${4:-python3}"
    local upgrade="${5:-0}"

    if [[ ! -f "$code_dir/setup.py" ]]; then
        echo "mse_install_cpmpy: not a CPMpy source tree (missing setup.py): $code_dir" >&2
        return 1
    fi

    if [[ ! -d "$venv_dir" ]]; then
        mse_create_venv "$venv_dir" "$python_bin"
    fi

    local pip_flags=()
    [[ "$upgrade" == "1" ]] && pip_flags+=(--upgrade)

    "$venv_dir/bin/python" -m pip install "${pip_flags[@]}" pip setuptools wheel

    "$venv_dir/bin/python" -m pip install "${pip_flags[@]}" -r "$code_dir/requirements.txt"

    if [[ -n "$extras" ]]; then
        "$venv_dir/bin/python" -m pip install "${pip_flags[@]}" -e "$code_dir[$extras]"
    else
        "$venv_dir/bin/python" -m pip install "${pip_flags[@]}" -e "$code_dir"
    fi

    # MSE submission defaults (psutil, pinned solvers, …) then optional extra pins.
    local mse_req="${MSE_SUBMISSION_REQUIREMENTS:-}"
    if [[ -n "$mse_req" && -f "$mse_req" ]]; then
        "$venv_dir/bin/python" -m pip install "${pip_flags[@]}" -r "$mse_req"
    fi
    local pins="${SOLVER_REQUIREMENTS:-}"
    if [[ -n "$pins" && -f "$pins" ]]; then
        "$venv_dir/bin/python" -m pip install "${pip_flags[@]}" -r "$pins"
    fi
}

# Standalone: submission layout ./bin/this_script, ./code/ (cpmpy checkout)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    BIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd -- "$BIN_DIR/.." && pwd)"
    CODE_DIR="$ROOT_DIR/code"

    ENV_FILE="$BIN_DIR/build_solver.env"
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    fi
    : "${VENV_DIR:=$BIN_DIR/$VENV_NAME}"

    if [[ -z "${MSE_SUBMISSION_REQUIREMENTS:-}" && -f "$BIN_DIR/mse_submission_requirements.txt" ]]; then
        MSE_SUBMISSION_REQUIREMENTS="$BIN_DIR/mse_submission_requirements.txt"
    fi
    if [[ -z "${SOLVER_REQUIREMENTS:-}" && -f "$BIN_DIR/solver_requirements.txt" ]]; then
        SOLVER_REQUIREMENTS="$BIN_DIR/solver_requirements.txt"
    fi

    mse_install_cpmpy "$CODE_DIR" "$VENV_DIR" "$CPMPY_EXTRAS" "$PYTHON" 0
fi
