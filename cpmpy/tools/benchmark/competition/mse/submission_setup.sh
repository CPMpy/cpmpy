#!/usr/bin/env bash
# Build a MaxSAT competition submission tree (and optionally a .zip).
#
# Python / cpmpy layout (cluster has Python + virtualenv only):
#   ./bin/build_solver.sh   run once manually: creates ./bin/.venv, pip-installs from ./code/
#   ./bin/build_solver.env  optional overrides (generated: CPMPY_EXTRAS)
#   ./bin/mse_submission_requirements.txt  copied from mse/requirements.txt (psutil, pins, …)
#   ./bin/solver_requirements.txt  optional extra pins (--solver-requirements)
#   ./bin/<solver_launcher>   compatibility copy; run_<TRACK> prefers ./code/<launcher> when available
#   ./bin/run_<TRACK>         EXACT*: <instance>; ANYTIME*: <instance> [time_limit_seconds]
#                             run_benchmark --solver from --solver plus run_tracks_args.sh flags.
#   Default staging dir       mse/mse_submission_staging_<solver> (override with --staging / STAGING_DIR).
#   ./code/<source_tree>      cpmpy checkout
#   ./doc/<description>.pdf
#
# Override defaults at the top, via environment variables, or with CLI flags.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CPMPY_SRC_DEFAULT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"

# --- defaults (edit or override with env / flags) ---
STAGING_DIR="${STAGING_DIR:-}"
BENCH_SOLVER="${BENCH_SOLVER:-}"
INTERMEDIATE="${INTERMEDIATE:-0}"
SOLVER_LAUNCHER="${SOLVER_LAUNCHER:-}"
SOLVER_BINARY="${SOLVER_BINARY:-}" # deprecated alias for SOLVER_LAUNCHER
SOLVER_NAME="${SOLVER_NAME:-}"
CPMPY_EXTRAS="${CPMPY_EXTRAS:-io.wcnf}"
CODE_SRC="${CODE_SRC:-$CPMPY_SRC_DEFAULT}"
DOC_PDF="${DOC_PDF:-}"
SOLVER_REQUIREMENTS_SRC="${SOLVER_REQUIREMENTS_SRC:-}"
ZIP_OUTPUT="${ZIP_OUTPUT:-}" # set with --zip PATH; if empty and --zip, default next to staging
TRACKS="${TRACKS:-EXACT-UW EXACT-W ANYTIME-W ANYTIME-UW}"
CLEAN_STAGING="${CLEAN_STAGING:-0}"
DO_ZIP="${DO_ZIP:-0}"
RUN_ARGS_SH="${RUN_ARGS_SH:-$SCRIPT_DIR/run_tracks_args.sh}"

usage() {
    cat <<EOF
Usage: $0 [options]

Create ./bin (build_solver.sh, launcher copy, run_<TRACK>), ./code, ./doc under a staging directory.
Optionally pack into a .zip (competition also allows .tar, .tar.gz, .tgz).

After unpack, run ./bin/build_solver.sh once before using run_<TRACK> or benchmarks.
run_EXACT-* expects one argument (<instance>); run_ANYTIME-* requires a second argument
interpreted as --time_limit seconds.

Required:
  --doc PATH             Path to solver description PDF

Optional:
  --launcher PATH        Launcher path. run_<TRACK> executes it from ./code/ when possible; ./bin copy is fallback
  --solver NAME          CPMpy solver name passed to run_benchmark.py as --solver NAME (recommended).
                         Default staging dir is mse/mse_submission_staging_<NAME> unless --staging is set.
  --bench-solver NAME    Deprecated alias for --solver
  --intermediate         Embed --intermediate in each generated run_<TRACK> wrapper

Common options:
  --staging DIR          Output directory (default: mse/mse_submission_staging_<solver>; requires
                         --solver or BENCH_SOLVER unless you set this flag or STAGING_DIR)
  --code-src DIR         Source tree to copy under ./code/ (default: cpmpy repo root)
  --solver-name NAME     Filename under bin/ for the launcher (default: basename of launcher script)
  --extras LIST          CPMpy pip extras for build_solver.env (default: $CPMPY_EXTRAS)
  --solver-requirements PATH
                         Copy to ./bin/solver_requirements.txt (pip -r after mse defaults; optional extras)
  --tracks LIST          Space- or comma-separated tracks (default: all four MSE tracks)
  --run-args PATH        Script defining mse_fixed_benchmark_args per track (default: mse/run_tracks_args.sh)
  --no-intermediate      Disable embedded --intermediate (useful with INTERMEDIATE=1 in env)
  --clean                Remove staging directory before building

Zip:
  --zip [PATH]           After building, create a zip archive.
                         If PATH is omitted, uses <staging>.zip next to staging.

Environment (same names as flags, uppercase): STAGING_DIR, SOLVER_LAUNCHER, DOC_PDF, BENCH_SOLVER, INTERMEDIATE,
  CODE_SRC, SOLVER_NAME, CPMPY_EXTRAS, SOLVER_REQUIREMENTS_SRC, TRACKS, RUN_ARGS_SH, CLEAN_STAGING, DO_ZIP=1, ZIP_OUTPUT
  (SOLVER_BINARY is a deprecated alias for SOLVER_LAUNCHER / --launcher.)

Examples:
  $0 --doc ./doc/solver.pdf --solver highs --solver-requirements ./requirements.txt --zip
  $0 --doc ./doc/solver.pdf --solver highs --solver-requirements ./requirements.txt --staging ./out/submit --zip
  $0 --doc ./doc/solver.pdf --solver highs --intermediate --solver-requirements ./requirements.txt --zip
  $0 --doc ./doc/solver.pdf --staging ./out/submit --zip
  $0 --doc ./doc/solver.pdf --solver pindakaas --solver-requirements ./solver_requirements.txt --zip
  $0 --doc ./doc/solver.pdf --solver highs --run-args ./my_run_tracks_args.sh --zip
  $0 --doc ./doc/solver.pdf --solver highs --launcher ./bin/custom_launcher.sh --zip
EOF
}

comma_to_space() {
    echo "$1" | tr ',' ' '
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --staging)
            STAGING_DIR="$2"
            shift 2
            ;;
        --launcher)
            SOLVER_LAUNCHER="$2"
            shift 2
            ;;
        --solver)
            BENCH_SOLVER="$2"
            shift 2
            ;;
        --bench-solver)
            echo "Warning: --bench-solver is deprecated; use --solver NAME." >&2
            BENCH_SOLVER="$2"
            shift 2
            ;;
        --intermediate)
            INTERMEDIATE=1
            shift
            ;;
        --solver-binary)
            SOLVER_LAUNCHER="$2"
            shift 2
            ;;
        --extras)
            CPMPY_EXTRAS="$2"
            shift 2
            ;;
        --solver-requirements)
            SOLVER_REQUIREMENTS_SRC="$2"
            shift 2
            ;;
        --solver-name)
            SOLVER_NAME="$2"
            shift 2
            ;;
        --code-src)
            CODE_SRC="$2"
            shift 2
            ;;
        --doc)
            DOC_PDF="$2"
            shift 2
            ;;
        --tracks)
            TRACKS="$(comma_to_space "$2")"
            shift 2
            ;;
        --run-args)
            RUN_ARGS_SH="$2"
            shift 2
            ;;
        --no-intermediate)
            INTERMEDIATE=0
            shift
            ;;
        --clean)
            CLEAN_STAGING=1
            shift
            ;;
        --zip)
            DO_ZIP=1
            if [[ $# -ge 2 && "$2" != --* ]]; then
                ZIP_OUTPUT="$2"
                shift 2
            else
                shift
            fi
            ;;
        --no-zip)
            DO_ZIP=0
            ZIP_OUTPUT=""
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

if [[ -z "$STAGING_DIR" ]]; then
    if [[ -z "${BENCH_SOLVER:-}" ]]; then
        echo "Error: default staging is named after the benchmark solver. Set --solver NAME (or BENCH_SOLVER)," >&2
        echo "  or pass --staging DIR / set STAGING_DIR explicitly." >&2
        exit 1
    fi
    _bench_safe="$(printf '%s' "$BENCH_SOLVER" | tr -c 'A-Za-z0-9._-' '_')"
    STAGING_DIR="$SCRIPT_DIR/mse_submission_staging_${_bench_safe}"
fi

DEFAULT_LAUNCH_REL="cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py"
SOLVER_LAUNCHER="${SOLVER_LAUNCHER:-$SOLVER_BINARY}"
if [[ -z "$SOLVER_LAUNCHER" ]]; then
    SOLVER_LAUNCHER="$CODE_SRC/$DEFAULT_LAUNCH_REL"
fi

if [[ ! -d "$CODE_SRC" ]]; then
    echo "Error: code source directory not found: $CODE_SRC" >&2
    exit 1
fi
if [[ ! -f "$SOLVER_LAUNCHER" ]]; then
    echo "Error: solver launcher not found: $SOLVER_LAUNCHER" >&2
    echo "  Use --launcher PATH, or ensure --code-src contains $DEFAULT_LAUNCH_REL" >&2
    exit 1
fi
if [[ -z "$DOC_PDF" ]]; then
    echo "Error: --doc is required (path to solver description PDF)." >&2
    exit 1
fi
if [[ ! -f "$DOC_PDF" ]]; then
    echo "Error: doc PDF not found: $DOC_PDF" >&2
    exit 1
fi
if [[ -n "$SOLVER_REQUIREMENTS_SRC" && ! -f "$SOLVER_REQUIREMENTS_SRC" ]]; then
    echo "Error: solver requirements file not found: $SOLVER_REQUIREMENTS_SRC" >&2
    exit 1
fi

if [[ ! -f "$RUN_ARGS_SH" ]]; then
    echo "Error: run track args script not found: $RUN_ARGS_SH" >&2
    echo "  Create mse/run_tracks_args.sh defining mse_fixed_benchmark_args(), or pass --run-args PATH." >&2
    exit 1
fi
RUN_ARGS_SH="$(cd -- "$(dirname -- "$RUN_ARGS_SH")" && pwd)/$(basename -- "$RUN_ARGS_SH")"
# shellcheck disable=SC1090
source "$RUN_ARGS_SH"
if ! declare -F mse_fixed_benchmark_args >/dev/null 2>&1; then
    echo "Error: $RUN_ARGS_SH must define mse_fixed_benchmark_args() { ... }" >&2
    exit 1
fi

if [[ -z "$SOLVER_NAME" ]]; then
    SOLVER_NAME="$(basename -- "$SOLVER_LAUNCHER")"
fi

STAGING_DIR="$(mkdir -p -- "$(dirname -- "$STAGING_DIR")" && cd -- "$(dirname -- "$STAGING_DIR")" && pwd)/$(basename -- "$STAGING_DIR")"
CODE_SRC="$(cd -- "$CODE_SRC" && pwd)"
SOLVER_LAUNCHER="$(cd -- "$(dirname -- "$SOLVER_LAUNCHER")" && pwd)/$(basename -- "$SOLVER_LAUNCHER")"
DOC_PDF="$(cd -- "$(dirname -- "$DOC_PDF")" && pwd)/$(basename -- "$DOC_PDF")"
USE_CODE_LAUNCHER=0
CODE_LAUNCHER_REL=""
if [[ "$SOLVER_LAUNCHER" == "$CODE_SRC/"* ]]; then
    USE_CODE_LAUNCHER=1
    CODE_LAUNCHER_REL="${SOLVER_LAUNCHER#"$CODE_SRC/"}"
fi

if [[ "$CLEAN_STAGING" == "1" && -d "$STAGING_DIR" ]]; then
    rm -rf -- "$STAGING_DIR"
fi

mkdir -p -- "$STAGING_DIR/bin" "$STAGING_DIR/code" "$STAGING_DIR/doc"

cp -a -- "$SCRIPT_DIR/build_solver.sh" "$STAGING_DIR/bin/build_solver.sh"
chmod +x -- "$STAGING_DIR/bin/build_solver.sh"

if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    cp -a -- "$SCRIPT_DIR/requirements.txt" "$STAGING_DIR/bin/mse_submission_requirements.txt"
fi
if [[ -n "$SOLVER_REQUIREMENTS_SRC" ]]; then
    cp -a -- "$SOLVER_REQUIREMENTS_SRC" "$STAGING_DIR/bin/solver_requirements.txt"
fi

# Optional overrides for build_solver.sh (sourced by build_solver.sh and by run_<TRACK>)
{
    echo "# Generated by submission_setup.sh; edit before zipping if needed."
    echo "# Optional: add bin/solver_requirements.txt for extra pinned packages (pip -r after mse_submission_requirements.txt)."
    echo '_MSE_BIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"'
    echo 'export VENV_DIR="$_MSE_BIN_DIR/.venv"'
    echo 'unset _MSE_BIN_DIR'
    if [[ -n "$BENCH_SOLVER" ]]; then
        printf '# run_benchmark.py --solver (embedded in bin/run_*): %q\n' "$BENCH_SOLVER"
    fi
    printf 'export CPMPY_EXTRAS=%q\n' "$CPMPY_EXTRAS"
} >"$STAGING_DIR/bin/build_solver.env"

cp -a -- "$SOLVER_LAUNCHER" "$STAGING_DIR/bin/$SOLVER_NAME"
chmod +x -- "$STAGING_DIR/bin/$SOLVER_NAME"

cp -a -- "$RUN_ARGS_SH" "$STAGING_DIR/bin/run_tracks_args.sh"

_mse_rb_solver_embed=""
if [[ -n "$BENCH_SOLVER" ]]; then
    _mse_rb_solver_embed+=$(printf '%q' --solver)" "
    _mse_rb_solver_embed+=$(printf '%q' "$BENCH_SOLVER")" "
fi
_mse_rb_intermediate_embed=""
if [[ "$INTERMEDIATE" == "1" ]]; then
    _mse_rb_intermediate_embed+=$(printf '%q' --intermediate)" "
fi

for track in $TRACKS; do
    run_script="$STAGING_DIR/bin/run_${track}"
    _mse_track_is_anytime=0
    if [[ "$track" == ANYTIME-* ]]; then
        _mse_track_is_anytime=1
    fi
    if ! mapfile -d '' -t _mse_rb_args < <(mse_fixed_benchmark_args "$track"); then
        echo "Error: mse_fixed_benchmark_args '$track' failed (check $RUN_ARGS_SH)." >&2
        exit 1
    fi
    _mse_rb_embed=""
    for _a in "${_mse_rb_args[@]}"; do
        _mse_rb_embed+=$(printf '%q' "$_a")" "
    done
    cat >"$run_script" <<EOF
#!/bin/bash
set -euo pipefail
DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
ENV="\$DIR/build_solver.env"
if [[ -f "\$ENV" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "\$ENV"
    set +a
fi
: "\${VENV_DIR:=\$DIR/.venv}"
if [[ ! -x "\$VENV_DIR/bin/python" ]]; then
    echo "Missing virtualenv at \$VENV_DIR. Run once from the submission root: bin/build_solver.sh" >&2
    exit 1
fi
TIME_LIMIT_ARGS=()
if [[ "$_mse_track_is_anytime" == "1" ]]; then
    if [[ \$# -ne 2 ]]; then
        echo "Usage: \$0 <instance_file> <time_limit_seconds>" >&2
        exit 2
    fi
    INSTANCE="\$1"
    TIME_LIMIT_ARGS=(--time_limit "\$2")
else
    if [[ \$# -ne 1 ]]; then
        echo "Usage: \$0 <instance_file>" >&2
        exit 2
    fi
    INSTANCE="\$1"
fi
PY="\$VENV_DIR/bin/python"
if [[ "$USE_CODE_LAUNCHER" == "1" ]]; then
    CODE_LAUNCHER="\$DIR/../code/$CODE_LAUNCHER_REL"
else
    CODE_LAUNCHER=""
fi
if [[ -f "\$CODE_LAUNCHER" ]]; then
    LAUNCHER="\$CODE_LAUNCHER"
else
    LAUNCHER="\$DIR/$SOLVER_NAME"
fi
case "\$LAUNCHER" in
    *.py)
        exec "\$PY" "\$LAUNCHER" "\$INSTANCE" "\${TIME_LIMIT_ARGS[@]}" $_mse_rb_solver_embed$_mse_rb_intermediate_embed$_mse_rb_embed
        ;;
    *)
        exec "\$LAUNCHER" "\$INSTANCE" "\${TIME_LIMIT_ARGS[@]}" $_mse_rb_solver_embed$_mse_rb_intermediate_embed$_mse_rb_embed
        ;;
esac
EOF
    chmod +x -- "$run_script"
done

# Copy source tree; exclude heavy / local-only paths common in dev checkouts.
RSYNC_EXCLUDES=(
    --exclude=.git
    --exclude=.venv
    --exclude=venv
    --exclude=__pycache__
    --exclude='*.pyc'
    --exclude=.pytest_cache
    --exclude=.mypy_cache
    --exclude=.ruff_cache
    --exclude=dist
    --exclude=build
    --exclude='*.egg-info'
    --exclude='mse_submission_staging_*'
    --exclude=docs
    --exclude=examples
)
# Avoid copying the submission staging tree into ./code/ when it lives under CODE_SRC.
if command -v realpath >/dev/null 2>&1; then
    _code_r="$(realpath -- "$CODE_SRC")"
    _stage_r="$(realpath -- "$STAGING_DIR")"
    if [[ "$_stage_r" == "$_code_r"/* ]]; then
        _rel="${_stage_r#"${_code_r}/"}"
        RSYNC_EXCLUDES+=(--exclude="/$_rel")
    fi
fi
if command -v rsync >/dev/null 2>&1; then
    rsync -a "${RSYNC_EXCLUDES[@]}" "$CODE_SRC/" "$STAGING_DIR/code/"
else
    echo "Warning: rsync not found; using cp -a (no excludes). Install rsync for smaller exports." >&2
    cp -a -- "$CODE_SRC/." "$STAGING_DIR/code/"
fi

cp -a -- "$DOC_PDF" "$STAGING_DIR/doc/$(basename -- "$DOC_PDF")"

echo "Submission tree ready at: $STAGING_DIR"
if [[ -n "$BENCH_SOLVER" ]]; then
    echo "  run_benchmark.py --solver $BENCH_SOLVER (embedded in each bin/run_<TRACK>)"
fi
if [[ "$INTERMEDIATE" == "1" ]]; then
    echo "  run_benchmark.py --intermediate (embedded in each bin/run_<TRACK>)"
fi
echo "  bin/build_solver.sh   (run once before experiments: cd into staging or submission root, then ./bin/build_solver.sh)"
echo "  bin/build_solver.env"
if [[ -f "$STAGING_DIR/bin/mse_submission_requirements.txt" ]]; then
    echo "  bin/mse_submission_requirements.txt"
fi
if [[ -f "$STAGING_DIR/bin/solver_requirements.txt" ]]; then
    echo "  bin/solver_requirements.txt"
fi
echo "  bin/$SOLVER_NAME"
echo "  bin/run_tracks_args.sh  (snapshot of per-track run_benchmark argv used when staging)"
for track in $TRACKS; do
    echo "  bin/run_${track}"
done
echo "  code/  (from $CODE_SRC)"
echo "  doc/$(basename -- "$DOC_PDF")"

if [[ "$DO_ZIP" == "1" ]]; then
    if ! command -v zip >/dev/null 2>&1; then
        echo "Error: --zip requires the 'zip' command (install zip)." >&2
        exit 1
    fi
    if [[ -z "$ZIP_OUTPUT" ]]; then
        ZIP_OUTPUT="$(dirname -- "$STAGING_DIR")/$(basename -- "$STAGING_DIR").zip"
    else
        ZIP_OUTPUT="$(mkdir -p -- "$(dirname -- "$ZIP_OUTPUT")" && cd -- "$(dirname -- "$ZIP_OUTPUT")" && pwd)/$(basename -- "$ZIP_OUTPUT")"
    fi
    if [[ "${ZIP_OUTPUT##*.}" != "zip" ]]; then
        echo "Error: --zip output must end with .zip (got: $ZIP_OUTPUT)" >&2
        exit 1
    fi
    rm -f -- "$ZIP_OUTPUT"
    parent="$(dirname -- "$STAGING_DIR")"
    base="$(basename -- "$STAGING_DIR")"
    (cd -- "$parent" && zip -rq "$ZIP_OUTPUT" "$base")
    echo "Wrote archive: $ZIP_OUTPUT"
fi
