#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/submission_setup.sh" --doc "$SCRIPT_DIR/PB26_position_paper.pdf" --solver-requirements "$SCRIPT_DIR/requirements.txt" --solver ortools --intermediate
"$SCRIPT_DIR/submission_setup.sh" --doc "$SCRIPT_DIR/PB26_position_paper.pdf" --solver-requirements "$SCRIPT_DIR/requirements.txt" --solver highs --intermediate