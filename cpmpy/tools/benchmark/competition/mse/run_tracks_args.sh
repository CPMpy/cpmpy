#!/usr/bin/env bash
# Per-track argv for run_benchmark.py (everything except the instance path and the
# run_benchmark --solver flag: pass the latter via submission_setup.sh --solver).
# submission_setup.sh sources this file and calls mse_fixed_benchmark_args TRACK
# for each track; tokens are NUL-delimited so spaces inside an argument are safe.
#
# Edit the case arms below, then re-run submission_setup.sh so bin/run_<TRACK>
# wrappers embed the new flags.

mse_fixed_benchmark_args() {
    case "$1" in
        EXACT-W)
            printf '%s\0' --runner mse --verbose
            ;;
        EXACT-UW)
            printf '%s\0' --runner mse --verbose
            ;;
        ANYTIME-W)
            printf '%s\0' --runner mse --verbose
            ;;
        ANYTIME-UW)
            printf '%s\0' --runner mse --verbose
            ;;
        *)
            echo "mse_fixed_benchmark_args: no args defined for track '$1'. Edit run_tracks_args.sh." >&2
            return 1
            ;;
    esac
}
