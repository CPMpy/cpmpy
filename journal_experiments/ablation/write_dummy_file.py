#!/usr/bin/env python
"""
Write a placeholder result JSON when ``run_model.py`` did not complete.

Accepts the same arguments as ``run_model.py`` and writes a failure record to
the same output path that ``run_model.py`` would have used.
"""
from run_model import DEFAULT_FAILURE_ERROR, build_arg_parser, write_failure_from_args

if __name__ == "__main__":
    parser = build_arg_parser()
    parser.add_argument(
        "--error", default=DEFAULT_FAILURE_ERROR,
        help="error message to store in the JSON record")
    write_failure_from_args(parser.parse_args())
