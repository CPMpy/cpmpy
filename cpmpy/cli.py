"""
Command-line interface for CPMpy.

This module provides a simple CLI to interact with CPMpy, primarily to display
version information about CPMpy itself and the available solver backends.

Usage:
    cpmpy <COMMAND>

Commands:
    version   Show the CPMpy library version and the versions of installed solver backends.
"""

import argparse
from cpmpy import __version__
import cpmpy as cp


def command_version(args):
    print(f"CPMpy version: {__version__}")
    cp.SolverLookup().print_version()

def main():
    parser = argparse.ArgumentParser(description="CPMpy command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # cpmpy version
    version_parser = subparsers.add_parser("version", help="Show version information on CPMpy and its solver backends")
    version_parser.set_defaults(func=command_version)

    args = parser.parse_args()
    args.func(args)
