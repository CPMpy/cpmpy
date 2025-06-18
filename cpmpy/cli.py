import argparse
from cpmpy import __version__
import cpmpy as cp

def command_status(args):
    cp.SolverLookup().print_status()

def command_version(args):
    print(f"CPMpy version: {__version__}")

def main():
    parser = argparse.ArgumentParser(description="CPMpy command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # cpmpy status
    status_parser = subparsers.add_parser("status", help="List available solvers")
    status_parser.set_defaults(func=command_status)

    # cpmpy version
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=command_version)

    args = parser.parse_args()
    args.func(args)
