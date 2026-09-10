"""
Command-line interface for CPMpy.

Usage:
    cpmpy <COMMAND>

Commands:
    version   Show the CPMpy library version and the versions of installed solver backends.
    skills    Install or update Agent Skills for the current project (requires cpmpy[skills]).
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys

from cpmpy import __version__
import cpmpy as cp

SKILLS_EXTRA_HINT = "cpmpy skills requires optional dependencies. Install with: pip install 'cpmpy[skills]'"

SKILLS_HELP = """\
usage: cpmpy skills [-h] [update] [library-skills args ...]

Install or refresh CPMpy Agent Skills in the current project via library-skills.

Commands:
  (none)              Interactive discover / install / repair
  update              Refresh project links to the currently installed CPMpy
                      (library-skills --yes). Does not upgrade the package;
                      run `pip install -U 'cpmpy[skills]'` first for new content.

Any other arguments are forwarded to library-skills, for example:
  cpmpy skills install -y --claude
  cpmpy skills list
  cpmpy skills update --claude --copy
"""


def command_version(args):
    print(f"CPMpy version: {__version__}")
    cp.SolverLookup().print_version()


def _library_skills_available() -> bool:
    return importlib.util.find_spec("library_skills") is not None


def run_library_skills(argv: list[str]) -> int:
    """
    Run the library-skills CLI with ``argv``. Returns the process exit code.
    """
    if not _library_skills_available():
        print(SKILLS_EXTRA_HINT, file=sys.stderr)
        return 1

    executable = shutil.which("library-skills")
    if executable:
        cmd = [executable, *argv]
    else:
        cmd = [sys.executable, "-m", "library_skills", *argv]
    completed = subprocess.run(cmd)
    return completed.returncode


def command_skills(argv: list[str]) -> int:
    """
    Native ``cpmpy skills`` command. ``update`` maps to ``library-skills --yes``.
    """
    if argv and argv[0] in ("-h", "--help"):
        print(SKILLS_HELP, end="")
        return 0
    if argv and argv[0] == "update":
        argv = ["--yes", *argv[1:]]
    return run_library_skills(argv)


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "skills":
        raise SystemExit(command_skills(argv[1:]))

    parser = argparse.ArgumentParser(description="CPMpy command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser(
        "version", help="Show version information on CPMpy and its solver backends"
    )
    version_parser.set_defaults(func=command_version)

    # Registered so `cpmpy --help` lists it; dispatch is handled above so
    # library-skills flags like --claude are not swallowed by argparse.
    subparsers.add_parser(
        "skills",
        help="Install or update Agent Skills for the current project (using library-skills)",
    )

    args = parser.parse_args(argv)
    args.func(args)
