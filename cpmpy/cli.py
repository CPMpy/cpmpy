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
from pathlib import Path

from cpmpy import __version__
import cpmpy as cp

SKILLS_EXTRA_HINT = "cpmpy skills requires optional dependencies. Install with: pip install 'cpmpy[skills]'"
DOCS_EXTRA_HINT = (
    "cpmpy skills update --regenerate requires the docs extra. "
    "Install with: pip install 'cpmpy[docs]'"
)
REGENERATE_SOURCE_HINT = """\
--regenerate is only available when CPMpy is installed from source.

This install looks like a wheel (e.g. pip install cpmpy). There is no docs/
tree here to rebuild from; generated reference files already ship in the
package. To pick up a newer snapshot:

    pip install -U cpmpy
    cpmpy skills update

To regenerate from the Sphinx docs, clone the repo and install editable:

    git clone https://github.com/CPMpy/cpmpy.git
    cd cpmpy
    pip install -e '.[docs]'
    cpmpy skills update --regenerate
"""

SKILLS_HELP_PREAMBLE = """\
usage: cpmpy skills [-h] [update] [library-skills args ...]

Install or refresh CPMpy Agent Skills in the current project via library-skills.

Commands:
  (none)              Interactive discover / install / repair
  update              Refresh project links to the currently installed CPMpy
                      (library-skills --yes). Does not upgrade the package;
                      run `pip install -U 'cpmpy[skills]'` first for new content.
"""

SKILLS_HELP_REGENERATE = """\
  update --regenerate Rebuild generated reference files from the Sphinx docs,
                      recopy them into the package, then refresh project links.
                      Requires pip install 'cpmpy[docs]'.
"""

SKILLS_HELP_EPILOG = """
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


def _source_root() -> Path | None:
    """
    Repo root if this is an editable/source install, else None.
    """
    candidate = Path(cp.__file__).resolve().parent.parent
    if (candidate / "skills" / "generate_references.py").is_file():
        return candidate
    return None


def _can_regenerate() -> bool:
    """
    True when generated refs can be rebuilt (source checkout, not a wheel).
    """
    return _source_root() is not None


def skills_help() -> str:
    text = SKILLS_HELP_PREAMBLE
    if _can_regenerate():
        text += SKILLS_HELP_REGENERATE
    return text + SKILLS_HELP_EPILOG


def _docs_extra_available() -> bool:
    return (
        importlib.util.find_spec("sphinx") is not None
        and importlib.util.find_spec("sphinx_llm") is not None
    )


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


def _load_sync_module(root: Path):
    sync_path = root / "skills" / "sync_into_package.py"
    spec = importlib.util.spec_from_file_location("cpmpy_sync_into_package", sync_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load skill sync module from {sync_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regenerate_skill_references() -> int:
    """
    Rebuild ``*.generated.md`` from Sphinx docs and recopy skills into the package.
    """
    root = _source_root()
    if root is None:
        print(REGENERATE_SOURCE_HINT, file=sys.stderr)
        return 1
    if not _docs_extra_available():
        print(DOCS_EXTRA_HINT, file=sys.stderr)
        return 1

    generator = root / "skills" / "generate_references.py"
    completed = subprocess.run([sys.executable, str(generator), "--build"])
    if completed.returncode:
        return completed.returncode

    _load_sync_module(root).sync_skills(
        root / "skills",
        root / "cpmpy" / ".agents" / "skills",
    )
    return 0


def command_skills_update(argv: list[str]) -> int:
    """
    ``cpmpy skills update`` → ``library-skills --yes``, optionally regenerating refs.
    """
    parser = argparse.ArgumentParser(
        prog="cpmpy skills update",
        description=(
            "Refresh project links to the currently installed CPMpy. "
            "Does not run pip; upgrade the package first for newer skill content."
        ),
    )
    can_regenerate = _can_regenerate()
    if can_regenerate:
        parser.add_argument(
            "--regenerate",
            action="store_true",
            help=(
                "Rebuild generated skill reference files from the Sphinx docs "
                "(requires pip install 'cpmpy[docs]')."
            ),
        )
    args, rest = parser.parse_known_args(argv)
    if not can_regenerate and "--regenerate" in rest:
        print(REGENERATE_SOURCE_HINT, file=sys.stderr)
        return 1
    if can_regenerate and args.regenerate:
        code = regenerate_skill_references()
        if code:
            return code
    return run_library_skills(["--yes", *rest])


def command_skills(argv: list[str]) -> int:
    """
    Native ``cpmpy skills`` command. ``update`` maps to ``library-skills --yes``.
    """
    if argv and argv[0] in ("-h", "--help"):
        print(skills_help(), end="")
        return 0
    if argv and argv[0] == "update":
        return command_skills_update(argv[1:])
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
