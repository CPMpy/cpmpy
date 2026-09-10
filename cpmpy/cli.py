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

CONTINUE_WITHOUT_GENERATED_PROMPT = "Continue without them? [y/N] "
NON_TTY_MISSING_REFS_HINT = (
    "Not a TTY; aborting. Generate the files with "
    "`cpmpy skills update --regenerate` or rerun from a terminal."
)

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


def _sync_helper_path(root: Path | None = None) -> Path | None:
    candidates = []
    if root is not None:
        candidates.append(root / "skills" / "sync_into_package.py")
    candidates.append(Path(__file__).resolve().parent.parent / "skills" / "sync_into_package.py")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_sync_module(root: Path | None = None):
    sync_path = _sync_helper_path(root)
    if sync_path is None:
        raise FileNotFoundError("skills/sync_into_package.py")
    spec = importlib.util.spec_from_file_location("cpmpy_sync_into_package", sync_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def confirm_continue_with_missing_generated_refs() -> bool:
    """Ask whether to continue after missing generated refs. Default is no."""
    if not _stdin_is_tty():
        print(NON_TTY_MISSING_REFS_HINT, file=sys.stderr)
        return False
    try:
        print(CONTINUE_WITHOUT_GENERATED_PROMPT, end="", file=sys.stderr, flush=True)
        answer = input()
    except EOFError:
        print(file=sys.stderr)
        return False
    return answer.strip().lower() in {"y", "yes"}


def warn_if_generated_refs_missing(*, skip: bool = False) -> bool:
    """
    On an editable/source install, warn if gitignored generated refs are absent.

    Returns True if the command should proceed. When files are missing, asks
    whether to continue (default no).
    """
    if skip:
        return True
    root = _source_root()
    if root is None:
        return True
    if _sync_helper_path(root) is None:
        return True
    missing = _load_sync_module(root).warn_if_generated_refs_missing(root / "skills")
    if not missing:
        return True
    return confirm_continue_with_missing_generated_refs()


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
    help_requested = "-h" in argv or "--help" in argv
    skip_missing_prompt = bool(
        argv and argv[0] == "update" and "--regenerate" in argv[1:]
    )
    if not help_requested:
        if not warn_if_generated_refs_missing(skip=skip_missing_prompt):
            return 1
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
