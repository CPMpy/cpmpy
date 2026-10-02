#!/usr/bin/env python3
"""
Copy Agent Skills into the importable CPMpy package.

Agent skills live in this directory (``skills/<name>/``). ``library-skills``
only discovers ``.agents/skills/*/SKILL.md`` inside the installed package, so
this script copies those directories to ``cpmpy/.agents/skills/`` at
build/install time. The destination is generated (gitignored); do not edit it
by hand.

Called from ``setup.py``'s ``build_py`` (wheels and editable installs). You can
also run it directly:

    python skills/sync_into_package.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent
DEFAULT_SRC = SKILLS_DIR
DEFAULT_DEST = REPO_ROOT / "cpmpy" / ".agents" / "skills"


def iter_skill_dirs(src_dir: Path) -> list[Path]:
    """Return skill directories under ``src_dir`` (each contains ``SKILL.md``)."""
    if not src_dir.is_dir():
        return []
    skills = []
    for skill in sorted(src_dir.iterdir()):
        if skill.is_dir() and (skill / "SKILL.md").is_file():
            skills.append(skill)
    return skills


def iter_generated_ref_files(src_dir: Path) -> list[Path]:
    """Return existing ``references/*.generated.md`` files under skill directories."""
    found: list[Path] = []
    if not src_dir.is_dir():
        return found
    for path in sorted(src_dir.rglob("*.generated.md")):
        if path.is_file() and path.parent.name == "references":
            found.append(path)
    return found


def clean_generated_refs(src_dir: Path) -> list[Path]:
    """Delete generated skill reference files under ``src_dir``. Returns removed paths."""
    removed: list[Path] = []
    for path in iter_generated_ref_files(src_dir):
        path.unlink()
        removed.append(path)
    return removed


def sync_skills(
    src: Path | str | None = None,
    dest: Path | str | None = None,
) -> list[str]:
    """
    Copy each ``skills/<name>/`` directory that contains a ``SKILL.md``.

    Wipes ``dest`` first so removed skills do not linger. Returns the names of
    skill directories that were copied. Generated ``*.generated.md`` files are
    included when present (they are committed under ``skills/``).
    """
    src_dir = Path(src) if src is not None else DEFAULT_SRC
    dest_dir = Path(dest) if dest is not None else DEFAULT_DEST

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for skill in iter_skill_dirs(src_dir):
        shutil.copytree(skill, dest_dir / skill.name)
        copied.append(skill.name)

    return copied


if __name__ == "__main__":
    names = sync_skills()
    dest = DEFAULT_DEST
    print(f"Copied {len(names)} skill(s) to {dest}: {', '.join(names) or '(none)'}")
