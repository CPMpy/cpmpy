#!/usr/bin/env python3
"""
Insert an empty Unreleased section at the top of changelog.md.

Run this once when preparing a release, after the current Unreleased section
has been renamed to the version being published. The release commit then
already contains an empty section for the next pull request.

Usage:
    python dev/new_changelog_entry.py [changelog.md]
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_VERSION_HEADING = re.compile(r"^##\s+v?(\d+\.\d+\.\d+)\b", re.MULTILINE)

UNRELEASED_TEMPLATE = """## Unreleased

<!-- One bullet per pull request, with a link to the pull request. See dev/README.md. -->

### Added

### Breaking changes

### Changed

### Fixed

### Removed

**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v{previous}...vX.Y.Z

"""


def add_unreleased_entry(changelog: str) -> str:
    """Return changelog text with an empty Unreleased section inserted at the top."""
    if "## Unreleased" in changelog.splitlines():
        raise ValueError(
            "changelog already has an Unreleased section; "
            "rename it to the version being released before running this script"
        )

    lines = changelog.splitlines(keepends=True)
    if not lines or not lines[0].startswith("# "):
        raise ValueError("changelog must start with a level-1 heading")

    rest = "".join(lines[1:]).lstrip("\n")
    previous = "<old.version>"
    match = _VERSION_HEADING.search(rest)
    if match:
        previous = match.group(1)
    return lines[0] + "\n" + UNRELEASED_TEMPLATE.format(previous=previous) + rest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1 or any(arg in ("-h", "--help") for arg in args):
        print("Usage: python dev/new_changelog_entry.py [changelog.md]")
        return 0 if args and args[0] in ("-h", "--help") else 1

    changelog_path = Path(args[0]) if args else REPO_ROOT / "changelog.md"
    try:
        updated = add_unreleased_entry(changelog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    changelog_path.write_text(updated, encoding="utf-8")
    print(f"Added an empty Unreleased section to {changelog_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
