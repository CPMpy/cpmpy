#!/usr/bin/env python3
"""
Generate skill reference files from this repo's Sphinx docs.

A skill's `references/*.generated.md` files are never hand-edited, they are
"generated" from the agent-friendly Markdown mirror that the `sphinx_llm` Sphinx
extension writes alongside the HTML build (one `<page>.html.md` per doc page
in `docs/_build/html/`, enabled in `docs/conf.py`). This keeps a skill's 
reference material in sync with the actual documentation.

Setup (only needed to run this script, not to use a skill):
    pip install -e ".[docs]"     # from the repo root; installs sphinx_llm
    python skills/generate_references.py --build
                                  # builds the docs, then generates

Or build the docs yourself first (`cd docs && make html`) and just run:
    python skills/generate_references.py

Usage:
    python skills/generate_references.py             # (re)generate every file below
    python skills/generate_references.py --build     # build the docs first
    python skills/generate_references.py --check     # verify generated files are
                                                     # up to date; exit 1 if not
                                                     # (use in CI after docs change)

To add a new generated reference file, add an entry to SOURCES below.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DOCS_BUILD_HTML = DOCS_DIR / "_build" / "html"

# Each entry: (sphinx_llm-generated markdown file, relative to
#              docs/_build/html/,
#              ordered list of heading texts to extract from it, or None to
#              copy the whole page,
#              output file relative to repo root)
SOURCES = [
    (
        "summary.html.md",
        None,
        "skills/cpmpy-model/references/api-cheatsheet.generated.md",
    ),
]

GENERATED_HEADER = """<!--
GENERATED FILE — do not edit by hand.
Source: docs/_build/html/{source}{sections_note}
(built by the sphinx_llm Sphinx extension, see docs/conf.py)
Regenerate: python skills/generate_references.py --build
See skills/README.md#keeping-reference-files-in-sync
-->

"""

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (line_index, level, text) for every Markdown heading line."""
    marks = []
    in_code_block = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        m = _HEADING_RE.match(line)
        if m:
            marks.append((i, len(m.group(1)), m.group(2)))
    return marks


def extract_sections(text: str, headings: list[str]) -> str:
    """Extract each named section (heading + its body + any subsections) from
    a Markdown document, in the order given by `headings`, not the order they
    appear in the source."""
    lines = text.splitlines(keepends=True)
    marks = _find_headings(lines)
    by_text: dict[str, list[int]] = {}
    for idx, (_line_i, _level, heading_text) in enumerate(marks):
        by_text.setdefault(heading_text, []).append(idx)

    chunks = []
    for wanted in headings:
        matches = by_text.get(wanted)
        if not matches:
            raise ValueError(
                f"heading {wanted!r} not found (looked for an exact Markdown "
                f"heading match)"
            )
        if len(matches) > 1:
            raise ValueError(f"heading {wanted!r} is ambiguous ({len(matches)} matches)")
        mark_idx = matches[0]
        start_line, level, _ = marks[mark_idx]
        end_line = len(lines)
        for line_i, other_level, _ in marks[mark_idx + 1 :]:
            if other_level <= level:
                end_line = line_i
                break
        chunk = "".join(lines[start_line:end_line]).rstrip("\n")
        chunks.append(chunk)
    return "\n\n".join(chunks) + "\n"


def build_docs() -> None:
    # Invoke sphinx-build as a module of *this* interpreter (sys.executable),
    # not via `make`/PATH — that can silently resolve to a different Python
    # environment's sphinx-build, one without sphinx_llm installed.
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(DOCS_DIR), str(DOCS_BUILD_HTML)],
        check=True,
    )


# sphinx_llm's markdown links other doc pages relative to docs/_build/html/
# (e.g. `modeling.html.md`, `api/expressions/globalconstraints.html.md#...`).
# Once copied into a skill's references/ dir those links resolve to the wrong
# place, so rewrite them relative to the generated file's own location instead.
_MD_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def _rebase_links(text: str, output_path: Path) -> str:
    prefix = os.path.relpath(DOCS_BUILD_HTML, output_path.parent)
    if prefix == ".":
        return text

    def repl(match: re.Match) -> str:
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#", "/")):
            return match.group(0)
        return f"]({prefix}/{target})"

    return _MD_LINK_RE.sub(repl, text)


def build(source: str, headings: list[str] | None, output: str) -> str:
    source_path = DOCS_BUILD_HTML / source
    if not source_path.exists():
        raise FileNotFoundError(
            f"{source_path.relative_to(REPO_ROOT)} not found. Build the docs first: "
            f"`pip install -e '.[docs]'` then `python skills/generate_references.py "
            f"--build` (or `cd docs && make html` yourself)."
        )
    text = source_path.read_text()
    body = text if headings is None else extract_sections(text, headings)
    body = _rebase_links(body, REPO_ROOT / output)

    sections_note = f" (sections: {', '.join(headings)})" if headings else ""
    header = GENERATED_HEADER.format(source=source, sections_note=sections_note)
    return header + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", action="store_true", help="build the Sphinx docs first (runs `make html`)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="don't write files; exit 1 if any generated file is stale/missing",
    )
    args = parser.parse_args()

    if args.build:
        build_docs()

    stale = []
    for source, headings, output in SOURCES:
        content = build(source, headings, output)
        output_path = REPO_ROOT / output
        if args.check:
            current = output_path.read_text() if output_path.exists() else None
            if current != content:
                stale.append(output)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            print(f"wrote {output}")

    if args.check and stale:
        print("stale or missing generated reference file(s):", file=sys.stderr)
        for output in stale:
            print(f"  {output}", file=sys.stderr)
        print("run: python skills/generate_references.py --build", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
