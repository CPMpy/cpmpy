"""
A small CLI tool for gerenating release notes for GitHub.

To create a new CPMpy release, one must call (!! do look at the wiki for more instructions !!):

```console
gh release create vX.Y.Z -F <file_with_release_notes>
```

The entire contents of <file_with_release_notes> will then be used as the markdown description of that release.
Simply passing `changelog.md` will add the entire changelog to the release notes of the newest release.
This script instead generates a temporary file (by default `release-notes.md`) with only the contents of the selected release.

Use this tool by calling
```console
python tools/extract_release_notes.py <VERSION> [changelog.md] [output.md]
```
For example:
```console
python tools/extract_release_notes.py 0.9.25
gh release create v0.9.25 -F release-notes.md
```
!! Again don't just copy paste the above when doing an actual release. Follow the steps described in the wiki.
"""


import sys
import re
from pathlib import Path

def extract_release_notes(version: str, changelog_path: str = "changelog.md") -> str:
    """
    Looks for the provided "version"'s section in the changelog and extracts its contents.
    """
    # Load changelog
    changelog = Path(changelog_path).read_text(encoding="utf-8")

    # Find start of correct section 
    version_heading_re = re.compile(rf"^##\s*\[?{re.escape(version)}\]?\s*$", re.MULTILINE)
    matches = list(version_heading_re.finditer(changelog))
    if not matches:
        raise ValueError(f"Version {version} not found in {changelog_path}")
    start = matches[0].end()
    rest = changelog[start:]

    # Find the start of the next version section
    next_section_re = re.compile(r"^##\s*\[?v?\d+\.\d+\.\d+\]?\s*$", re.MULTILINE)
    next_match = next_section_re.search(rest)
    end = next_match.start() if next_match else len(rest)

    # Slice section
    section = rest[:end].strip()

    # Remove comment lines: <!-- comment -->
    section = re.sub(r"^\s*<!--.*?-->\s*$", "", section, flags=re.MULTILINE)

    # Promote all markdown headers by one level: ### → ##, #### → ###, etc.
    promoted_section = re.sub(r"^(#+)", lambda m: "#" * max(2, len(m.group(1)) - 1), section, flags=re.MULTILINE)

    return promoted_section.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_release_notes.py vX.Y.Z [changelog.md] [output.md]")
        sys.exit(1)

    version = sys.argv[1]
    changelog_file = sys.argv[2] if len(sys.argv) > 2 else "changelog.md"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "release-notes.md"

    try:
        # Extract from changelog
        notes = extract_release_notes(version, changelog_file)

        # Format the release notes
        release_notes = "# Release notes\n\n" + notes

        # Write to file
        Path(output_file).write_text(release_notes, encoding="utf-8")
        print(f"Release notes for {version} written to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    