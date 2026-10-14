# CPMpy development

This directory contains a collection of scripts and documentation used during the development of CPMpy.



| File | Description |
| - | - |
| new_changelog_entry.py | Inserts an empty `Unreleased` section at the top of `changelog.md`. Run once before each release, after the current section has its version number. |
| extract_release_notes.py | Takes as input the changelog.md file and extracts + formats a specified version to serve as release notes on GitHub. |

---

## Changelog

Record changes in `changelog.md` as they land. Each pull request adds its own entry under `## Unreleased`, so the section is already filled in when a release is prepared.

### In every pull request

Add one bullet to the `## Unreleased` section, under the heading that fits the change:

1. **Added** — new features: solvers, globals, or anything else that was not in the interface before.
2. **Breaking changes** — changes that require users to update their code.
3. **Changed** — changes to existing behaviour that stay compatible. Keep these few; the external interface should stay stable. An example is changing which exception is raised in a specific situation.
4. **Fixed** — bug fixes.
5. **Removed** — removals, including dropped support such as an older Python version.

Write a short description, usually the pull request title, and a link to the pull request. Add the link once the pull request number is known. Emphasise notable additions, such as a new solver, with bold:

```
* **New solver**: Rc2 MaxSAT solver [#729](https://github.com/CPMpy/cpmpy/pull/729)
```

Order bullets by importance, impact, and novelty. New solvers at the top, small fixes lower down. Group related changes together.

A custom section is fine when a set of changes does not fit the headings above. For example:

```
### Internal improvements
... a collection of refactoring changes to the transformation waterfall
```

If `## Unreleased` is missing, create it before adding the bullet:

```
python dev/new_changelog_entry.py
```

### Publishing a release

The `## Unreleased` section should already list the changes. Before publishing, turn it into the release entry and open the next section in the same commit:

- Replace the `Unreleased` heading with the version number (`## X.Y.Z`).
- Replace `vX.Y.Z` in that section's **Full Changelog** line with that same version.
- Delete any heading that has no bullets, and delete the HTML comment under the heading.
- Open the next cycle:

```
python dev/new_changelog_entry.py
```

That inserts a new empty `## Unreleased` section above the version just written. Its comparison line points at that version and leaves `vX.Y.Z` for the one after. The script refuses to run while an `Unreleased` section is still present, so rename the current section first. Commit the result with the release. The first pull request after the release then already has a section to fill in.

Create the GitHub release notes from this changelog with `extract_release_notes.py`:

```
python dev/extract_release_notes.py X.Y.Z
```

Use the resulting markdown file only as input for the GitHub release command.