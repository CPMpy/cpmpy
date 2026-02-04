# CPMpy development

This directory contains a collection of scripts and documentation used during the development of CPMpy.



| File | Description |
| - | - |
| extract_release_notes.py | Takes as input the changelog.md file and extracts + formats a specified version to serve as release notes on GitHub. |

---

## Release notes

Before publishing a new release, the `changelog.md` file must be updated. This easiest approach is to go to the [commit history page on GitHub](https://github.com/CPMpy/cpmpy/commits/master/), find the tag of the previous release, and start adding each change one by one to the changelog. Each line should contain a short description of the change (usually the title of the PR) and a link to the change (to the PR or commit). Some changes we want to emphasise, such as the addition of a new solver. To highlight these, add a bold tag in front of the change. E.g.:

```
* **New solver**: Rc2 MaxSAT solver [#729](https://github.com/CPMpy/cpmpy/pull/729)
```

The changes don't necesarily need to be in order. Try to group them per topic and sorted by importance / impact / novelty. New solvers at the top, bug fixes at the bottom. 

Further, the changes are divided into sections.

1) **Added**

    New additions to CPMpy, like new solvers, new globals, ... Any added feature to our interface that wasn't there before.

2) **Changed**

    Things that changed to our interface. These should hopefully be kept to a minimum, keeping our external interface stable. These can include things such as changing what type of exception gets thrown for a specific situation.

3) **Fixed**

    Bug fixes.

4) **Removed**

    Things that have become deprecated. Or for example when an older Python version is no longer supported.

In some situations you can add custom sections to clearly group a set of less usual changes together. E.g.:

```
### Internal improvements
... a collection of refactoring changes to the transformation waterfall
```

Once all changes have been listed, close off the section with a link to a comparison:

```
**Full Changelog**: https://github.com/CPMpy/cpmpy/compare/v<old.version>...v<new.version>
```

To create the release notes for GitHub from this changelog, use the `extract_release_notes.py` script. The resulting markdown file does not need to be commited, only use it as input for the GitHub release command.