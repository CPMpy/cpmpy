# Agent Skills

This directory holds **Agent Skills** for CPMpy. They follow the open
[Agent Skills specification](https://agentskills.io/specification).

## Using skills in a project

Skills are **authored here** (`skills/<name>/`) and **copied into the installed
package** at `cpmpy/.agents/skills/` when you build or `pip install` CPMpy (including
editable installs), via [`sync_into_package.py`](sync_into_package.py). That copy is
gitignored; do not edit it by hand.

To link them into a consuming project's `.agents/skills/` (so Cursor, Claude Code,
etc. can load them):

```bash
pip install 'cpmpy[skills]'    # CPMpy plus library-skills
cpmpy skills                   # interactive install into the current project
```

After upgrading CPMpy, refresh the project's links to the newly installed content
(this does not run `pip` itself):

```bash
pip install -U 'cpmpy[skills]'
cpmpy skills update
```

To also rebuild generated reference files (`*.generated.md`) from the Sphinx docs,
use a **source checkout** and the docs extra:

```bash
pip install -e '.[docs]'           # Sphinx + sphinx_llm
cpmpy skills update --regenerate
```

`--regenerate` is only advertised in `cpmpy skills --help` for a source/editable
install; a wheel from PyPI has no docs tree to rebuild from. It runs
`skills/generate_references.py --build`, recopies `skills/` into
`cpmpy/.agents/skills/`, then `library-skills --yes`. `cpmpy[docs]` is
enough if `cpmpy[skills]` is already installed.

A clone of this repo includes `*.generated.md` so `npx skills add` and a
plain `pip install -e .` both get the reference files. Rebuild them when
docs change (`pip install 'cpmpy[docs]'` then `cpmpy skills update --regenerate`).

`cpmpy skills update` is `library-skills --yes`. Any other arguments are forwarded
to [library-skills](https://library-skills.io/), e.g. `cpmpy skills install -y --claude`
or `cpmpy skills list`. Without the extra, `cpmpy skills` tells you to
`pip install 'cpmpy[skills]'`.

The `skills/` tree is also what `npx skills add` looks at in this git repo,
including committed `*.generated.md` reference files.