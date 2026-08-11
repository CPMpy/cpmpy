# AGENTS.md — CPMpy

Guidance for AI agents working in this repository. Prefer matching existing code over inventing new conventions. When unsure, ask.

Longer human docs: `docs/developers.md`, `docs/adding_solver.md`, `tests/README.md` (also `docs/testing.md`). Prefer those over inventing process; this file is the short agent-oriented subset.

## Environment

Use this checkout’s development interpreter (conda/venv already configured for the machine).
Prefer `python -m pip` / `python -m pytest` / `python -m mypy` so tools share that env.

## What this repo is

CPMpy is a constraint modeling library (`import cpmpy as cp`). Typical flow: users build **expressions**, put them in a **Model**, then a **solver** **transforms** those expressions into forms it supports and posts them to the backend API.

| Area | Path | Notes |
|------|------|--------|
| Package | `cpmpy/` | Installable library |
| Model | `cpmpy/model.py` | `Model` container |
| Exceptions | `cpmpy/exceptions.py` | CPMpy-specific errors |
| Expressions | `cpmpy/expressions/` | Variables, operators, globals |
| Transformations | `cpmpy/transformations/` | Copy-on-write rewrites |
| Solvers | `cpmpy/solvers/` | `CPM_<name>` interfaces |
| Tools | `cpmpy/tools/` | Use CPMpy; not part of the core |
| CLI | `cpmpy/cli.py` | `cpmpy version` and related entry points |
| Tests | `tests/` | pytest |
| Docs | `docs/` | Sphinx; publishes to readthedocs |
| Examples | `examples/` | Extra examples welcome; also `tests/test_examples.py` |
| Dev scripts | `dev/` | Maintainer scripts/notes; not shipped |

Public API is re-exported from `cpmpy/__init__.py`. Prefer `import cpmpy as cp` in user-facing / test code; never `from cpmpy import *`.

### Import map (library work)

- Variables: `cpmpy.expressions.variables` (`boolvar`, `intvar`, `cpm_array`, `NegBoolView`, …). Names with a leading `_` (e.g. `_BoolVarImpl`) are private — do not use them from outside the package.
- Core expr: `cpmpy.expressions.core` (`Expression`, `Comparison`, `Operator`, `BoolVal`, …)
- Globals: `globalconstraints.py` / `globalfunctions.py` (each global should implement `decompose()`; solvers may still post some natively via `supported_global_constraints`)
- Builtins: `python_builtins.py` (`cp.sum`, `cp.all`, …)
- Helpers: `expressions.utils` (`is_int`, `is_any_list`, `eval_comparison`, …)
- Solvers: `SolverInterface` / `SolverLookup` in `solvers/`
- Never import an optional solver package at module top level — only inside `supported()` and methods that need it

### Transformation waterfall (solver-facing)

Transformations are **copy-on-write** (return new expressions; never mutate inputs). There is no single global pipeline: each solver’s `transform()` chains only what it needs. Lower-level solvers typically reuse the early steps of higher-level ones (the “waterfall”). Always read the target solver’s `transform()` — do not invent an order from the `transformations/` module list. The autosummary order in `cpmpy/transformations/__init__.py` is **illustrative, not normative**.

Solver interfaces are **eager**: `add()` immediately calls `transform()` and posts the result to the backend (this is what enables incremental use). Transformation helpers are **stateless functions** (expressions in → expressions out). The usual exception is **CSE**: the solver owns a `csemap` dict and passes it into transforms that support it — see `docs/adding_solver.md`.

Canonical examples:

**OR-Tools** (`cpmpy/solvers/ortools.py`):

`toplevel_list` → `no_partial_functions` → `push_down_negation` → `decompose_in_tree` → `flatten_constraint` → `reify_rewrite` → `only_numexpr_equality` → `only_bv_reifies` → `only_implies`

**PySAT** (`cpmpy/solvers/pysat.py`) — continues further toward Boolean/linear form:

`toplevel_list` → `no_partial_functions` → `push_down_negation` → `decompose_linear` → `simplify_boolean` → `flatten_constraint` → `linearize_reified_variables` → `only_bv_reifies` → `only_implies` → `linearize_constraint` → `int2bool` → `only_positive_coefficients`

Shared early pattern across many solvers: normalize → safen → push negations → decompose unsupported globals → flatten; then solver-specific reification / comparison / linearize / int2bool steps.

New solvers: copy `solvers/TEMPLATE.py`, follow `docs/adding_solver.md` (including registration in `SolverLookup.base_solvers()`, `solvers/__init__.py`, docs, and CI where applicable).

## Scope of changes

- Prefer small, focused diffs. No drive-by refactors.
- Base PRs on current `master`. Only documentation changes land directly on `master`; everything else goes through a PR.
- PRs must pass the test suite and mypy (`mypy cpmpy tests`, see `.github/workflows/python-linting.yml`). Bugfixes should include a regression test (typically the bug-report case).
- WIP PRs are fine — most changes go through at least one review iteration.
- If something should be usable as `cp.*`, re-export it from `cpmpy/__init__.py`. User-facing behavior changes should update Sphinx docs under `docs/` as well.
- Do not git commit unless the user asks.

## Code style

Match the code already around the change. Prefer small, focused edits over clever structure.

CPMpy-specific:

- Prefer `list.extend([...])` / `list.append(x)` over `lst += [...]`.
- Be explicit about bool checks: `if arg is None`, `if len(arg) == 0`, not bare `if arg:`.
- Reuse existing helpers (`is_num`, `is_any_list`, `eval_comparison`, `get_variables`, …) and the same naming / argument order as neighboring code.
- Prefer existing globals / `cp.sum` / `cp.all` over ad-hoc reimplementations.
- New solvers: follow `CPM_<name>` and `solvers/TEMPLATE.py`; see `docs/adding_solver.md`.
- Never import an optional solver package at module top level — only inside `supported()` and methods that need it.

### Typing

Public functions and methods should have type hints on arguments and return values (`docs/developers.md`). Keep hints accurate (`Optional[...]` when `None` is allowed); do not paper over with overly broad types or unnecessary `# type: ignore` / `cast`.

- Prefer typing public surfaces and anything mypy already covers; match neighboring code density — do not mass-annotate unrelated private helpers in the same PR.
- Prefer existing expression aliases where they fit (`NestedBoolExprLike`, `ExprLike`, … from `cpmpy.expressions.core`).
- Modern builtins are fine (`list[...]`, `dict[...]`); many modules also use `from __future__ import annotations`.
- Changes must keep `mypy` clean (`mypy.ini`). CI runs `mypy cpmpy tests`. Locally: `python -m mypy cpmpy tests`.
- Docstrings should still mention types in `Arguments` / `Returns` even when the signature is annotated.

### Docstrings

Document non-obvious code: clarifying line comments where needed, plus docstrings on methods, classes, and modules. Google-ish layout (`cpmpy/model.py` is the reference):

```
Description of the method

Arguments:
  - arg (type): ...
  - arg2 (type): ... (default: ...)

Returns:
  - name (type): description
```

- Use Sphinx backlinks whenever referencing code. `:func:` for module-level functions, `:meth:` for class/instance methods. Targets in other modules need the full qualified path in `<...>`:
  ``:func:`boolvar() <cpmpy.expressions.variables.boolvar>` ``,
  ``:meth:`model.minimize(obj) <cpmpy.model.Model.minimize>` ``,
  ``:meth:`cp.SolverLookup.get("ortools") <cpmpy.solvers.utils.SolverLookup.get>` ``.
- Inline code in double backticks; multi-line in ``.. code-block:: python``.
- Document defaults as ``(default: ...)``.
- Include argument/return types in the docstring when possible (in addition to signature annotations).

## Modeling reminders (for tests / examples only)

```python
import cpmpy as cp

x = cp.intvar(1, 10, shape=3)
m = cp.Model(cp.AllDifferent(x), x[0] == 1)
m.maximize(cp.sum(x))
assert m.solve()
```

- Integers only (no floats/fractions).
- Use CPMpy builtins for vectorized ops (`cp.sum`, `cp.any`, …).
- Prefer global constraints/functions when they fit.
- To index non-NDVarArrays with an expression, use `cp.cpm_array(...)[idx]`.
- Solvers: `cp.SolverLookup.get(name, model=None)`; list installed/ready ones with `cp.SolverLookup.supported()` (`solvernames()` is deprecated); overview via `cpmpy version` CLI.

## Writing tests

Full suite docs: `tests/README.md` (also included as `docs/testing.md`). Follow patterns in neighboring test files.

### Where to put tests

| Change | Typical file(s) |
|--------|------------------|
| Model | `test_model.py` |
| Expressions / ops | `test_expressions.py`, `test_builtins.py`, `test_variables.py` |
| Globals (constraints & functions) | `test_globalconstraints.py`, also `test_constraints.py` |
| Solve constraints | `test_constraints.py` |
| Transformations | `test_trans_*.py`, `test_transf_*.py`, `test_flatten.py`, `test_cse.py`, … |
| Solver high-level | `test_solvers.py` |
| Solver low-level API | `test_solverinterface.py` |
| `solveAll` | `test_solveAll.py` |
| Tools | `test_tools_*.py`, `test_tool_*.py` |
| Solver-only backend | `test_<solver>_*.py` or mark with `requires_solver` |

Prefer extending an existing file over adding a new one unless the area is clearly separate.

### How to structure a test

- Use `import cpmpy as cp` and pytest.
- Descriptive names; short docstring when the intent is not obvious from the name.
- Prefer small, deterministic models (known sat/unsat or fixed domains).
- For bugfixes: name or comment with the issue (e.g. `test_bug_168`, “from #143”) and assert the previously broken behavior.
- When checking constraint semantics, test both `.solve()` outcome and `.value()` after assigning (see `test_globalconstraints.py` oracle tuples).
- Reset private counters in `setup_method` only when the test relies on variable naming/ids (existing globals tests do this).
- Helpers shared across tests live in `tests/utils.py` (e.g. `skip_on_missing_pblib`) — do not invent parallel skip logic.

Minimal pattern:

```python
import cpmpy as cp

def test_basic_model():
    x = cp.intvar(0, 10, name="x")
    m = cp.Model(x >= 5)
    assert m.solve()
    assert x.value() >= 5
```

### Solver fixtures and markers

Use the fixtures/markers from `tests/conftest.py` — do not hardcode skip logic for “solver not installed” when a marker already covers it. Related skip helpers also live in `tests/utils.py` (e.g. `skip_on_missing_pblib`); prefer a `conftest` marker when it fits, otherwise reuse those helpers instead of inventing new skip logic.

- **`solver` fixture** — for tests that should run under `--solver=…` parametrisation. A `solver` argument is enough for pytest to inject it (functions, or methods on a class). Prefer also marking with `@pytest.mark.usefixtures("solver")` so the dependency is explicit; on classes that use `self.solver`, that mark is required unless every method takes a `solver` argument.
- **`@pytest.mark.requires_solver("name", …)`** — only those solvers; **must** declare a `solver` parameter. Skipped if not installed.
- **`@pytest.mark.requires_dependency("package")`** — optional Python package.
- **`@pytest.mark.generate_constraints.with_args(generator)`** — parametrise `constraint` (see `test_constraints.py`).
- **`@pytest.mark.depends_on_solver`** — indirect solver dependence.
- **`@pytest.mark.dataset_download`** — needs network dataset downloads; skipped unless `--include-datasets` is passed.

Examples:

```python
@pytest.mark.usefixtures("solver")
def test_with_any_selected_solver(solver):
    x = cp.intvar(0, 10)
    assert cp.Model(x >= 5).solve(solver=solver)

@pytest.mark.requires_solver("cplex")
def test_cplex_only(solver):
    ...

@pytest.mark.usefixtures("solver")
class TestMyFeature:
    def test_with_solver(self):
        assert cp.Model(cp.boolvar()).solve(solver=self.solver)
```

Make tests solver-agnostic when the behavior is only part of the modeling-side of things; use `solver` fixtures for all tests that require a solver-call. Prefer patterns already used in the neighboring test file.

### Running tests (agents)

```sh
python -m pytest tests/ --ignore=tests/test_examples.py
python -m pytest tests/test_model.py -n auto
python -m pytest tests/ --solver=ortools  # run tests with ortools as solver
python -m pytest tests/ --solver=all      # run tests with every installed solver
python -m pytest tests/ --solver=None     # literal string None: skip solver-parametrised tests
```

**Always** pass `--ignore=tests/test_examples.py` in agent runs unless the user explicitly wants examples exercised.

`--solver` currently mainly affects a subset (notably `test_constraints.py`, `test_solverinterface.py`, `test_solvers_solhint.py`). Omitting `--solver` is not the same as `--solver=None`: default without the flag uses OR-Tools for non-specific tests and still runs installed solver-specific tests.

Type check (required for CI): `python -m mypy cpmpy tests` (see `mypy.ini`).

## Review checklist (quick)

When reviewing or before finishing a change:

- [ ] Correctness + edge cases; no public-API breakage
- [ ] Consistent with neighboring modules / waterfall / solver template
- [ ] No top-level optional solver imports
- [ ] Docstrings and accurate type hints on public surfaces; `mypy cpmpy tests` clean
- [ ] Style: extend/append, explicit None/len checks
- [ ] Tests updated; bugfix has a regression test
- [ ] Relevant pytest tests pass
