---
name: cpmpy-explain
description: Explain why a CPMpy model is unsatisfiable, or why an expected solution is missing, using cpmpy.tools.explain (MUS, QuickXplain, smallest/optimal MUS, OCUS, MSS, MCS, MARCO enumeration). Use when a CPMpy model returns UNSAT or a wrong solution, when the user asks which constraints conflict, what to relax or remove to make it feasible, why a solution is optimal or why X and not Y, or wants step-by-step explanations or hints (e.g. for puzzles like sudoku) of a CPMpy model.
license: Apache-2.0
compatibility: Requires Python 3.10+ and cpmpy
metadata:
  author: CPMpy team
  domain: constraint-programming
  library: cpmpy
---

# CPMpy explanation tools

Find *which constraints* cause a CPMpy model to be unsatisfiable, and *which constraints to drop or relax* to make it satisfiable. All tools live in `cpmpy.tools.explain` and work on arbitrary CPMpy expressions, including global constraints and nested expressions.

## Contents

- [Which tool answers which question](#which-tool-answers-which-question)
- [Quick start](#quick-start)
- [Solver requirements](#solver-requirements)
- [Making explanations readable](#making-explanations-readable)
- [Workflow: debugging an unexpected result](#workflow-debugging-an-unexpected-result)
- [More involved tasks](#more-involved-tasks)
- [Common issues](#common-issues)
- [Further reading](#further-reading)
- Worked recipes (QuickXplain, weighted MUS, OCUS, MCS/MSS, MARCO, assumptions): [references/recipes.md](references/recipes.md)
- Advanced use cases (step-wise hints, optimality, "why X not Y", repair, slack, counterfactuals): [references/advanced.md](references/advanced.md)

## Which tool answers which question

All functions take `soft` (the constraints that may be part of the answer) and optionally `hard` (always enforced, never reported), and return a **list of constraints from `soft`**.

| User question | Concept | Function |
|---|---|---|
| "Why is it UNSAT?" (any small conflict) | MUS: Minimal Unsatisfiable Subset | `mus(soft, hard)` |
| "Why is it UNSAT?" in terms of the constraints I care about most | Preferred MUS (order of `soft` = preference, earlier = preferred) | `quickxplain(soft, hard)` |
| "Give me the smallest reason" | Smallest MUS | `smus(soft, hard)` |
| "Give me the cheapest reason" (some constraints are harder to understand) | Optimal MUS, minimises `sum(weights)` | `optimal_mus(soft, hard, weights)` |
| "Give me the cheapest reason that satisfies extra conditions" | OCUS: Optimal *Constrained* US | `ocus(soft, hard, weights, meta_constraint)` |
| "What can I keep?" | MSS: Maximal Satisfiable Subset | `mss(soft, hard)`, weighted: `mss_opt(soft, hard, weights)` |
| "What is the least I must remove/relax?" | MCS: Minimal Correction Subset (complement of MSS) | `mcs(soft, hard)`, weighted: `mcs_opt(soft, hard, weights)` |
| "Show me all conflicts / all fixes" | MUS + MCS enumeration | `marco(soft, hard)`, a generator yielding `("MUS", cons)` / `("MCS", cons)` |

Fixing one MUS does not make the model satisfiable when there are several conflicts. To actually *repair* a model, use an MCS. To *understand* it, use a MUS.

## Quick start

```python
import cpmpy as cp
from cpmpy.tools.explain import mus, mcs

# Example UNSAT model
x = cp.boolvar(shape=3, name="x")
model = cp.Model(
    x[0],
    x[0] | x[1],
    x[2].implies(x[1]),
    ~x[0],
)
assert model.solve() is False

print(mus(model.constraints))   # [x[0], ~x[0]]   -> the conflict
print(mcs(model.constraints))   # [~x[0]] or [x[0]] -> remove this to become SAT
```

Domain knowledge belongs in `hard`. Constraints that are true by definition (channeling, the structure of the problem) should go in `hard` so they never show up in the explanation. Only the constraints the user can reason about or change go in `soft`:

```python
mus(soft=user_requirements, hard=structural_constraints)
```

## Solver requirements

Use `cpmpy version` to check which solvers are installed. The non-naive tools need *assumption support*: the solver must accept `solve(assumptions=[...])` and return an unsat core through `get_core()` (see [unsat core extraction](https://cpmpy.readthedocs.io/en/latest/unsat_core_extraction.html.md)). A solver without it raises `NotSupportedError` from `get_core()`. To check:

```python
from cpmpy.solvers.solver_interface import SolverInterface
cls = cp.SolverLookup.lookup("gurobi")
has_assumptions = cls.get_core is not SolverInterface.get_core   # False for gurobi
```

When the user's solver has no assumption support, use the `*_naive` variant **with that solver**, or explain with `ortools` instead.

## Making explanations readable

The tools return the constraint objects you passed in, so **the granularity and names of `soft` decide how good the explanation is**.

- **Group related constraints** into one soft constraint with `cp.all(group)`. Do **not** pass a nested Python list: `soft` is flattened, so `[[c1, c2], [c3]]` is treated as `[c1, c2, c3]`.
- **Keep a dict from soft constraint to a human label**, and translate the result back:
  ```python
  groups = {
      "all shifts are covered": cp.all(coverage_constraints),
      "nurse A is off on Monday": shifts[A, MON] == OFF,
      "at most 5 shifts per nurse": cp.all(max_shift_constraints),
  }
  label = {c: name for name, c in groups.items()}
  conflict = mus(list(groups.values()), hard=structural)
  print([label[c] for c in conflict])
  ```
- **Give decision variables meaningful, unique names** (`cp.intvar(..., name="start")`), so `print(constraint)` reads well.

## Workflow: debugging an unexpected result

1. **UNSAT model**: `print(model)` first, then call `mus(model.constraints)` and inspect the (much smaller) set. Use `mcs` to find what to relax.
2. **SAT, but a known-good solution is rejected**: add that solution as constraints, which makes the model UNSAT, then explain:
   ```python
   expected = [x == 1, y == 2, z == 3]
   conflict = mus(soft=model.constraints, hard=expected)  # constraints that forbid the expected solution
   ```
   For an optimisation model, drop the objective. To check that a value is reachable, add `model.objective_ == value` as a hard constraint instead.
3. **Error while solving**: this is not a job for the explanation tools. See [how to debug](https://cpmpy.readthedocs.io/en/latest/how_to_debug.html.md).

## More involved tasks

Most other explanation questions reduce to a MUS/MCS call after adding the right constraints. Details, code and reference implementations: [references/advanced.md](references/advanced.md).

| Task | Reduction |
|---|---|
| Step-wise explanation / "give me a hint" (e.g. sudoku) | `ocus(rules + facts + negated_targets, weights, meta_constraint=cp.sum(negated_targets) == 1)`, repeated |
| "Why is there no better solution?" | `mus(model.constraints, hard=[model.objective_ < best])` |
| "Why X and not Y?" | `mus(model.constraints, hard=[Y])`, or re-solve with `Y` and compare |
| Interactive repair of an infeasible model | loop: `mus` → user drops a constraint → repeat; or `mcs_opt` in one go |
| Relax numeric constraints instead of dropping them | slack variables + minimise violation |
| "Which weights make my solution optimal?" | inverse optimisation (XCP-explain `counterfactual.py`) |

These loops call a solver many times. On bigger instances, use an incremental SAT solver (`solver="pindakaas"`/`"exact"`/`"pysat"`), and an incremental hitting-set solver (`hs_solver="gurobi"`/`"exact"`) for `ocus`.

## Common issues

| Problem | Likely cause | Fix |
|---|---|---|
| `AssertionError: MUS: model must be UNSAT` / `The model should be UNSAT!` | `hard + soft` is satisfiable, **or** the solver silently ignores assumptions | Check `cp.Model(hard + soft).solve()` is `False`; use an assumption-capable solver or the `*_naive` variant |
| `NotSupportedError: ... unsat core extraction` | Solver has no assumption support | Use `ortools` or the `*_naive` variant |
| `mus`/`smus` return `[]`, `quickxplain` raises `ValueError: max() ...`, `mcs` raises a bare `AssertionError` | `hard` on its own is UNSAT | Check `cp.Model(hard).solve()`; move constraints from `hard` to `soft` |
| Explanation lists many tiny constraints | Too fine-grained `soft` | Group with `cp.all(...)` per user-level requirement |
| `OCUSException` | No unsatisfiable subset meets the `meta_constraint` | Relax the meta-constraint |
| `smus` / `optimal_mus` / `marco` slow | Many hitting-set iterations | Use `mus`/`quickxplain` (subset-minimal, not optimal), coarser groups, or `hs_solver="gurobi"`/`"exact"` |
| `mus` gives a different MUS each run/solver | Many MUSes exist; `mus` returns *a* MUS | Use `quickxplain` (preference order), `smus`/`optimal_mus` (cost), or `marco` (all) |

## Further reading

- Worked recipes (weights, OCUS meta-constraints, MARCO enumeration): [references/recipes.md](references/recipes.md)
- Advanced use cases (step-wise hints, optimality, counterfactuals, ML + CP), with links to reference implementations: [references/advanced.md](references/advanced.md)
- Explainable constraint solving tutorial (nurse rostering, all techniques): https://github.com/CPMpy/XCP-explain
- Debugging guide: https://cpmpy.readthedocs.io/en/latest/how_to_debug.html.md
- Assumptions and unsat cores: https://cpmpy.readthedocs.io/en/latest/unsat_core_extraction.html.md
- API docs: https://cpmpy.readthedocs.io/en/latest/api/tools/explain.html
- Full documentation for agents: https://cpmpy.readthedocs.io/en/latest/llms-full.txt
- Advanced examples: https://github.com/CPMpy/cpmpy/tree/master/examples/advanced (`ocus_explanations.py`, `explain_stepwise_csp.ipynb`, `counterfactual_explain.py`, `quickxplain.py`, `marco_musmss_enumeration.py`)
