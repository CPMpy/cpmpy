---
name: cpmpy-model
description: Write and solve problems in CPMpy, a constraint programming and modelling library in Python, based on numpy, with direct solver access. Use this skill when writing models in CPMpy.
license: Apache-2.0
compatibility: Requires Python 3.10+. Solver backends other than OR-Tools need their own packages through extras.
metadata:
  author: CPMpy team
  domain: constraint-programming
  library: cpmpy
---

# CPMpy modeling

Model and solve constrained (optimisation) problems using CPMpy with one of its many solver backends across SMT, CP, ILP, PB, (max)SAT. Model once in a numpy-style high-level language, using **decision variables** plus **constraints** over them, optionally with an **objective**, then solve with any backend.

For an overview of the available backends and their abilities: [index](https://cpmpy.readthedocs.io/en/latest/index.html)

For a compact, complete API listing: [references/api-cheatsheet.generated.md](references/api-cheatsheet.generated.md)

For the full narrative documentation (global constraints catalog, solver selection, incremental solving, I/O, debugging), see [documentation](https://cpmpy.readthedocs.io/en/latest/llms-full.txt).


## Solving

Status values: [solver interface](https://cpmpy.readthedocs.io/en/latest/api/solvers/solver_interface.html).

| Status | Meaning |
|---|---|
| `OPTIMAL` | optimisation: objective proven optimal |
| `FEASIBLE` | solution found, optimality not proven (time limit), or a satisfaction problem solved |
| `UNSATISFIABLE` | proven: no solution |
| `UNKNOWN` | limit reached with no solution found |

Satisfaction problems report `FEASIBLE`, not `OPTIMAL`. Read values with `var.value()`, and the objective with `model.objective_value()`. Both are `None` until a solution exists; read them only after `solve()` returned `True`. Options for the solver backend must be passed as keyword arguments:

```python
model.solve(solver="ortools", time_limit=10, num_workers=1, random_seed=7)
```

A model can also be extended after solving: `model.add(constraint)` and solve again.

## Further reading

- Documentation: [cpmpy.readthedocs.io](https://cpmpy.readthedocs.io/en/latest/)
- API reference: [model](https://cpmpy.readthedocs.io/en/latest/api/model.html), [global constraints](https://cpmpy.readthedocs.io/en/latest/api/expressions/globalconstraints.html), [OR-Tools backend](https://cpmpy.readthedocs.io/en/latest/api/solvers/ortools.html)
- Examples: [CPMpy examples folder](https://github.com/CPMpy/cpmpy/tree/master/examples)