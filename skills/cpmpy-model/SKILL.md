---
name: cpmpy-model
description: Write and solve problems in CPMpy, a constraint programming and modelling library in Python, based on numpy, with direct solver access.
license: Apache-2.0
compatibility: Requires Python 3.10+
metadata:
  author: CPMpy team
  domain: constraint-programming
  library: cpmpy
---

# CPMpy modeling

Model and solve constrained (optimisation) problems using CPMpy with one of its many solver backends across SMT, CP, ILP, PB, (max)SAT. Model once in a numpy-style high-level language, using **decision variables** plus **constraints** over them, optionally with an **objective**, then solve with any backend.

For an overview of the available backends and their abilities: [index](https://cpmpy.readthedocs.io/en/latest/index.html)

For a compact, complete API listing: [references/api-cheatsheet.generated.md](references/api-cheatsheet.generated.md)

For the full narrative documentation (global constraints catalog, solver selection, incremental solving, I/O, debugging), see [modeling](https://cpmpy.readthedocs.io/en/latest/modeling.html).

