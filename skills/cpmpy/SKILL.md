---
name: cpmpy
description: A meta skill for CPMpy. What is CPMpy, how to install it, routing to more specialised skills. Use this skill as the main entry point for tasks involving CPMpy (Constraint Programming in Python). Route requests to the appropriate specialized CPMpy skill. Use for general CPMpy questions and tasks involving multiple CPMpy capabilities.
license: Apache-2.0
compatibility: Requires Python 3.10+
metadata:
  author: CPMpy team
  domain: constraint-programming
  library: cpmpy
---

# CPMpy

CPMpy is a high-level modelling library for constraint programming in Python: write NumPy-like expressions and constraints, and a backend solver (OR-Tools CP-SAT by default) searches for solutions. A declarative model+solve paradigm. CPMpy supports many solver backends across SMT, CP, ILP, PB, (max)SAT.

Model once using **decision variables** plus **constraints** over them, optionally with an **objective**. Then solve with a backend of choice, CPMpy will take care of the most efficient translation / transformation to the solver's interface. This allows easy comparison of different solver technologies on the same problem, finding which works best on the problem at hand.


## Install

```bash
pip install cpmpy
cpmpy version
```

Python 3.10 or newer. OR-Tools is installed with CPMpy and is the default solver. Other backends come as extras, e.g. `pip install "cpmpy[z3]"`.

For an overview of the available backends and their abilities: [index](https://cpmpy.readthedocs.io/en/latest/index.html)


## Quick start

```python
import cpmpy as cp

x, y = cp.intvar(0, 5, shape=2, name=["x", "y"])
model = cp.Model(x + y >= 5)
model.minimize(2 * x + 3 * y)
if model.solve():
    print("x =", x.value(), "y =", y.value(), "cost =", model.objective_value())
    print("status:", model.status().exitstatus.name)   # OPTIMAL
```

Output: `x = 5 y = 0 cost = 10`, then `status: OPTIMAL`.


## Available skills

| Skill | Responsibility |
|---|---|
| cpmpy-model | how to model, formalise constraints, solving |
| cpmpy-explain | explain using MUS techniques, unsatisfiability, what if? |


## Routing rules

### 1. Model construction
Use `cpmpy-model` when the user wants to:
- Translate a problem into a constraint model.
- Create or modify CPMpy variables and constraints.
- Improve model formulation or performance.



When a request spans multiple capabilities:
- Select all necessary skills.
- Avoid loading unrelated skills.