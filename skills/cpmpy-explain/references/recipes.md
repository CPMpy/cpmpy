# CPMpy explanation recipes

Worked patterns for `cpmpy.tools.explain`. Every snippet is self-contained and runs with only `ortools` installed.

## Contents

- [Preferred explanation with QuickXplain](#preferred-explanation-with-quickxplain)
- [Weighted (optimal) MUS](#weighted-optimal-mus)
- [Constrained explanation with OCUS](#constrained-explanation-with-ocus)
- [Weighted repair with MCS / MSS](#weighted-repair-with-mcs--mss)
- [Enumerate all conflicts and repairs with MARCO](#enumerate-all-conflicts-and-repairs-with-marco)
- [Why does the solver not find my solution?](#why-does-the-solver-not-find-my-solution)
- [Low level: assumptions and unsat cores](#low-level-assumptions-and-unsat-cores)
- [More involved tasks](#more-involved-tasks)

## Preferred explanation with QuickXplain

`quickxplain` returns the MUS that is lexicographically preferred according to the **order** of `soft`: put the constraints you would *rather* see in the explanation first.

```python
import cpmpy as cp
from cpmpy.tools.explain import quickxplain

x = cp.intvar(0, 10, shape=3, name="x")
soft = [
    x[0] + x[1] <= 4,   # most preferred in the explanation
    x[0] >= 3,
    x[1] >= 3,
    x[2] == x[0] + x[1],
    x[2] >= 7,          # least preferred
]
print(quickxplain(soft))
# [x[0] + x[1] <= 4, x[0] >= 3, x[1] >= 3]
```

## Weighted (optimal) MUS

Use weights to say how *hard to understand* each constraint is. `optimal_mus` minimises `sum(weights)` over all MUSes; `smus` is the unweighted special case (fewest constraints).

```python
import cpmpy as cp
from cpmpy.tools.explain import optimal_mus, smus

x = cp.intvar(0, 10, shape=3, name="x")
soft = [
    cp.AllDifferent(x),   # global constraint: costly to explain
    x[0] == x[1],
    x[0] >= 5,
    x[1] <= 4,
]
weights = [10, 1, 1, 1]

print(smus(soft))                    # one of the 2-constraint MUSes
print(optimal_mus(soft, weights=weights))  # [x[0] == x[1], x[0] >= 5, x[1] <= 4], avoids AllDifferent
```

## Constrained explanation with OCUS

`ocus` adds a `meta_constraint`: a Boolean CPMpy expression *over the soft constraints themselves*. Each soft constraint stands for "this constraint is in the explanation". Useful to force or forbid constraints, or to require "at most one of this group".

```python
import cpmpy as cp
from cpmpy.tools.explain import ocus

x = cp.intvar(0, 10, shape=3, name="x")
soft = [
    x[0] == x[1],
    x[0] >= 5,
    x[1] <= 4,
    x[0] + x[1] <= 3,
]

# explanation must use the sum constraint
print(ocus(soft, meta_constraint=soft[3]))
# explanation must not use the equality
print(ocus(soft, meta_constraint=~soft[0]))
```

If no MUS satisfies the meta-constraint, `ocus` raises `cpmpy.tools.explain.OCUSException`.

## Weighted repair with MCS / MSS

`mcs_opt` returns the cheapest set of constraints to remove; `mss_opt` the most valuable set to keep. Weights = the cost of dropping each constraint (e.g. priority of a user request).

```python
import cpmpy as cp
from cpmpy.tools.explain import mcs_opt, mss_opt

slot = cp.intvar(1, 3, shape=3, name="slot")   # meeting slot of 3 people
requests = {
    "alice wants slot 1": slot[0] == 1,
    "bob wants slot 1":   slot[1] == 1,
    "carol wants slot 2": slot[2] == 2,
    "carol wants to meet alice": slot[2] == slot[0],
}
priority = [3, 1, 2, 5]
hard = [cp.AllDifferent(slot[:2])]              # alice and bob can't share

soft = list(requests.values())
label = {c: n for n, c in requests.items()}
print("drop:", [label[c] for c in mcs_opt(soft, hard, priority)])
print("keep:", [label[c] for c in mss_opt(soft, hard, priority)])
```

`mcs_grow` / `mss_grow` give a subset-minimal (not cost-optimal) answer without needing optimisation, and are often faster on large models.

## Enumerate all conflicts and repairs with MARCO

`marco` is a generator. Stop early when you have enough; the full enumeration can be exponential.

```python
import itertools
import cpmpy as cp
from cpmpy.tools.explain import marco

x = cp.boolvar(shape=3, name="x")
soft = [x[0], x[0] | x[1], x[2].implies(x[1]), ~x[0], ~x[1]]

for kind, cons in itertools.islice(marco(soft), 10):
    print(kind, cons)

# only MUSes
muses = [cons for _, cons in marco(soft, return_mcs=False)]
```

Pass `map_solver="gurobi"` or `"exact"` (incremental) for large instances.

## Why does the solver not find my solution?

Add the expected solution as hard constraints, then explain the now-UNSAT model.

```python
import cpmpy as cp
from cpmpy.tools.explain import mus

x = cp.intvar(0, 9, shape=3, name="x")
model = cp.Model(
    cp.AllDifferent(x),
    x[0] + x[1] == x[2],
    x[2] < 5,
)
expected = [x[0] == 2, x[1] == 3, x[2] == 5]
print(mus(soft=model.constraints, hard=expected))
# [x[2] < 5]  -> this is the constraint that rules out the expected solution
```

For an optimisation model, explain the feasibility part only (`model.constraints`), optionally with `hard=expected + [model.objective_ == value]`.

## Low level: assumptions and unsat cores

All default tools build on `make_assump_model` + `solve(assumptions=...)` + `get_core()`. Use this directly to write your own algorithm or to reuse one incremental solver.

```python
import cpmpy as cp
from cpmpy.tools.explain import make_assump_model

x = cp.intvar(0, 9, shape=3, name="x")
soft = [x[0] > x[1], x[1] > x[2], x[2] > x[0]]

model, soft, assump = make_assump_model(soft, name="a")  # adds a[i] -> soft[i]
s = cp.SolverLookup.get("ortools", model)
assert s.solve(assumptions=assump) is False
core = s.get_core()                       # subset of assump, not necessarily minimal
amap = dict(zip(assump, soft))
print([amap[a] for a in core])
```

OR-Tools is not incremental across `solve(assumptions=...)` calls; `exact`, `pysat`, `z3` are.

## More involved tasks

Step-wise explanations (hints), explaining optimality, "why X and not Y", interactive repair, slack relaxation and counterfactuals: see [advanced.md](advanced.md).
