# Solving with assumptions

When a model is unsatisfiable, it can be desirable to get a better idea of which Boolean variables make it unsatisfiable. Commonly, these Boolean variables are 'switches' that turn constraints on or off, hence such Boolean variables can be used to get a better idea of which _constraints_ make the model unsatisfiable.

In the satisfiability literature, the Boolean variables of interests are called _assumption literals_ and the solver will assume they are true. Any subset of these assumptions that, when true, makes the model unsatisfiable is called an unsatisfiable _core_.

Lazy Clause Generation solvers, like OR-Tools, are built on SAT solvers and hence can inherit the ability to define assumptions and extract an unsatisfiable core.

Since version 8.2, OR-Tools supports declaring assumptions, and extracting an unsat core. We also implement this functionality in CPMpy, using PySAT-like `s.solve(assumptions=[...])` and `s.get_core()`:

```python
from cpmpy import *
from cpmpy.solvers import CPM_ortools

bv = boolvar(shape=3)
iv = intvar(0,9, shape=3)

# circular 'bigger then', UNSAT
m = Model(
    bv[0].implies(iv[0] > iv[1]),
    bv[1].implies(iv[1] > iv[2]),
    bv[2].implies(iv[2] > iv[0])
)

s = CPM_ortools(m)
print(s.solve(assumptions=bv))
print(s.status())
print("core:", s.get_core())
print(bv.value())
```

This opens the door to more advanced use cases, such as Minimal Unsatisfiable Subsets (MUS) and QuickXplain-like tools to help debugging.

In our tools, we implemented a simple MUS deletion based algorithm, using assumptions.

```python
from cpmpy.tools import mus

print(mus(m.constraints))
```

We welcome any additional examples that use CPMpy in this way! Here is one example: the [MARCO algorithm for enumerating all MUS/MSSes](http://github.com/tias/cppy/tree/master/examples/advanced/marco_musmss_enumeration.py). Here is another: a [stepwise explanation algorithm](https://github.com/CPMpy/cpmpy/blob/master/examples/advanced/ocus_explanations.py) for SAT problems (implicit hitting-set based).

More information on how to use these tools can be found in [the tools API documentation](./api/tools.rst)

One OR-Tools specific caveat is that this particular (default) solver's Python interface is by design _stateless_. That means that, unlike in PySAT, calling `s.solve(assumptions=bv)` twice for a different `bv` array does NOT REUSE anything from the previous run: no warm-starting, no learnt clauses that are kept, no incrementality, so there will be some pre-processing overhead. If you know of another CP solver with a (Python) assumption interface that is incremental, let us know!

A final-final note is that you can manually warm-start OR-Tools with a previously found solution through `s.solution_hint()`; see also the MARCO code linked above.
