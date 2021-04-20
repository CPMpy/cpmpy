# UnSAT core extraction with assumption variables

When a model is unsatisfiable, it can be desirable to get a better idea of which Boolean variables make it unsatisfiable. Commonly, these Boolean variables are 'switches' that turn constraints on, hence such Boolean variables can be used to get a better idea of which _constraints_ make the model unsatisfiable.

In the SATisfiability literature, the Boolean variables of interests are called _assumption_ variables and the solver will assume they are true. The subset of these variables that, when true, make the model unsatisfiable is called an unsatisfiable _core_.

Lazy Clause Generation solvers, like or-tools, are built on SAT solvers and hence can inherit the ability to define assumption variables and extract an unsatisfiable core.

Since version 8.2, or-tools supports declaring assumption variables, and extracting an unsat core. We also implement this functionality in CPMpy, using PySAT-like `s.solve(assumptions=[...])` and `s.get_core()`:

```python
bv = BoolVar(shape=3)
iv = IntVar(0,9, shape=3)

# circular 'bigger then', UNSAT
m = Model([
    bv[0].implies(iv[0] > iv[1]),
    bv[1].implies(iv[1] > iv[2]),
    bv[2].implies(iv[2] > iv[0])
])

s = CPMpyORTools(m)
print(s.solve(assumptions=bv))
print(s.status())
print("core:", s.get_core())
print(bv.value())
```

This opens the door to more advanced use cases, such as Minimal Unsatisfiable Subsets and QuickXplain-like tools to help debugging. We welcome any examples or additions that use CPMpy in this way!! Here is one example: the [MARCO algorithm for enumerating all MUS/MSSes](http://github.com/tias/cppy/tree/master/examples/advanced/marco_musmss_enumeration.py); also useful to find/debug a single MUS!

One final caveat is that the or-tools Python interface is by design _stateless_. That means that, unlike in PySAT, calling `s.solve(assumptions=bv)` twice for a different `bv` array does NOT REUSE anything from the previous run: no warm-starting, no learnt clauses that are kept, no incrementality, so there will be some pre-processing overhead. If you know of another CP solver with a (Python) assumption interface that is incremental, let us know!!

A final-final note is that you can manually warm-start or-tools with a previously found solution with s.solution\_hint(); see also the MARCO code linked above.

For
