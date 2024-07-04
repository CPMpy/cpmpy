# How to debug

You get an error, or no error, but also no (correct) solution... Annoying, you have a bug.

The bug can be situated in one of three layers:
- your problem specification
- the CPMpy library
- the solver

coincidentally, they are ordered from most likely to least likely. So let's start at the bottom.

If you don't have a bug yet, but are curious, here is some general advise from expert modeller [Håkan Kjellerstrand](http://www.hakank.org/):
- Test the model early and often. This makes it easier to detect problems in the model.
- When a model is not working, try to comment out all the constraints and then activate them again one by one to test which constraint is the culprit.
- Check the domains (see lower). The domains should be as small as possible, but not smaller. If they are too large it can take a lot of time to get a solution. If they are too small, then there will be no solution.


## Debugging the solver

If you get an error and have difficulty understanding it, try searching on the internet if other users have had the same.

If you don't find it, or if the solver runs fine and without error, but you don't get the answer you expect; then try swapping out the solver for another solver and see what gives...

Replace `model.solve()` by `model.solve(solver='minizinc')` for example. You do need to install MiniZinc and minizinc-python first though.

Either you have the same output, and it is not the solver's fault, or you have a different output and you actually found one of these rare solver bugs. Report on the bugtracker of the solver, or on the CPMpy github page where we will help you file a bug 'upstream' (or maybe even work around it in CPMpy).

## Debugging a modeling error

You get an error when you create an expression? Then you are probably writing it wrongly. Check the documentation and the running examples for similar examples of what you wish to express.

Here are a few quirks in Python/CPMpy:
  - when using `&` and `|`, make sure to always put the subexpressions in brackets. E.g. `(x == 1) & (y == 0)` instead of `x == 1 & y == 0`. The latter wont work, because Python will unfortunately think you meant `x == (1 & (y == 0))`.
  - you can write `vars[other_var]` but you can't write `non_var_list[a_var]`. That is because the `vars` list knows CPMpy, and the `non_var_list` does not. Wrap it: `non_var_list = cpm_array(non_var_list)` first, or write `Element(non_var_list, a_var)` instead.
  - only write `sum(v)` on lists, don't write it if `v` is a matrix or tensor, as you will get a list in response. Instead, use NumPy's `v.sum()` instead.

Try printing the expression `print(e)` or subexpressions, and check that the output matches what you wish to express. Decompose the expression and try printing the individual components and their piecewice composition to see what works and when it starts to break.

If you don't find it, report it on the CPMpy github Issues page and we'll help you (and maybe even extend the above list of quirks).

## Debugging a `solve()` error

You get an error either from CPMpy (e.g. the flattening, or the solver interface) or the solver itself is saying the model is invalid. This may be because you have modelled something impossible, or because you have a corner case that CPMpy does not yet capture.

If you have a model that fails in this way, try the following code snippet to see what constraint is causing the error:

```python
model = ... # your code, a `Model()`

for c in model.constraints:
    print("Trying",c)
    Model(c).solve()
```

The last constraint printed before the exception is the curlpit... Please report on Github. We want to catch corner cases in CPMpy, even if it is a solver limitation, so please report on the CPMpy github Issues page.

Or maybe, you got one of CPMpy's NotImplementedErrors. Share your use case with us on Github and we will implement it. Or implemented it yourself first, that is also very welcome ; )

## Debugging an UNSATisfiable model

First, print the model:

```print(model)```

and check that the output matches what you want to express. Do you see anything unusual? Start there, see why the expression is not what you intended to express, as described in 'Debugging a modeling error'.

If that does not help, try printing the 'transformed' **constraints**, the way that the solver actually sees them, including decompositions and rewrites:

```python
s = SolverLookup.get("ortools")  # or whatever solver you are using
print(s.transform(model.constraints))
```

Note that you can also print individual expressions like this, e.g. `print(s.transform(expression))` which helps to zoom in on the curlpit.

If you want to know about the **variable domains** as well, to see whether something is wrong there, you can do so as follows:

```python
s = SolverLookup.get("ortools")  # or whatever solver you are using
ct = s.transform(model.constraints)
from cpmpy.transformations.get_variables import print_variables
print_variables(ct)
print(ct)
```

Printing the **objective** as the solver sees it requires you to look into the solver interface code of that solver. However, the following is a good first check that can already reveal potentially problematic things:

```python
s = SolverLookup.get("ortools")  # or whatever solver you are using
from cpmpy.transformations.flatten_model import flatten_objective
(obj_var, obj_expr) = flatten_objective(model.objective)
print(f"Optimizing {obj_var} subject to", s.transform(obj_expr))
``` 

### Automatically minimising the UNSAT program
If the above is unwieldy because your constraint problem is too large, then consider automatically reducing it to a 'Minimal Unsatisfiable Subset' (MUS).

This is now part of our standard tools, that you can use as follows:

```python
from cpmpy.tools import mus

x = boolvar(shape=3, name="x")
model = Model(
    x[0],
    x[0] | x[1],
    x[2].implies(x[1]),
    ~x[0],
    )

unsat_cons = mus(model.constraints)
```

With this smaller set of constraints, repeat the visual inspection steps above.

(Note that for an UNSAT problem there can be many MUSes, the `examples/advanced/` folder has the MARCO algorithm that can enumerate all MSS/MUSes.)

### Correcting an UNSAT program

As many MUSes (=conflicts) may exist in the problem, resolving one of them does not necessarily make the model satisfiable.

In order to find which constraints are to be corrected, you can use the `tools.mcs` tool which computes a 'Minimal Correction Subset' (MCS).
By removing these contraints (or altering them), the model will become satisfiable.

Note that a Minimal Correction Subset is the complement of a Maximal Satisfiable Subset (MSS).
MSSes can be calculated optimally using a Max-CSP (resp. Max-SAT) formuation.
By weighting each of the constraints, you can define some preferences on which constraints should be satisfied over others.

```python
from cpmpy.tools import mcs, mss
import cpmpy as cp

x = cp.boolvar(shape=3, name="x")
model = cp.Model(
    x[0],
    x[0] | x[1],
    x[2].implies(x[1]),
    ~x[0],
    )

sat_cons = mss(model.constraints) # x[0] or x[1], x[2] -> x[1], ~x[0]
cons_to_remove = (mcs(model.constraints)) # x[0]
```

## Debugging a satisfiable model, that does not contain an expected solution

We will ignore the (possible) objective function here and focus on the feasibility part. 
Actually, in case of an optimisation problem where you know a certain value is attainable, you can add `objective == known_value` as constraint and proceed similarly.

Add the solution that you know should be a feasible solution as a constraint:
`model.add( (x == 1) & (y == 2) & (z == 3) ) # yes, brackets around each!`

You now have an UNSAT program! That means you can follow the steps above to better understand and correct it.

## Debugging a satisfiable model, which returns an impossible solution

This one is most annoying... Double check the printing of the model for oddities, also visually inspect the flat model. Try enumerating all solutions and look for an unwanted pattern in the solutions. Try a different solver. 

Try generating an explanation sequence for the solution... this requires a satisfaction problem, so remove the objective function or add a constraint that constraints the objective function to the value attained by the impossible solution.

As to generating the explanation sequence, check out our advanced example on [stepwise OCUS explanations](https://github.com/CPMpy/cpmpy/blob/master/examples/advanced/ocus_explanations.py)

