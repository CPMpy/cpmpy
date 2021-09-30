# How to debug

You get an error, or no error, but also no (correct) solution... Annoying, you have a bug.

The bug can be situated in one of three layers:
- your problem specification
- the CPMpy library
- the solver

coincidentally, they are ordered from most likely to least likely. So let's start at the bottom.

## Debugging the solver

If you get an error, try searching on the internet if other users have had the same.

If you don't find it, or if the solver runs fine and without error, but you don't get the answer you expect; then try swapping out the solver for another solver and see what gives...

Replace `model.solve()` by `model.solve(solver='minizinc')` for example. You do need to install MiniZinc and minizinc-python first though.

Either you have the same output, and it is not the solver's fault, or you have a different output and you actually found one of these rare solver bugs. Report on the bugtracker of the solver, or on the CPMpy github page where we will help you file a bug 'upstream' (or maybe even work around it in CPMpy).

## Debugging a modeling error

You get an error when you create an expression? Then you are probably writing it wrongly. Check the documentation and the running examples for similar examples of what you wish to express.

Here are a few quirks in Python/CPMpy:
  - when using `&` and `|`, make sure to always put the subexpressions in brackets. E.g. `(x == 1) & (y == 0)` instead of `x == 1 & y == 0`. The latter wont work, because Python will unfortunately think you meant `x == (1 & (y == 0))`.
  - you can write `vars[other_var]` but you can't write `non_var_list[a_var]`. That is because the `vars` list knows CPMpy, and the `non_var_list` does not. Wrap it: `non_var_list = cpm_array(non_var_list)` first, or write `Element(non_var_list, a_var)` instead.

Try printing the expression `print(e)` or subexpressions, and check that the output matches what you wish to express. Decompose the expression and try printing the individual components and their piecewice composition to see what works and when it starts to break.

If you don't find it, report it on the CPMpy github Issues page and we'll help you (and maybe even extend the above list of quirks).

## Debugging a `solve()` error

You get an error either from CPMpy (e.g. the flattening, or the solver interface) or the solver itself is saying the model is invalid. This may be because you have modelled something impossible, or because you have a corner case that CPMpy does not yet capture.

The two are hard to tell apart. We want to catch corner cases in CPMpy, even if it is a solver limitation, so please report on the CPMpy github Issues page.

Or maybe, you got one of CPMpy's NotImplementedErrors. Share your use case with us and we will implement it. Or implemented it yourself first, that is also very welcome ; )

## Debugging an UNSATisfiable model

First, print the model:

`print(model)` and check that the output matches what you want to express. Do you see anything unusual? Start there, see why the expression is now what you intended to express, as described in 'Debugging a modeling error'.

If that does not help, try printing the 'flat normal form' of the model, which also shows the intermediate variables that are automatically created by CPMpy:

```python
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_model
print(flatten_model(model))
```

Note that you can also print individual flattened expressions with `print(flatten_constraint(expression))` which helps to zoom in on the curlpit.

If you want to know about the variable domains as well, to see whether something is wrong there, you can do so as follows:

```python
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_model
from cpmpy.transformations.get_variables import print_variables
mf = flatten_model(model)
print_variables(mf)
print(mf)`
```

### Automatically minimising the UNSAT program
If the above is unwieldy because your constraint problem is too large, then consider automatically reducing it to its 'UNSAT core'. You can use 'MUSX' in examples/advanced/ for this:

```python
from examples.advanced.musx import musx

x,y,z = boolvar(3)
model = Model(
    x,
    ~x,
    x|y,
    z.implies(x)
    )

unsat_cons = musx(model.constraints)
model2 = Model(unsat_cons)
```


With this smaller program, repeat the visual inspection steps above.


## Debugging a satisfiable model, that does not contain an expected solution

We will ignore the (possible) objective function here and focus on the feasibility part. Actualy, in case of an optimisation problem where you know a certain value is attainable, you can add `objective == known_value` as constraint and proceed similarly.

Add the solution that you know should be a feasible solution as a constraint:
`model.add( (x == 1) & (y == 2) & (z == 3) ) # yes, brackets around each!`

You now have an UNSAT program! That means you can follow the steps in 'Automatically minimising the UNSAT program' above to better understand it.

## Debugging a satisfiable model, which returns an impossible solution

This one is most annoying... Double check the printing of the model for oddities, also visually inspect the flat model. Try enumerating all solutions and look for an unwanted pattern in the solutions. Try a different solver. 

Try generating an explanation sequence for the solution... this requires a satisfaction problem, so remove the objective function or add a constraint that constraints the objective function to the value attained by the impossible solution.

As to generating the explanation sequence, we will add this as a demo soon...


