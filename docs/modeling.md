# Modeling and solving with CPMpy

This page explains and demonstrates how to use CPMpy to model and solve combinatorial problems, so you can use it to solve for example routing, scheduling, assignment and other problems.

## Installation

Installation is available through the `pip` python package manager. This will also install and use `ortools` as default solver:

```commandline
pip install cpmpy
```

See [installation instructions](installation_instructions.html) for more details. 

## Using the library

To conveniently use CPMpy in your python project, include it as follows:
```python
from cpmpy import *
```
This will overload the built-in `any()`, `all()`, `min()`, `max()`, `sum()` functions, such that they create CPMpy expressions when used on decision variables (see below). This convenience comes at the cost of some overhead for all uses of these functions in your code.

You can also import it as a package, which does not overload the python built-ins:
```python
import cpmpy as cp
```
We will use the latter in this document.

## Decision variables

Constraint modeling consists of expressing _constraints_ on _decision variables_, after which a solver will find a satisfying _assignment_ to these decision variables.

CPMpy supports discrete decision variables, namely Boolean and integer decision variables:

```python
import cpmpy as cp

b = cp.boolvar(name="b")
x = cp.intvar(1,10, name="x")
```

Decision variables have a **domain**, a set of allowed values. For Boolean variables this is implicitly the values 'False' and 'True'. For Integer decision variables, you have to specify the lower-bound and upper-bound (`1` and `10` respectively above).

Decision variables have a **unique name**. You can set it yourself, otherwise a unique name will automatically be assigned to it. If you print `print(b, x)` decision variables, it will print the name. Did we already say the name must be unique? Many solvers use the name as unique identifier, and it is near-impossible to debug with non-uniquely named variables.

A solver will set the **value** of the decision variables for which it solved, if it can find a solution. You can retrieve it with `v.value()`. Variables are not tied to a solver, so you can use the same variable in multiple models and solvers. When a solve call finishes, it will overwrite the value of all its decision variables. 

Finally, by providing a **shape** you automatically create a **numpy n-dimensional array** of decision variables. They automatically get their index appended to their name to ensure it is unique:

```python
import cpmpy as cp

b = cp.boolvar(shape=4, name="b")
print(b)  # [b[0] b[1] b[2] b[3]]

x = cp.intvar(1,10, shape=(2,2), name="x")
print(x)  # [[x[0,0] x[0,1]]
          #  [x[1,0] x[1,1]]]
```
You can also call `v.value()` on these n-dimensional arrays, which will return an n-dimensional **numpy** array of values. And you can do vectorized operations and comparisons, like in regular numpy. As we will see below, this is very convenient and avoids having to write out many loops. It also makes it compatible with many existing scientific python tools, including machine learning and visualisation libraries, so a lot less glue code to write.

See [the API documentation on variables](api/expressions/variables.html) for more detailed information.

## Creating a model

A **model** is a collection of constraints over decision variables, optionally with an objective function. It represents a problem for which a solution must be found, e.g. by a solver. A solution is an assignment to the decision variables, such that each of the constraints is satisfied.

In CPMpy, the `Model()` object is a simple container that stores a list of CPMpy expressions representing constraints. It can also store a CPMpy expression representing an objective function that must be minimized or maximized. Constraints are added in the constructor, or using the built-in `+=` addition operator that corresponds to calling the `__add__()` function.

Here is an example, where we explain how to express constraints in the next section:

```python
import cpmpy as cp

# Decision variables
(x,y,z) = cp.intvar(1,10, shape=3)  # Python unpacks the array into the individual variables

# Initialise the model, here with 2 constraints
m = cp.Model(
   x == 1,
   x + y > 5
)

# Adding more constraints
m += (y - z != x)
m += (x + y + z <= 15)
# you can also add a list of constraints, which is interpreted as a conjunction of constraints
m += [v <= 9 for v in [x,y,z]]

print(f"The model contains {len(m.constraints)} constraints")
print(m)  # pretty printing of the model, very useful for debugging
```
The `Model()` object has a number of other helpful functions, such as `to_file()` to store the model and `copy()` for creating a copy.

## Expressing constraints

A constraint is a relation between decision variables that restricts what values these variables can take.

We now review the different types of constraints in CPMpy.

### Logical constraints

To express **conjunction, disjunction and negation** of a constraint, we overwite the Python bitwise operators: `&` for conjunction (read as 'and'), `|` for disjunction (read as 'or') and `~` for negation (read as 'not').  

Some examples:

```python
import cpmpy as cp

# Decision variables
(a,b,c) = cp.boolvar(shape=3)

m = cp.Model(
   a | b,
   ~(a & c),
   (b | c) & ~a
)
```

Unfortunately, we can not overwite the `and`, `or` and `not` expression that we typically use in `if` expressions, so remember to use `&`,`|`,`~` instead. Also unfortunate is that Python bitwise operators have precedence over all other operators, so `a == 0 | b == 1` is **wrongly** interpreted by Python as `a == (0 | b) == 1` instead of the `(a == 0) | (b == 1)` that you probably intend. So make sure to **always write explicit brackets** when using `&`,`|`,`~`!

For **n-ary** conjunctions and disjunctions we overloaded the `all()` and `any()` functions:

```python
import cpmpy as cp

# Decision variables
bv = cp.boolvar(shape=3)

m = cp.Model(
    any([bv[0], bv[1], bv[2]]),
    any(v for v in bv),  # equivalent to above
    any(bv),  # equivalent to above
    ~all(bv)
)
```

Note how these functions accept manually created arrays, iterators or n-dimensional arrays alike.

For **equivalence**, also called reification, we overload the `==` comparison:
```python
import cpmpy as cp

# Decision variables
a,b,c = cp.boolvar(shape=3)

m = cp.Model(
    a == b,  # equivalence: a -> b & b -> a
    a != b   # same as ~(a==b) and a == ~b
)
```

Finally for **implication** we decided that it would be most readable to introduce a function `implies()` to our (Boolean) expression objects, e.g.:
```python
import cpmpy as cp

# Decision variables
a,b,c = cp.boolvar(shape=3)

m = cp.Model(
    a.implies(b),
    b.implies(a),
    a.implies(~c),
    (~c).implies(a)
)
```
For reverse implication, you switch the arguments yourself; it is difficult to read reverse implications out loud anyway. And as before, always use brackets around subexpressions to avoid surprises!



### Simple comparison constraints

We overload Pythons comparison operators: `==, !=, <, <=, >, >=`. Comparisons are allowed between any CPMpy expressions as well as Boolean and integer constants.

On a technical note, we treat Booleans as a subclass of integer expressions. This means that a Boolean (output) expression can be used anywhere a numeric expression can be used. But the inverse is not true: integers can NOT be used with Boolean operators:

```python
import cpmpy as cp

bv = cp.boolvar()
iv = cp.intvar(0,10)

m = cp.Model(
    bv == True,         # allowed
    bv > 0,             # allowed but silly
    iv > 3,             # allowed
    iv != 6,            # allowed
    iv == True,         # allowed but avoid, means `iv == 1`
    iv == bv,           # allowed but avoid, means `(iv == 1) == bv`
    # bv & iv,          # not allowed, choose one of:
    bv & (iv == 1),     # allowed
    bv & (iv != 0),     # allowed
)
```

CPMpy's array of decision variables is numpy-compatible, so it accepts **vectorized** operations on arrays of expressions:

```python
import cpmpy as cp

iv = cp.intvar(0, 10, shape=3)

m = cp.Model(
    iv == 1,  # a vectorized operation, equivalent to:
    [iv[0] == 1, iv[1] == 1, iv[2] == 1]
)
```
You can convert a pure Python list of expressions into a numpy-compatible array by using `cpm_array()`:
```python
import cpmpy as cp

x,y,z = cp.intvar(0, 10, shape=3)

m = cp.Model(
    # [x,y,z] == 1,  # does not work on plain Python arrays
    cp.cpm_array([x,y,z]) == 1  # does work, vectorized
)
```


### Arithmetic constraints

We overload Python's built-in arithmetic operators `+,-,*,//,%`. These can be used to built arbitrarily nested numeric expressions, which can then be turned into a constraint by adding a comparison to it.

We also overwrite the built-in functions `abs(),sum(),min(),max()` which can be used to created numeric expressions. Some examples:

```python
import cpmpy as cp

xs = cp.intvar(0, 10, shape=3, name="xs")
ys = cp.intvar(1, 10, shape=3, name="ys")

m = cp.Model(
    xs[0] - ys[0] == 5,
    cp.sum(xs) != 1,
    3*xs[0] < cp.abs(5 - cp.max(xs) + cp.min(ys))
)
```

All these operations can also be performed **vectorized** on arrays of the same shape, like in typical numpy code:
```python
import cpmpy as cp
import numpy as np

xs = cp.intvar(0, 10, shape=3, name="xs")
w = np.array([1,3,-5])

m = cp.Model( 
    cp.sum(w*xs) > 3,  # 1*xs[0] + 3*xs[1] + (-5)*xs[2] > 3
    # wsum(w,xs) > 3,  # XXX, I think we should add this fast-path...
    xs + w != 0,  # [xs[0] + 1 != 0, xs[1] + 3 != 0, xs[2] + (-5) != 0]
    cp.max(xs - w) == np.arange(3),  # max(xs[0] - 1) == 0, max(xs[1] - 3) == 1, max(xs[2] + 5) == 2]
)
```

Note that because of our overloading of `+,-,*,//` some numpy functions like `np.sum(some_array)` or `some_array.sum()` will also create a CPMpy expression when used on CPMpy decision variables. However, this is not guaranteed, and other functions like `np.max(some_array)` will not. To **avoid surprises**, you should hence always take care to call the CPMpy functions.

-- XXX I was wrong, we should overload NDVarArray.min and max

### Global constraints

You may wonder if you are allowed to use functions like `abs(),min(),max()` because some solvers might not have support for it? The answer is _yes you can use them_, because they are **global constraints**. 

In constraint solving, a global constraint is a function that expresses a relation between decision variables. There are **two pathways when solving** a model with global constraints: 1) the solver natively supports them, or 2) the constraint modelling library automatically _decomposes_ the constraint into an equivalent set of simpler constraints.

A good example is the `AllDifferent()` global constraint that ensures all its arguments have distinct values. `AllDifferent(x,y,z)` can be decomposed into `[x!=y, x!=z, y!=z]`. In general, the decomposition consists of _n*(n-1)_ pairwise inequalities, which are simpler constraints that most solvers support.

However, a solver that has specialised datastructures for this constraint specifically does not need to create the decomposition. Furthermore, for AllDifferent solvers can implement specialised algorithms that can propagate strictly stronger than the decomposed constraints can.



#### Global constraints

A non-exhaustive list of global constraints that are available in CPMpy is: `Xor(), AllDifferent(), AllDifferentExcept0(), Table(), Circuit(), Cumulative(), GlobalCardinalityCount()`.   

For their meaning and more information on how to define your own global constraints, see [the API documentation on global constraints](api/expressions/globalconstraints.html). Global constraints can also be reified (e.g. used in an implication or equality constraint). 

CPMpy will automatically decompose them if needed. If you want to see the decomposition yourself, you can call the `decompose()` function on them.

```python
import cpmpy as cp
x = cp.intvar(1,4, shape=4, name="x")
b = cp.boolvar()
cp.Model(
    cp.AllDifferent(x),
    cp.AllDifferent(x).decompose(),  # equivalent: [(x[0]) != (x[1]), (x[0]) != (x[2]), ...
    b.implies(cp.AllDifferent(x)),
    cp.Xor(b, cp.AllDifferent(x)),  # etc...
)
```
-- XXX smth on non-equivalent decompositions? see ongoing thread...

#### Numeric global constraints

Coming back to the Python-builtin functions `min(),max(),abs()`, these are a bit special because they have a numeric return type. In fact, constraint solvers typically implement a global constraint `MinimumEq(args, var)` that represents `min(args) == var`, so it combines a numeric function with a comparison, where it will ensure that the bounds of the expressions on both sides satisfy the comparison relation.

However, CPMpy also wishes to support the expressions `min(xs) > v` as well as `v + min(xs) != 4` and other nested expressions.

In CPMpy we do this by instantiating min/max/abs as **numeric global constraints**. E.g. `min([x,y,z])` becomes `Minimum([x,y,z])` which inherits from `GlobalConstraint` but has a numeric return type. Our library will transform the constraint model, including arbitrarly nested expressions, such that the numeric global constraint is used in a comparison with a variable. Then, the solver will either support it, or we will call `decompose_comparison()` on the numeric global constraint, which will decompose e.g. `min(xs) == v`.

A non-exhaustive list of **numeric global constraints** that are available in CPMpy is: `Minimum(), Maximum(), Count(), Element()`.   

For their meaning and more information on how to define your own global constraints, see [the API documentation on global constraints](api/expressions/globalconstraints.html).

```python
import cpmpy as cp
x = cp.intvar(1,4, shape=4, name="x")
s = cp.SolverLookup.get("ortools")
print(s.transform(cp.min(x) + cp.max(x) - 5 > 2*cp.Count(x, 2)))
# [(sum([IV5, IV6, -5])) > (IV4),
#  (min(x[0],x[1],x[2],x[3])) == (IV5), (max(x[0],x[1],x[2],x[3])) == (IV6),
#  (sum([2] * [IV3])) == (IV4),
#  (sum([BV0, BV1, BV2, BV3])) == (IV3),
#  (~BV0) -> (x[0] != 2), (BV0) -> (x[0] == 2),
#  (~BV1) -> (x[1] != 2), (BV1) -> (x[1] == 2),
#  (~BV2) -> (x[2] != 2), (BV2) -> (x[2] == 2),
#  (~BV3) -> (x[3] != 2), (BV3) -> (x[3] == 2)]

```

#### the Element numeric global constraint

The `Element(Arr,Idx)` global constraint enforces that the result equals `Arr[Idx]` with `Arr` an array of constants or variables (the first argument) and `Idx` an integer decision variable, representing the index into the array.

```python
import cpmpy as cp

arr = cp.intvar(1,10, shape=4)
idx = cp.intvar(0,len(arr))  # indexing is offset 0

m = cp.Model(
    cp.AllDifferent(arr),
    arr[idx] == 2
)
m.solve()
print(f"arr: {arr.value()}, idx: {idx.value()}, val: {arr[idx].value()}")
# example output -- arr: [2 1 3 4], idx: 0, val: 2
```

The `arr[idx]` works because `arr` is a CPMpy `NDVarArray()` and we overloaded the `__getitem__()` python function.

This does not work on NumPy arrays though, as they don't know CPMpy. So you have to **wrapped the array** in our `cpm_array()` or call `Element()` directly:

```python
import numpy as np
import cpmpy as cp

arr = np.arange(4)  # array([0, 1, 2, 3])
idx = cp.intvar(0,len(arr))  # indexing is offset 0

m = cp.Model()
#m += (arr[idx] == 2)             # does not work, numpy does not know what to do
# IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

cparr = cp.cpm_array(arr)         # wrap in CPMpy array
m += (cparr[idx] == 2)            # works

m += (cp.Element(arr, idx) == 2)  # also works, identical to above

m.solve()
print(f"arr: {arr.value()}, idx: {idx.value()}, val: {arr[idx].value()}")
# arr: [0 1 2 3], idx: 2, val: 2
```

         


## Objective functions

If a model has no objective function specified, then it is a satisfaction problem: the goal is to find out whether a solution, any solution, exists. When an objective function is added, this function needs to be minimized or maximized.

Any CPMpy expression can be added as objective function. Solvers are especially good in optimizing linear functions or the minimum/maximum of a set of expressions. Other (non-linear) expressions are supported too, just give it a try.

```python
import cpmpy as cp
m = cp.Model()

# Variables
b = cp.boolvar(name="b")
x = cp.intvar(1,10, shape=3, name="x")

# Constraints
m += (x[0] == 1)
m += cp.AllDifferent(x)
m += b.implies(x[1] + x[2] > 5)

# Objective function (optional)
m.maximize(cp.sum(x) + 100*b)

print(m)
if m.solve():
    print(x.value(), b.value())
else:
    print("No solution found.")
```

## Solving a model

CPMpy can be used as a declarative modeling language: you create a `Model()`, add constraints and call `solve()` on it. See the example above.

The return value of `solve()` is a Boolean indicating whether a solution was found. So regardless of whether it was a satisfaction or optimisation problem or with a timeout, it returns true if 'a' solution has been found in the process.

To know the exact solver state and runtime after solve, call `status()`. In case of an optimisation problem, you can get the objective value of the solution with `objective_value()`.

```python
import cpmpy as cp
xs = cp.intvar(1,10, shape=3)
m = cp.Model(cp.AllDifferent(xs), maximize=cp.sum(xs))

hassol = m.solve()
print("Status:", m.status())  # Status: ExitStatus.OPTIMAL (0.03033301 seconds)
if hassol:
    print(m.objective_value(), xs.value())  # 27 [10  9  8]
else:
    print("No solution found.")
```


## Finding all solutions

You can also conveniently use CPMpy to find all solutions using the `solveAll()` function:

```python
import cpmpy as cp
x = cp.intvar(0, 3, shape=2)
m = cp.Model(x[0] > x[1])

n = m.solveAll()
print("Nr of solutions:", n)  # Nr of solutions: 6
```

When using `solveAll()`, a solver will use an optimized native implementation behind the scenes when that exists.

It has a `display=...` argument that can be used to display expressions or as a callback, as well as the `solution_limit=...` argument to set a solution limit. It also accepts any named argument, like `time_limit=...`, that the underlying solver accepts.
```python
n = m.solveAll(display=[x,cp.sum(x)], solution_limit=3)
# [array([1, 0]), 1]
# [array([2, 0]), 2]
# [array([3, 0]), 3]
```

There is much more to say on enumerating solutions and the use of callbacks or blocking clauses. See the [the detailed documentation on finding multiple solutions](multiple_solutions.html).

## Debugging a model

The best place to start is to **print** the model you have created, or the individual constraints. If that looks fine (e.g. no integers, or shorter or longer expressions then what you intended), then have a look at what the constraint(s) look like after transformation:

```python
import cpmpy as cp

cons = []  # ... imagine a list of constraints
print(cons)

m = cp.Model(cons)  # any model created
print(m)

s = cp.SolverLookup.get("ortools")
print(s.transform(m.constraints))  # print the list of constraints after solver-specific transformations
```

If that is not sufficient, have a look at our detailed [Debugging guide](how_to_debug.md).

## Selecting a solver

The default solver is ortools CP-SAT, an award winning constraint solver. But CPMpy supports multiple other solvers: a MIP solver (gurobi), SAT solvers (those in PySAT), the Z3 SMT solver, even a knowledge compiler (PySDD) and any CP solver supported by the text-based MiniZinc language.

See the full list of solvers known by CPMpy with:

```python
import cpmpy as cp
cp.SolverLookup.solvernames()
```

On my system, with pysat and minizinc installed, this gives `['ortools', 'minizinc', 'minizinc:chuffed', 'minizinc:coin-bc', ..., 'pysat:minicard', 'pysat:minisat22', 'pysat:minisat-gh']

You can specify a solvername when calling `solve()` on a model:

```python
import cpmpy as cp
x = cp.intvar(0,10, shape=3)
m = cp.Model(cp.sum(x) <= 5)
# use named solver
m.solve(solver="ortools")
```

Note that for other solvers, you will need to **install additional package(s)**. You can check if a solver, e.g. "gurobi", is supported by calling `cp.SolverLookup.get("gurobi")` and it will raise a helpful error if it is not yet installed on your system. See [the API documentation](api/solvers.html) of the solver for detailed installation instructions.

## Model versus solver interface

A `Model()` is a **lazy container**. It simply stores the constraints. Only when `solve()` is called will it instantiate a solver, and send the entire model to it at once. So `m.solve("ortools")` is equivalent to:
```python
s = SolverLookup.get("ortools", m)
s.solve()
```


Solver interfaces allow more than the generic model interface, because, well, they can support solver-specific features. Such as solver-specific parameters, passing a previous solution to start from, incremental solving, unsat core extraction, solver-specific callbacks etc.

Importantly, the solver interface supports the same functions as the `Model()` object (for adding constraints, an objective, solve, solveAll, status, ...). So if you want to make use of some features of a solver, simply replace `m = Model()` by `m = SolverLookup.get("your-preferred-solvername")` and your code remains valid. Below, we replace `m` by `s` for readability.

```python
import cpmpy as cp
x = cp.intvar(0,10, shape=3) 
s = cp.SolverLookup.get("ortools")
# we are operating on the ortools interface here
s += (cp.sum(x) <= 5)
s.solve()
print(s.status())
```

On a technical note, observe that a solver object does not modify the Model object with which it is initialised. So adding constraints to the solver does not add them to that model, and calling `s.solve()` does not update the status of `m.status()`, only `m.solve()` does that.

## Setting solver parameters

Now lets use our solver-specific powers. 
For example, with `m` a CPMpy Model(), you can do the following to make or-tools use 8 parallel cores and print search progress:

```python
import cpmpy as cp
s = cp.SolverLookup.get("ortools", m)
# we are operating on the ortools interface here
s.solve(num_search_workers=8, log_search_progress=True)
```

Modern CP-solvers support a variety of hyperparameters. ([OR-tools parameters](https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto) for example).
Using the solver interface, any solver parameter can be passed using the `.solve()` call.
These parameters will then be posted to the native solver object before solving the model.

```python
s.solve(cp_model_probing_level = 2,
        linearization_level = 0,
        symmetry_level = 1)
```

See [the API documentation of the solvers](api/solvers.html) for information and links on the parameters supported. See our documentation page on [solver parameters](solver_parameters.html) if you want to tune your hyperparameters automatically. 



You can use any of these solvers by passing its name to the `Model.solve()` parameter 'solver' as such:

```python
a,b = boolvar(2)
Model(a|b).solve(solver='minizinc:chuffed')
```


## Incremental solving
It is important to realize that a CPMpy solver interface is _eager_. That means that when a CPMpy constraint is added to a solver object, CPMpy _immediately_ translates it and posts the constraints to the underlying solver.

This has two potential benefits for incremental solving, whereby you add more constraints and variables inbetween solve calls:

  1) CPMpy only translates and posts each constraint once, even if the model is solved multiple times; and 
  2) if the solver itself is incremental then it can reuse any information from call to call, as the state of the native solver object is kept between solver calls and can therefore rely on information derived during a previous `solve` call.

```python
gs = SolverLookup.get("gurobi")

gs += sum(ivar) <= 5 
gs.solve()

gs += sum(ivar) == 3
# the underlying gurobi instance is reused, only the new constraint is added to it.
# gurobi is an incremental solver and will look for solutions starting from the previous one.
gs.solve()
```
 
_Technical note_: ortools its model representation is incremental but its solving itself is not (yet?). Gurobi and the PySAT solvers are fully incremental, as is Z3. The text-based MiniZinc language is not incremental.

## Using solver-specific CPMpy functions

We sometimes add solver-specific features to the CPMpy interface, for convenient access. Two examples of this are `solution_hint()` and `get_core()` which is supported by the OrTools and PySAT solvers and interfaces. Other solvers may work differently and not have these concepts.

`solution_hint()` tells the solver that it could use these variable-values first during search, e.g. typically from a previous solution:
```python
from cpmpy import *
x = intvar(0,10, shape=3)
s = SolverLookup.get("ortools")
s += sum(x) <= 5
# we are operating on a ortools' interface here
s.solution_hint(x, [1,2,3])
s.solve()
print(x.value())
```

`get_core()` asks the solver for an unsatisfiable core, in case a solution did not exist and assumption variables were used. See the documentation on [Unsat core extraction](unsat_core_extraction.html).

See [the API documentation of the solvers](api/solvers.html) to learn about their special functions.


## Direct solver access
Some solvers implement more constraints then available in CPMpy. But CPMpy offers direct access to the underlying solver, so there are two ways to post such solver-specific constraints.

### DirectConstraint
The `DirectConstraint` will directly call a function of the underlying solver when added to a CPMpy solver. 

You provide it with the name of the function you want to call, as well as the arguments:

```python
from cpmpy import *
iv = intvar(1,9, shape=3)

s = SolverLookup.get("ortools")

s += AllDifferent(iv)
s += DirectConstraint("AddAllDifferent", iv)  # a DirectConstraint equivalent to the above for OrTools
```

This requires knowledge of the API of the underlying solver, as any function name that you give to it will be called. The only special thing that the DirectConstraint does, is automatically translate any CPMpy variable in the argument to the native solver variable.

Note that any argument given will be checked for whether it needs to be mapped to a native solver variable. This may give errors on complex arguments, or be inefficient. You can tell the `DirectConstraint` not to scan for variables with `noarg` argument, for example:

```python
from cpmpy import *
trans_vars = boolvar(shape=4, name="trans")

s = SolverLookup.get("ortools")

trans_tabl = [ # corresponds to regex 0* 1+ 0+
    (0, 0, 0),
    (0, 1, 1),
    (1, 1, 1),
    (1, 0, 2),
    (2, 0, 2)
]
s += DirectConstraint("AddAutomaton", (trans_vars, 0, [2], trans_tabl),
                      novar=[1, 2, 3])  # optional, what not to scan for vars
```

A minimal example of the DirectConstraint for every supported solver is [in the test suite](https://github.com/CPMpy/cpmpy/tree/master/tests/test_direct.py).

The `DirectConstraint` is a very powerful primitive to get the most out of specific solvers. See the following examples: [nonogram_ortools.ipynb](https://github.com/CPMpy/cpmpy/tree/master/examples/nonogram_ortools.ipynb) using of a helper function that generates automatons with DirectConstraints; [vrp_ortools.py](https://github.com/CPMpy/cpmpy/tree/master/examples/vrp_ortools.ipynb) demonstrating ortools' newly introduced multi-circuit global constraint through DirectConstraint; and [pctsp_ortools.py](https://github.com/CPMpy/cpmpy/tree/master/examples/pctsp_ortools.ipynb) that uses a DirectConstraint to use ortools circuit to post a sub-circuit constraint as needed for this price-collecting TSP variant.

### Directly accessing the underlying solver

The `DirectConstraint("AddAllDifferent", iv)` is equivalent to the following code, which demonstrates that you can mix the use of CPMpy with calling the underlying solver directly: 

```python
from cpmpy import *
iv = intvar(1,9, shape=3)

s = SolverLookup.get("ortools")

s += AllDifferent(iv)  # the traditional way, equivalent to:
s.ort_model.AddAllDifferent(s.solver_vars(iv))  # directly calling the API, has to be with native variables
```

observe how we first map the CPMpy variables to native variables by calling `s.solver_vars()`, and then give these to the native solver API directly.  This is in fact what happens behind the scenes when posting a DirectConstraint, or any CPMpy constraint.

While directly calling the solver offers a lot of freedom, it is a bit more cumbersome as you have to map the variables manually each time. Also, you no longer have a declarative model that you can pass along, print or inspect. In contrast, a `DirectConstraint` is a CPMpy expression so its use is identical to other constraints.

## Hyperparameter search across different parameters
Because CPMpy offers programmatic access to the solver API, hyperparameter search can be straightforwardly done with little overhead between the calls.

The tools directory contains a utility to efficiently search through the hyperparameter space defined by the solvers `tunable_params`.
This utlity is based on the SMBO framework and speeds up the search by implementing adaptive capping.

The parameter tuner is based on the following publication: 
>Ignace Bleukx, Senne Berden, Lize Coenen, Nicholas Decleyre, Tias Guns (2022). Model-Based Algorithm
>Configuration with Adaptive Capping and Prior Distributions. In: Schaus, P. (eds) Integration of Constraint
>Programming, Artificial Intelligence, and Operations Research. CPAIOR 2022. Lecture Notes in Computer Science,
>vol 13292. Springer, Cham. https://doi.org/10.1007/978-3-031-08011-1_6

Solver interfaces not providing the set of tunable parameters can still be tuned by using this utility and providing the parameter (values) yourself.

```python
from cpmpy import *
from cpmpy.tools import ParameterTuner

model = Model(...)

tunables = {
    "search_branching":[0,1,2],
    "linearization_level":[0,1],
    'symmetry_level': [0,1,2]}
defaults = {
    "search_branching": 0,
    "linearization_level": 1,
    'symmetry_level': 2
}

tuner = ParameterTuner("ortools", model, tunables, defaults)
best_params = tuner.tune(max_tries=100)
best_runtime = tuner.best_runtime
```
