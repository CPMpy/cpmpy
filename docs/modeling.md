# Modeling

CPMpy is a library for modeling and solving constrained satisfaction and optimisation problems (CSPs and COPs in the AI literature).

A constraint model consists of 3 key parts:

  * _Decision variables_ with their domain (allowed values)
  * _Constraints_ over decision variables
  * Optionally an _objective function_

```python
from cpmpy import *
m = Model()

# Variables
b = boolvar(name="b")
x = intvar(1,10, shape=3, name="x")

# Constraints
m += (x[0] == 1)
m += AllDifferent(x)
m += b.implies(x[1] + x[2] > 5)

# Objective function (optional)
m.maximize(sum(x) + 100*b)

print(m)
print(m.solve(), x.value(), b.value())
```

See <https://github.com/CPMpy/cpmpy/blob/master/examples/quickstart_sudoku.ipynb> for a more realistic step-by-step example.


## Variables

CPMpy supports discrete decision variables. All variables are numpy arrays, so creating 1 of them or an array/tensor of them is similar:

  * Boolean variables: `boolvar(shape=1, name=None)`
  * Integer variables: `intvar(lb, ub, shape=1, name=None)`

See [the API documentation on variables](api/expressions/variables.html) for more information.


## Constraints
Constraints have to be _added_ to a model, which is done with the `+=` operator, e.g. `model += constraint`. You can also add lists of constraints and other nested expressions.

Using Python's built-in comparison operators `==,!=,<,<=,>,>=` and logical operators `&,|,~,^` on CPMpy variables will automatically create constraint expressions that can be added to a model.

You can also use the built-in arithmetic operators `+,-,*,//,%` and we overwrite the built-in `abs,sum,min,max,all,any` functions so that you can use them in the construction of expressions.

CP languages like CPMpy also offer what is called **global constraints**. Convenient expressions that capture part of the problem structure and that the solvers can typically use efficiently too. A non-exhaustive list of global constraints is: `AllDifferent(), AllEqual(), Circuit(), Table(), Element()`.

See [the API documentation on expressions](api/expressions.html) for more information.


## Objective function

If a model has no objective function specified, then it is a satisfaction problem: the goal is to find out whether a solution, any solution, exists. When an objective function is added, this function needs to be minimized or maximized.

Any CPMpy expression can be added as objective function. Solvers are especially good in optimizing linear functions or the minimum/maximum of a set of expressions. Other (non-linear) expressions are supported too, just give it a try.


## Other uses of the `Model()` object

The `Model()` object has a number of other helpful functions, such as `status()` to print the status of the last `solve()` call, `solveAll()` to find all solutions, `to_file()` to store the model (you can print a model too, for debugging) and `copy` for creating a copy.

See [the API documentation on Model](api/model.html) for more information.
