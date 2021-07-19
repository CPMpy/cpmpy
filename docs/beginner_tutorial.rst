Getting started with Constraint Programming and CPMpy
=====================================================

Constraint Programming
----------------------

Many real-life decisions involve searching over a large number of possible solutions to find one that satisfies all constraints and/or optimizes an objective function. For example in timetabling, scheduling, packing, routing and many more.

To decide if a problem is feasible or finding the best one amongst all the options is hard task to do by hand. And enumerating all possible solutions and simply checking whether they are good (generate-and-test) is usually infeasible in practice.

Instead, the paradigm of **constraint programming (CP)** allow you to:

1. Model the space of possible solutions through *decision variables*
2. Model relations between variables through *constraints* and an *objective function*
3. Have a state-of-the-art solver compute the answer efficiently.

So despite the word 'Programming' in Constraint Programming (since forever), as a user you only have to focus on *modeling* the problem, not on programming the search. This is the convenience and appeal of Constraint Programming.

Satisfaction versus Optimisation
--------------------------------

A **constraint satisfaction problem (CSP)** consists of a set of variables and constraints stablishing relationships between them. Each variable has a finite of possible values (its domain). The goal is to assign values to the variables in its domains satisfying all the constraints. 

A more general version, called **constraint optimization programming (C0P)**, finds amongst all the feasible solutions the one that optimizes some measure, called 'objective function'.

The state-of-the-art CP solvers can perform both very efficiently, so it is up to you to decide wether you have a satisfaction or an optimisation problem.


What is necessary to model a CP problem?
----------------------------------------

A typical CP problem is defined by the following elements:

**Variables**: Variables represents the decisions to be made. Depending on the decisions to be made variables can be *Boolean*, whenever a Yes or No decision is needed to be made, or *Integer*, whenever an integer number is necessary to represent a decision. In the first case, we say the **domain** of a Boolean variable is the set {True, False}. For integer variables we represent this as an interval of integer numbers, [a,b].

**Constraints**: Constraints are all the conditions that variables must satisfy. A set of values of the variables satisfying all the constraints is named a *feasible* solution. In CP, constraints can be boolean expressions, arithmetic operations or [global constrains](https://github.com/tias/cppy/blob/master/docs/api/constraints.rst).

Moreover, if we want to model an constrained optimization problem we also need to specify an 

**Objective function:** This is a function of the set of variables returning a real number. This metric is *maximized* or *minimized* over the set of all feasible solutions. An *optimal solution* is the one that satisfies all the constrains and returns the biggest value of the objective function (the smallest in case of minimization).

Example: cryptarithemtic
------------------------

A cryptarithmetic puzzle is a mathematical challenge where the digits of some numbers are represented by letters (or symbols). Each letter represents a unique digit. The goal is to find the digits such that a given mathematical equation is verified. 

For example, we aim to allocate to the letters S,E,N,D,M,O,R,Y a digit between 0 and 9, being all the letters allocated to a different digit and such that the expression: 

SEND + MORE = MONEY

is satisfied. This problem lies into the setting of **constraint satisfaction problem (CSP)**. Here the variables are each letter S,E,N,D,M,O,R,Y and their domain is {0,1,2,...,9}. The constraints represents the fact that the values of the ltters need to sum up. And to be mathematically clean, the first letters can not be `0`.

Cryptarythmetic in CPMpy
------------------------

First we need to import all the tools that we will need to create our CP model, namely numpy and our CPMpy library:

```python
import numpy as np
from cpmpy import *
```

Secondly, as in every constraint programming model we need to define the decision variables:

```python
s,e,n,d,m,o,r,y = intvar(0,9, shape=8)
```

This line indicates that we are creating 8 integer decision variables, s,e,n,d,m,o,r,y, and each will take a value between 0 and 9 (inclusive) in the solution. The `shape` argument informs the shape of the tensor (in this case, a vector of size 8, unpacked over the individual letters).

Thirdly, the constraints. We will immediately wrap them in a `Model()` object:


Constraints are included in the model as a list. First, we create a list to add the constraints. Then, we append an 'all different constraint' in a straightforward fashion. Finally, we add the constraint saying SEND + MORE = MONEY. 

```python
model = Model(
    AllDifferent([s,e,n,d,m,o,r,y]),
    (    sum(   [s,e,n,d] * np.array([       1000, 100, 10, 1]) ) \
       + sum(   [m,o,r,e] * np.array([       1000, 100, 10, 1]) ) \
      == sum( [m,o,n,e,y] * np.array([10000, 1000, 100, 10, 1]) ) ),
    s > 0,
    m > 0,
)
```

The first line uses the `AllDifferent` global constraint. It is a CP primitive that will enforce that all variables get a different value. CP solvers have highly optimized procedures to enforce such constraints, hence the choice to model this with one `AllDifferent` global constraint rather then specifying that each pair of variables to have different values.

The second line (split over 3 lines) enforces the mathematical relation. Because CPMpy is based on the omnipresent numpy scientific library, you can perform products and other operators on combinations of CPMpy and NumPy arrays.

The last two lines enforce that the starting digits are not 0.

Solving a CPMpy model
---------------------

Solving a model is as easy as calling `.solve()` on it, which will automatically search for a solver installed on the system, and make it solve the model:

```python
model.solve()
```

The return value will be whether the model was satisfiable or not (True/False) in case of a satisfaction problem, and what the optimal value was in case of an optimisation problem.

The solution will be backpopulated in the decision variables used, and can be obtained by calling the `.value()` function on a decision variable. For example:

```python
if model.solve():
    print("  S,E,N,D =   ", [x.value() for x in [s,e,n,d]])
    print("  M,O,R,E =   ", [x.value() for x in [m,o,r,e]])
    print("M,O,N,E,Y =", [x.value() for x in [m,o,n,e,y]])
else:
    print("No solution found")
```

And that is all there is to it...

To get more familiar with these concepts, you can experiment with modeling and solving the sudoku puzzle problem in [the following notebook](https://github.com/CPMpy/cpmpy/blob/master/examples/quickstart_sudoku.ipynb).

And many more examples on scheduling, packing, routing and more in the [examples folder](https://github.com/CPMpy/cpmpy/blob/master/examples/).


### References


<!---Add more references -->

To learn more about theory and practice of constraint programming you may want to check some references:

1. Rossi, F., Van Beek, P., & Walsh, T. (Eds.). (2006). Handbook of constraint programming. Elsevier.
2. Apt, K. (2003). Principles of constraint programming. Cambridge university press.
