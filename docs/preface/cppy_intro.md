## Constraint Programming

Many real-life decisions involve a large number of options. To decide if a problem is feasible or finding the best one amongst all the options is hard task to do by hand. In other words, to enumerate all the possible combinations of single decisions and evaluate them is infeasible in practice. To avoid this "*brute force*" approach, the paradigm of **constraint programming (CP)** allow us to:

1. Model relationships between single decisions smartly
2. Give an answer efficiently.

A **constraint satisfaction problem (CSP)** consists of a set of variables and constraints stablishing relationships between them. Each variable has a finite of possible values (its domain). The goal is to assign values to the variables in its domains satisfying all the constraints. A more general version, called **constraint optimization programming (C0P)**, finds amongst all the feasible solutions the one that optimizes some measure, called 'objective function'.

### What is necessary to model a CP?

A typical CP is defined by the following elements:

**Variables**: Variables represents the decisions to be made. Depending on the decisions to be made variables can be *Boolean*, whenever a Yes or No decision is needed to be made, or *Integer*, whenever an integer number is necessary to represent a decision. In the first case, we say the **domain** of a Boolean variable is the set {True, False}. For integer variables we represent this as an interval of integer numbers, {a,b}.

**Constraints**: Constraints are all the conditions that variables must satisfy. A set of values of the variables satisfying all the constraints is named a *feasible* solution.

Moreover, if we want to model an optimization problem we also need an objective function.

### Example

A cryptarithmetic puzzle is a mathematical exercise where the digits of some numbers are represented by letters (or symbols). Each letter represents a unique digit. The goal is to find the digits such that a given mathematical equation is verified. 

For example, we aim to allocate to the letters S,E,N,D,M,O,R,Y a digit between 0 and 9, being all the letters allocated to a different digit and such that the expression: 

SEND + MORE = MONEY

is satisfied. This problem lies into the setting of **constraint satisfaction problem (CSP)**. Here the variables are each letter S,E,N,D,M,O,R,Y and their domain is {0,1,2,...,9}. The constraints represents the fact that


The cpmpy implementation for this CSP looks like:

```python
from cpmpy import *
import numpy as np

# Construct the model
s,e,n,d,m,o,r,y = IntVar(0,9, 8)

constraint = []
constraint += [ alldifferent([s,e,n,d,m,o,r,y]) ]
constraint += [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                 + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
                == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) ) ]

model = Model(constraint)
print(model)

result = model.solve()
print("  S,E,N,D =   ", [x.value() for x in [s,e,n,d]])
print("  M,O,R,E =   ", [x.value() for x in [m,o,r,e]])
print("M,O,N,E,Y =", [x.value() for x in [m,o,n,e,y]])
```


A possible feasible allocation/solution is 


```python
  S,E,N,D =   [2, 8, 1, 7]
  M,O,R,E =   [0, 3, 6, 8]
  M,O,N,E,Y = [0, 3, 1, 8, 5]
```

Note that we can find an slightly different version of this problem by optimizing an objective function, for example, optimizing the number formed by the word MONEY:

<img src="https://render.githubusercontent.com/render/math?math=\max%20\quad10000%20M%20%2B%201000%20O%20%2B%20100%20N%20%2B%2010%20E%20%2B%201%20Y">


To implement this COP, we need only to modify the Model statement by adding an objective function:

```python
coefs  = np.flip(10**np.arange(5))
objective = np.dot([m,o,n,e,y],coefs)
model = Model(constraint, maximize = objective)
```
And the result will be:
```python
  S,E,N,D =   [9, 5, 6, 7]
  M,O,R,E =   [1, 0, 8, 5]
  M,O,N,E,Y = [1, 0, 6, 5, 2]
```

In this [Jupyter notebook](https://github.com/tias/cppy/blob/master/docs/examples/explaining_smm.ipynb), we will look in detail this example. But first you may want to look some references for a global overview of Constraint Programming.


### References


<!---Add more references -->

To learn more about theory and practice of constraint programming you may want to check some references:

1. Rossi, F., Van Beek, P., & Walsh, T. (Eds.). (2006). Handbook of constraint programming. Elsevier.
2. Apt, K. (2003). Principles of constraint programming. Cambridge university press.
