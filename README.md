## CpMPy: CP modeling made easy in Python

Welcome to CpMPy. Licensed under the MIT License.

CpMPy is a numpy-based light-weight Python library for conveniently modeling constraint problems in Python. It aims to connect to common constraint solving systems that have a Python API, such as MiniZinc (with solvers gecode, chuffed, ortools, picatsat, etc), or-tools through its Python API and more.

It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure.

A longer description of its motivation and architecture is in [this short paper](modref19_cppy.pdf).

The software is in ALPHA state, and more of a proof-of-concept really. Do send suggestions, additions, API changes, or even reuse some of these ideas in your own project!

Check the CP [tutorial](https://github.com/tias/cppy/blob/master/docs/overview.rst).

### Install the library

### Documentation

Get the full CpMPy [documentation](https://cpmpy.readthedocs.io/en/latest/). 

### Examples

The following examples show the elegance of building on Python/Numpy:
```python
from cppy import *
import numpy as np

# Construct the model
s,e,n,d,m,o,r,y = IntVar(0,9, 8)

constraint = []
constraint += [ alldifferent([s,e,n,d,m,o,r,y]) ]
constraint += [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                 + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
                == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) ) ]
constraint += [ s > 0, m > 0 ]

model = Model(constraint)
print(model)

stats = model.solve()
print("  S,E,N,D =  ", [x.value() for x in [s,e,n,d]])
print("  M,O,R,E =  ", [x.value() for x in [m,o,r,e]])
print("M,O,N,E,Y =", [x.value() for x in [m,o,n,e,y]])
```

sudoku and others need matrix indexing, which numpy supports extensively:
```python
from cppy import *
import numpy

x = 0 # cells whose value we seek
n = 9 # matrix size
given = numpy.array([
    [x, x, x,  2, x, 5,  x, x, x],
    [x, 9, x,  x, x, x,  7, 3, x],
    [x, x, 2,  x, x, 9,  x, 6, x],

    [2, x, x,  x, x, x,  4, x, 9],
    [x, x, x,  x, 7, x,  x, x, x],
    [6, x, 9,  x, x, x,  x, x, 1],

    [x, 8, x,  4, x, x,  1, x, x],
    [x, 6, 3,  x, x, x,  x, 8, x],
    [x, x, x,  6, x, 8,  x, x, x]])


# Variables
puzzle = IntVar(1, n, shape=given.shape)

constraint = []
# constraints on rows and columns
constraint += [ alldifferent(row) for row in puzzle ]
constraint += [ alldifferent(col) for col in puzzle.T ]

# constraint on blocks
for i in range(0,n,3):
    for j in range(0,n,3):
        constraint += [ alldifferent(puzzle[i:i+3, j:j+3]) ]

# constraints on values
constraint += [ puzzle[given>0] == given[given>0] ]

model = Model(constraint)
stats = model.solve()
```

and an OR problem for good faith:
```python
from cppy import *
import numpy

# data
demands = [8, 10, 7, 12, 4, 4]
slots = len(demands)

# variables
x = IntVar(0,sum(demands), slots)

constraint  = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
constraint += [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

objective = sum(x) # number of buses

model = Model(constraint, minimize=objective)
stats = model.solve()
```

See more [examples](https://github.com/tias/cppy/tree/master/examples).

### FAQ

Problem: I get the following error:
```python
"IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"
```

Solution: Indexing an array with a variable is not allowed by standard numpy arrays, but it is allowed by cpmpy-numpy arrays. First convert your numpy array to a cpmpy-numpy array with the 'cparray()' wrapper:
```python
m = cparray(m); m[X] == True
```

### Roadmap

TODOs:

- auto translate to or-tools
- auto translate to numberjack, which is Python-based but not numpy-based
- add more models (see Hakan K's page(s))
- publish on pypi, with proper docs

### License

This library is delivered under the MIT License, (see [LICENSE](https://github.com/tias/cppy/blob/master/LICENSE)).
