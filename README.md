## Welcome to CPMpy

CPMpy is a numpy-based library for conveniently modeling constraint programming problems in Python.

It aims to connect to common constraint solving systems that have a Python API, such as or-tools, as well as other CP modeling languages with a python API (python-MiniZinc, PyCSP3, NumberJack) that in turn support a wide range of solvers.

It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure. You can read about its origins and design decisions in [this short paper](https://github.com/tias/cppy/blob/master/docs/modref19_cppy.pdf).

### Quick start

CPMpy is available in the Python Package Index, and hence can be installed as follows:

    pip install cpmpy

Installing it this way automatically installs the dependencies (numpy and ortools), after which you are ready to go.
Note that CPMpy also supports other solvers (e.g. Minizinc) but the default solver is ortools, for the further detials please check the [documentation](https://cpmpy.readthedocs.io/en/latest/tutorial/how_to_install.html).  

You can then model and solve constraint programming problems using python and numpy, for example:
```python
import numpy as np
from cpmpy import *

e = 0 # value for empty cells
given = np.array([
    [e, e, e,  2, e, 5,  e, e, e],
    [e, 9, e,  e, e, e,  7, 3, e],
    [e, e, 2,  e, e, 9,  e, 6, e],

    [2, e, e,  e, e, e,  4, e, 9],
    [e, e, e,  e, 7, e,  e, e, e],
    [6, e, 9,  e, e, e,  e, e, 1],

    [e, 8, e,  4, e, e,  1, e, e],
    [e, 6, 3,  e, e, e,  e, 8, e],
    [e, e, e,  6, e, 8,  e, e, e]])


# Variables
puzzle = IntVar(1,9, shape=given.shape)

constraints = []
# Constraints on rows and columns
constraints += [ alldifferent(row) for row in puzzle ]
constraints += [ alldifferent(col) for col in puzzle.T ] # numpy's Transpose

# Constraints on blocks
for i in range(0,9, 3):
    for j in range(0,9, 3):
        constraints += [ alldifferent(puzzle[i:i+3, j:j+3]) ] # python's indexing

# Constraints on values (cells that are not empty)
constraints += [ puzzle[given!=e] == given[given!=e] ] # numpy's indexing


# Solve and print
if Model(constraints).solve():
    print(puzzle.value())
else:
    print("No solution found")
```

You can try it yourself in [this notebook](https://github.com/tias/cppy/blob/master/examples/quickstart_sudoku.ipynb).

### Documentation

New to constraint programming? Check our [CP basics tutorial](https://github.com/tias/cppy/blob/master/docs/preface/cppy_intro.md).

See also the more extensive documentation on [ReadTheDocs](https://cpmpy.readthedocs.io/).

Including the [API documentation](https://cpmpy.readthedocs.io/en/latest/api/model.html)

### More examples

The following examples show the elegance of building on Python/Numpy:
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
constraint += [ s > 0, m > 0 ]

model = Model(constraint)
print(model)

stats = model.solve()
print("  S,E,N,D =  ", [x.value() for x in [s,e,n,d]])
print("  M,O,R,E =  ", [x.value() for x in [m,o,r,e]])
print("M,O,N,E,Y =", [x.value() for x in [m,o,n,e,y]])
```

and an OR problem for good faith:
```python
from cpmpy import *
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

See more examples in the [examples/](https://github.com/tias/cppy/tree/master/examples) directory, including notebooks.


### Helping out
We welcome any feedback, as well as hearing about how you are using it. You are also welcome to reuse any parts in your own project.

A good starting point to help with the development, would be to write more CP problems in CPMpy, and add them to the examples folder.

CPMpy is still in Beta stage, and bugs can still occur. If so, please report the issue on Github!

Are you a solver developer? We are willing to integrate solvers that have a python API on pip. If this is the case for you, or if you want to discuss what it best looks like, do contact us!

### Roadmap

If you are curious, some things we are working on, or considering:

- more tests, better docs
- more examples
- showcases of how we use it in our research
- (idea) a program analyzer that can detect whether a model is a native SAT or MIP problem
- (idea) integration to PySAT when only Boolean variables are used

### FAQ

Problem: I get the following error:
```python
"IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"
```

Solution: Indexing an array with a variable is not allowed by standard numpy arrays, but it is allowed by CPMpy-numpy arrays. First convert your numpy array to a cpmpy-numpy array with the 'cparray()' wrapper:
```python
m = cparray(m); m[X] == True
```
