CPMpy is a Constraint Programming and Modeling library in Python, based on numpy, with direct solver access.

Constraint Programming is a methodology for solving combinatorial optimisation problems like assignment problems or covering, packing and scheduling problems. Problems that require searching over discrete decision variables.

CPMpy allows to model search problems in a high-level manner, by defining decision variables and constraints and an objective over them (similar to MiniZinc and Essence'). You can freely use numpy functions and indexing while doing so. This model is then automatically translated to state-of-the-art solver like or-tools, which then compute the optimal answer. 

Getting started:

- Watch the [tutorial video](https://www.youtube.com/watch?v=A4mmmDAdusQ) on YouTube
- Try it out [online without installation](https://mybinder.org/v2/gh/CPMpy/cpmpy/HEAD?labpath=examples%2Fquickstart_sudoku.ipynb) or browse the [examples/](examples/)
- Install as easily as `pip3 install cpmpy`, or see the detailed [installation instructions](https://cpmpy.readthedocs.io/en/latest/installation_instructions.html)
- Full documentation at [read the docs](https://cpmpy.readthedocs.io/) for more.

Here is a quick highlight of some key features:

- conveniently modeling and solving problems like [sudoku](examples/sudoku.py), [cryptarithmetic](examples/send_more_money.py), [jobshop scheduling](examples/jobshop.py), [traveling salesman problem](examples/tsp.py) and [more](examples/).
- logging search progress and arbitrarily [modifying solver parameters](https://cpmpy.readthedocs.io/en/latest/solver_parameters.html)
- intuitive [hyperparameter search](examples/advanced/hyperparameter_search.py) for a solver
- easy UNSAT core extraction and computing [Minimal Unsatisfiable Subsets](https://cpmpy.readthedocs.io/en/latest/unsat_core_extraction.html) (MUS) of CP problems


It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure. You can read about its origins and design decisions in [this short paper](https://github.com/tias/cppy/blob/master/docs/modref19_cppy.pdf).

### An example
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
puzzle = intvar(1,9, shape=given.shape, name="puzzle")

model = Model(
    # Constraints on rows and columns
    [AllDifferent(row) for row in puzzle],
    [AllDifferent(col) for col in puzzle.T], # numpy's Transpose
)

# Constraints on blocks
for i in range(0,9, 3):
    for j in range(0,9, 3):
        model += AllDifferent(puzzle[i:i+3, j:j+3]) # python's indexing

# Constraints on values (cells that are not empty)
model += (puzzle[given!=e] == given[given!=e]) # numpy's indexing


# Solve and print
if model.solve():
    print(puzzle.value())
else:
    print("No solution found")
```

You can try it yourself in [this notebook](https://github.com/tias/cppy/blob/master/examples/quickstart_sudoku.ipynb).


### Helping out
We welcome any feedback, as well as hearing about how you are using it. You are also welcome to reuse any parts in your own project.

A good starting point to help with the development, would be to write more CP problems in CPMpy, and add them to the examples folder.

CPMpy is still in Beta stage, and bugs can still occur. If so, please report the issue on Github!

Are you a solver developer? We are willing to integrate solvers that have a python API on pip. If this is the case for you, or if you want to discuss what it best looks like, do contact us!

### FAQ
Problem: I get the following error:
```python
"IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"
```

Solution: Indexing an array with a variable is not allowed by standard numpy arrays, but it is allowed by CPMpy-numpy arrays. First convert your numpy array to a cpmpy-numpy array with the 'cparray()' wrapper:
```python
m = cparray(m); m[X] == True
```

### Acknowledgments
Part of the development received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 101002802, [CHAT-Opt](https://people.cs.kuleuven.be/~tias.guns/chat-opt.html)).

You can cite CPMpy as follows: "Guns, T. (2019). Increasing modeling language convenience with a universal n-dimensional array, CPpy as python-embedded example. The 18th workshop on Constraint Modelling and Reformulation at CP (ModRef 2019).

```
@inproceedings{guns2019increasing,
    title={Increasing modeling language convenience with a universal n-dimensional array, CPpy as python-embedded example},
    author={Guns, Tias},
    booktitle={Proceedings of the 18th workshop on Constraint Modelling and Reformulation at CP (Modref 2019)},
    volume={19},
    year={2019}
}
```

