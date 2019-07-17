Welcome to CPpy

CPpy is a Python-embedded modeling language for constraint programming. It allows you to model your problems in a natural way that follows the math.

It is inspired by CVXpy, SciPy and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure.

Currently, it is only a language generating a model tree. Here is a TODO list:
- add element constraints
- add more models (see Hakan K's page(s))
- auto translate to numberjack, which is Python-based but not numpy-based
- auto translate to minizinc and use pymzn

The following examples show the elegance of building on Python/Numpy:
```python
from cppy import *
import numpy as np

# Construct the model.
s,e,n,d,m,o,r,y = IntVar(0,9, 8)

constr_alldiff = alldifferent([s,e,n,d,m,o,r,y])
constr_sum = [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
               == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) )
             ]
constr_0 = [s > 0, m > 0]

model = Model(constr_alldiff, constr_sum, constr_0)
stats = model.solve()

print("  S,E,N,D =  ", [x.value for x in [s,e,n,d]])
print("  M,O,R,E =  ", [x.value for x in [m,o,r,e]])
print("M,O,N,E,Y =", [x.value for x in [m,o,n,e,y]])
```

sudoku and others need matrix indexing, which numpy supports extensively:
```python
x = 0 # cells whose value we seek
puzzle = numpy.array([
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
(n,_) = puzzle.shape # get matrix dimensions
x = IntVar(1, n, puzzle.shape)

# constraints on values
constr_values = ( x[puzzle>0] == puzzle[puzzle>0] )

# constraints on rows and columns
constr_row = [alldifferent(row) for row in x]
constr_col = [alldifferent(col) for col in x.T]

# constraint on blocks
constr_block = [] 
for i in range(0,n,3):
    for j in range(0,n,3):
        constr_block.append( alldifferent(x[i:i+3, j:j+3]) )

model = Model(constr_values, constr_row, constr_col, constr_block)
```

and an OR problem for good faith:
```python
# Problem data.
demands = [8, 10, 7, 12, 4, 4]
slots = len(demands)

# Construct the model.
x = IntVar(0,sum(demands), slots)

objective = Minimise(sum(x)) # number of buses

constr_demand = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
constr_midnight = [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

model = Model(objective, constr_demand, constr_midnight)
```
