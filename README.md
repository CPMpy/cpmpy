Welcome to CPPY

CPPY is a Python-embedded modeling language for constraint programming. It allows you to model your problems in a natural way that follows the math.

It is inspired by CVXPY, Keras and Numberjack, and as most modern scientific Python tools, it uses numpy arrays as basic data structure.

For now, it is also a FICTIONAL language, though an implementable one. Here is the TODO list:
- add more models (see Hakan K's page(s))
- implement the basic objects so the examples at least run error free
- auto translate to numberjack, which is Python-based but not numpy-based
- auto translate to minizinc and use pymzn

The following examples show the elegance of building on Python/Numpy:
```python
from cppy import *
import numpy

# Construct the model.
s,e,n,d,m,o,r,y = IntVar(0, 9, size=8)

c_adiff = alldifferent([s,e,n,d,m,o,r,y])
c_math = [ Sum(   numpy.flip([s,e,n,d]) * numpy.power(10, range(0,4)) ) +
           Sum(   numpy.flip([m,o,r,e]) * numpy.power(10, range(0,4)) ) ==
           Sum( numpy.flip([m,o,n,e,y]) * numpy.power(10, range(0,5)) )
         ]
c_0 = [s > 0, m > 0]

model = Model(c_adiff, c_math, c_0)

stats = model.solve()
print "  S,E,N,D =", [x.value for x in [s,e,n,d]]
print "  M,O,R,E =", [x.value for x in [m,o,r,e]]
print "M,O,N,E,Y =", [x.value for x in [m,o,n,e,y]]
```

sudoku and others need matrix indexing, which numpy supports extensively:
```python
# Problem data.
n = 9
puzzle = numpy.array([
    [0, 0, 0, 2, 0, 5, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 7, 3, 0],
    [0, 0, 2, 0, 0, 9, 0, 6, 0],
    [2, 0, 0, 0, 0, 0, 4, 0, 9],
    [0, 0, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 9, 0, 0, 0, 0, 0, 1],
    [0, 8, 0, 4, 0, 0, 1, 0, 0],
    [0, 6, 3, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 6, 0, 8, 0, 0, 0]])

# Construct the model.
x = IntVar(1, n, puzzle.shape)

c_val = [] # constraint on values
for index, v in np.ndenumerate(puzzle):
    if v != 0:
        c_val.append( x[index] != v )

# constraints on rows and columns
c_row = [alldifferent(row) for row in puzzle]
c_col = [alldifferent(col) for col in puzzle.T]

c_block = [] # constraint on blocks
reg = numpy.sqrt(n)
for i in xrange(0,n,reg):
    for j in xrange(0,n,reg):
        c_block.append( alldifferent(puzzle[i:i+3, j:j+3]) )

model = Model(c_val, c_row, c_col, c_block)
```

and an OR problem for good faith:
```python
# Problem data.
demands = [8, 10, 7, 12, 4, 4]
slots = len(demands)

# Construct the model.
x = IntVar(0, sum(demands), size=slots)

objective = Minimise(Sum(x)) # number of buses

c_demand = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
c_midnight = [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

model = Model(objective, c_demand, c_midnight)
```
