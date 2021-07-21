"""
Example of diverse solution search

From the docs/multiple_solutions.md file
"""
import numpy as np
from cpmpy import *
from cpmpy.solvers import CPM_ortools


print("Hamming distance example:")
# Diverse solutions, Hamming distance (inequality)
x = boolvar(shape=6)
m = Model(sum(x) == 2)
m = CPM_ortools(m) # optional but faster

K = 3
store = []
while len(store) < 3 and m.solve():
    print(len(store), ":", x.value())
    store.append(x.value())
    # Hamming dist: nr of different elements
    m.maximize(sum([sum(x != sol) for sol in store]))


print("Euclidean distance example:")
# Diverse solutions, Euclidian distance (absolute difference)
x = intvar(0,4, shape=6)
m = Model(sum(x) > 10, sum(x) < 20)
m = CPM_ortools(m) # optional but faster

K = 3
store = []
while len(store) < K and m.solve() is not False:
    print(len(store), ":", x.value())
    store.append(x.value())
    # Euclidian distance: absolute difference in value
    m.maximize(sum([sum( abs(np.add(x, -sol)) ) for sol in store]))
