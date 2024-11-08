#!/usr/bin/env python
from cpmpy import *
import numpy as np
import math
"""
A Price-Collecting TSP variant:
    - a path over a subset of locations must be followed
    - minimize penalty of locations NOT visited
    - minimize travel cost

This variant is used to model the scheduling of a hot strip mill
in the steel industry for example.

Demonstrates the power of using ortools' arc-based "circuit" propagator
for everything related to circuit/subcircuit/path/subpath

The data is based on the TSP example (which is based on ortools')
"""

def compute_euclidean_distance_matrix(locations):
    """Computes distances between all points (from ortools docs)."""
    n_city = len(locations)
    distances = np.zeros((n_city,n_city))
    for from_counter, from_node in enumerate(locations):
        for to_counter, to_node in enumerate(locations):
            if from_counter != to_counter:
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances.astype(int)


locations= [
    (288, 149), (288, 129), (270, 133), (256, 141), (256, 163), (246, 157),
    (236, 169), (228, 169), (228, 148), (220, 164), (212, 172), (204, 159)
]
N = len(locations)

# generate a random prize for each city
np.random.seed(1)
penalties = np.random.randint(10,30, N)

# generate distances
distance_matrix = compute_euclidean_distance_matrix(locations)


# we use Boolean variables for each arc
arc_in = boolvar(shape=(N,N), name="arc_in")

s = SolverLookup.get("ortools")
#    - a path over a subset of locations must be followed
# providing self arcs (i->i) makes ortools circuit behave as subcircuit 
ort_arcs = [(i,j,b) for (i,j),b in np.ndenumerate(arc_in)]
# add extra dummy column (no self-loop) to allow dummy start (path)
ort_arcs += [(i,N+1,boolvar()) for i in range(N)]
# add extra dummy row (no self-loop) to allow dummy stop (path)
ort_arcs += [(N+1,i,boolvar()) for i in range(N)]
s += DirectConstraint("AddCircuit", ort_arcs)

#    - minimize penalty of locations NOT visited
#    - minimize travel cost
obj_penalties = np.sum( penalties * np.diag(arc_in) )
obj_distances = np.sum( arc_in * distance_matrix )
s.minimize(obj_penalties + obj_distances)

s.solve()
print(s.status())

print(f"Solution has {np.sum(np.diag(arc_in.value()))} stops skipped, with a penalty of {obj_penalties.value()}")
print(f"\ttravel cost is {obj_distances.value()} with a total cost of {int(s.objective_value())}")
def display(e):
    # start is a column that has no entry
    cur = np.argmin(np.sum(e, axis=0))
    msg = f"{cur}"
    while True:
        if np.max(e[cur]) == 0:
            # row has no entry, reached the end
            break
        cur = np.argmax(e[cur]) # next: first true one
        msg += f" --> {cur}"
    print(msg)
display(arc_in.value())
