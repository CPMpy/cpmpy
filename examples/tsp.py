#!/usr/bin/python3
from cpmpy import *
import numpy as np
import math
"""
Taken from Google Ortools example https://developers.google.com/optimization/routing/tsp
  
The Traveling Salesman Problem (TSP) is stated as follows.
Let a directed graph G = (V, E) be given, where V = {1, ..., n} is
a set of nodes, E <= V x V is a set of arcs. Let also each arc
e = (i,j) be assigned a number c[i,j], which is the length of the
arc e. The problem is to find a closed path of minimal length going
through each node of G exactly once.
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
        (288, 149), (288, 129), (270, 133), (256, 141), (256, 157), (246, 157),
        (236, 169), (228, 169), (228, 161), (220, 169), (212, 169), (204, 169)
]
distance_matrix = compute_euclidean_distance_matrix(locations)
n_city = len(locations)


# x[i,j] = 1 means that the salesman goes from node i to node j 
x = intvar(0, 1, shape=distance_matrix.shape, name="x") 

# y[i,j] is the number of cars, which the salesman has after leaving
# node i and before entering node j; in terms of the network analysis,
# y[i,j] is a flow through arc (i,j)
# This will help in subtour elimination
y = intvar(0, n_city-1, shape=distance_matrix.shape, name="y")

model = Model(
    # the salesman leaves and enters each node i exactly once 
    [sum(x[i,:])==1 for i in range(n_city)],
    [sum(x[:,i])==1 for i in range(n_city)],
    # no self visits
    [sum(x[i,i] for i in range(n_city))==0],

    # salesman leaves with no load
    sum(y[0,:]) == 0,
    # flow out of node i through all outgoing arcs is equal to
    # flow into node i through all ingoing arcs + 1
    # for all but starting node '0'
    [sum(y[i,:])==sum(y[:,i])+1 for i in range(1,n_city)],
)

# capacity constraint at each node (conditional on visit)
for i in range(n_city):
    for j in range(n_city):
        model += y[i,j] <= (n_city-1)*x[i,j]

# the objective is to minimze the travelled distance 
model.minimize(sum(x*distance_matrix))

# print(model)

val = model.solve()
print(model.status())

print("Total Cost of solution",val)
sol = x.value()

source = 0
dest = np.argmax(sol[source])
msg = "0"
while dest != 0:
    msg += f" --> {dest}"
    source = dest
    dest = np.argmax(sol[source])
msg += f" --> {dest}"
print(msg)
