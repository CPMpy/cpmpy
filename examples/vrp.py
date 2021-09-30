#!/usr/bin/python3
from cpmpy import *
import numpy as np
import math
"""
In the Vehicle Routing Problem (VRP), the goal is to find 
a closed path of minimal length
for a fleet of vehicles visiting a set of locations.
If there's only 1 vehicle it reduces to the TSP.
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

# data
depot = 0
locations= [
        (288, 149), (288, 129), (270, 133), (256, 141), (256, 163),
        (236, 169), (228, 169), (228, 148), (220, 164)
]
# Pickup demand of each location 
demand = [0,3,4,8,8,10,5,3,9] # depot has no demand
distance_matrix = compute_euclidean_distance_matrix(locations)

n_city = len(locations)
# 3 vehicles and capacity of each vehicle 20
n_vehicle, q = 3, 20


# x[i,j] = 1 means that a vehicle goes from node i to node j 
x = intvar(0, 1, shape=distance_matrix.shape) 
# y[i,j] is a flow of load through arc (i,j)
y = intvar(0, q, shape=distance_matrix.shape)

model = Model(
    # constraint on number of vehicles (from depot)
    sum(x[0,:]) <= n_vehicle,
    # vehicle leaves and enter each node i exactly once 
    [sum(x[i,:])==1 for i in range(1,n_city)],
    [sum(x[:,i])==1 for i in range(1,n_city)],
    # no self visits
    [sum(x[i,i] for i in range(n_city))==0],

    # from depot takes no load
    sum(y[0,:]) == 0,
    # flow out of node i through all outgoing arcs is equal to 
    # flow into node i through all ingoing arcs + load capacity @ node i
    [sum(y[i,:])==sum(y[:,i])+demand[i] for i in range(1,n_city)],
)

# capacity constraint at each node (conditional on visit)
for i in range(n_city):
    for j in range(n_city):
        model += y[i,j] <= q*x[i,j]

# the objective is to minimze the travelled distance 
# sum(x*dist) does not work because 2D array, use .sum()
model.minimize((x*distance_matrix).sum())

# print(model)

val = model.solve()
print(model.status())

print("Total Cost of solution",val)
sol = x.value()
firsts = np.where(sol[0]==1)[0]
for i, dest in enumerate(firsts):
    msg = f"Vehicle {i}: 0"

    source = 0
    dist = 0
    while dest != 0:
        dist += distance_matrix[source,dest]
        msg += f" --{y[source,dest].value()}--> {dest}[{demand[dest]}]"
        source = dest
        dest = np.argmax(sol[source])

    dist += distance_matrix[source,dest]
    msg += f" --{y[source,dest].value()}--> {dest} :: total km={dist}"
    print(msg)
