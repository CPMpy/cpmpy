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
    """Creates callback to return distance between points."""
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

n_city = len(locations)
distance_matrix = compute_euclidean_distance_matrix(locations)

# x[i,j] = 1 means that the salesman goes from node i to node j 
x = IntVar(0, 1, shape=distance_matrix.shape) 

# y[i,j] is the number of cars, which the salesman has after leaving
# node i and before entering node j; in terms of the network analysis,
# y[i,j] is a flow through arc (i,j)
# This will help in subtour elimination
y = IntVar(0, n_city-1, shape=distance_matrix.shape)


constraint  = []
# # the salesman leaves and enter  each node i exactly once 
constraint  += [sum(x[i,:])==1 for i in range(n_city)]
constraint  += [sum(x[:,i])==1 for i in range(n_city)]
## No self loop
constraint += [sum(x[i,i] for i in range(n_city))==0]

# flow into node i through all ingoing arcs  equal to 
# flow out of node i through all outgoing arcs + 1
constraint  += [sum(y[:,i])==sum(y[i,:])+1 for i in range(1,n_city)]
#the salesman leaves with n-1 flows
constraint  += [sum(y[0,:])==(n_city-1) ]

#the objective is to minimze  the travel distance 
for i in range(n_city):
    for j in range(n_city):
        constraint  += [y[i,j] <= (n_city-1)*x[i,j]]
objective = sum(x*distance_matrix)

model = Model(constraint, minimize=objective)
# print(model)

stats = model.solve()
print(stats)
print(x.value())
print(objective.value())

solution = x.value()
objective = np.sum(solution*distance_matrix)
print("Solution")
print("Total Cost of solution",objective)
dest = 100
source = 0
while dest !=0:
    dest = np.argmax(solution[source])
    if source==0:
        print("First, from Stop index {} to Stop Index {}".format(source,dest))
    elif dest==0:
        print("And finally, from Stop index {} to Stop Index {}".format(source,dest))
    else:
        print("Then, from Stop index {} to Stop Index {}".format(source,dest))
    source = dest

