#!/usr/bin/python3
from cpmpy import *
import numpy as np
import math
"""
  Taken from Google Ortools example https://developers.google.com/optimization/routing/tsp
  
In the Vehicle Routing Problem (VRP), the goal is to find 
a closed path of minimal length
for a fleet of vehicles visiting a set of locations.
If there's only 1 vehicle it reduces to the TSP.
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
depot = 0
locations= [
        (288, 149), (288, 129), (270, 133), (256, 141), (256, 157), (246, 157),
        (236, 169), (228, 169), (228, 161), (220, 169)
]
# Pickup Capacities of each location 
capacities = [0,3,4,8,9,8,10,5,3,9] # depot has no capcity


n_city = len(locations)
distance_matrix = compute_euclidean_distance_matrix(locations)
n_vehicle = 3
# 3 vehicles and capacity of each vehicle 25
q =  25

# x[i,j] = 1 means that a vehicle goes from node i to node j 
x = IntVar(0, 1, shape=distance_matrix.shape) 
# y[i,j] is a flow of load through arc (i,j)
y = IntVar(0, q, shape=distance_matrix.shape)

constraint  = []
# constarint on number of vehicles
constraint  += [sum(x[0,:])<= n_vehicle ]
# # vehicle leaves and enter  each node i exactly once 
constraint  += [sum(x[i,:])==1 for i in range(1,n_city)] 
constraint  += [sum(x[:,i])==1 for i in range(1,n_city)]
constraint += [sum(x[i,i] for i in range(n_city))==0] ## No self loop
# the vehicle return at the depot with no load
constraint  += [sum(y[:,0])==0 ]
# flow into node i through all ingoing arcs  equal to 
# flow out of node i through all outgoing arcs + load capcity @ node i
constraint  += [sum(y[:,i])==sum(y[i,:])+capacities[i] for i in range(1,n_city)]

objective =0 
#the objective is to minimze  the travel distance 
for i in range(n_city):
    for j in range(n_city):
        constraint  += [y[i,j] <= q*x[i,j]]
        objective += x[i,j]*distance_matrix[i,j] 

## this is not working
# objective = sum(x*distance_matrix)

model = Model(constraint, minimize=objective)
# print(model)

stats = model.solve()
print(stats)
print(x.value())

sol = x.value()
objective = np.sum(sol*distance_matrix)
print('Solution')
print("Total Cost of solution",objective)
firsts = np.where(sol[0]==1)[0]
for f in range(len(firsts)):
    print("Vehicle {}".format(f+1))
    
    dest = 100
    source = firsts[f]
    print("Vehicle {} goes from Depot to Stop Index {}".format(f+1,source))
    
    while dest !=0:
        dest = np.argmax(sol[source])


        if source==0:
            print("Vehicle {} goes from Stop index {} to Stop Index {}".format(f+1,source,dest))
        elif dest==0:
            print("And finally, Vehicle {} goes from Stop index {} to Depot".format(f+1,source))
        else:
            print("Then, Vehicle {} goes  from Stop index {} to Stop Index {} ".format(f+1, source,dest))
        source = dest
    print("______________________")