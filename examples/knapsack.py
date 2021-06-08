#!/usr/bin/python3
"""
Knapsack problem in CPMpy
 
Based on the Numberjack model of Hakan Kjellerstrand
"""
from cpmpy import *
import numpy as np

# Problem data
n = 10
np.random.seed(1)
values = np.random.randint(0,10, n)
weights = np.random.randint(1,5, n)
capacity = np.random.randint(sum(weights)*.2, sum(weights)*.5)

# Construct the model.
x = BoolVar(shape=n, name="x")

constraint = [ sum(x*weights) <= capacity ]
objective  = sum(x*values)

model = Model(constraint, maximize=objective)
print(model)
print("") # blank line

print("Value:", model.solve()) # solve returns objective value
print("Solution:", x.value())
print("In items: ", [i+1 for i,val in enumerate(x.value()) if val]) # offset 0+1
