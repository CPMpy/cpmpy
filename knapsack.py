#!/usr/bin/python3
"""
Knapsack problem in CPpy
 
Based on the Numberjack model of Hakan Kjellerstrand
"""
from cppy import *
import numpy as np

# Problem data
n = 10
np.random.seed(1)
values = np.random.randint(0,10, n)
weights = np.random.randint(1,5, n)
capacity = np.random.randint(sum(weights)*.3, sum(weights)*.6)

# Construct the model.
x = BoolVar(n)

constraint = [ sum(x*weights) <= capacity ]
objective  = sum(x*values)

model = Model(constraint, maximize=objective)
print(model)

# Statistics are returned after solving.
stats = model.solve()
# Variables can be asked for their value in the found solution
print("Value:", objective.value())
print("Solution:", x.value())
