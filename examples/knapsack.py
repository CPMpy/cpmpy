#!/usr/bin/python3
"""
Knapsack problem in CPMpy
 
Based on the Numberjack model of Hakan Kjellerstrand
"""
import numpy as np
from cpmpy import *

# Problem data
n = 10
np.random.seed(1)
values = np.random.randint(0,10, n)
weights = np.random.randint(1,5, n)
capacity = np.random.randint(sum(weights)*.2, sum(weights)*.5)

# Construct the model.
x = boolvar(shape=n, name="x")

model = Model(
            sum(x*weights) <= capacity,
        maximize=
            sum(x*values)
        )

print("Value:", model.solve()) # solve returns objective value
print(f"Capacity: {capacity}, used: {sum(x.value()*weights)}")
items = np.where(x.value())[0]
print("In items:", items)
print("Values:  ", values[items])
print("Weights: ", weights[items])
