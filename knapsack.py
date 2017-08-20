#!/usr/bin/python
"""
Knapsack problem in CPPY.
 
Simple knapsack problem.

Based on the Numberjack model of Hakan Kjellerstrand

"""
from cppy import *
import numpy

# Problem data.
n = 10
numpy.random.seed(1)
values = numpy.random.randn(n)
weights = numpy.random.randn(n)
r = numpy.random.randint(sum(weights)*.3, sum(weights)*.6)

# Construct the model.
x = BoolVar(n)

objective = Maximise(Sum(x*values))
constraint = [Sum(x*weights) <= r]

model = Model(objective, constraint)

# Statistics are returned after solving.
stats = model.solve()
# Variables can be asked for their value in the found solution
print objective.value
print x.value
