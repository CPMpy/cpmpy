#!/usr/bin/python
"""
Bus scheduling in CPpy

Based on the Numberjack model of Hakan Kjellerstrand:
Problem from Taha "Introduction to Operations Research", page 58.
This is a slightly more general model than Taha's.
"""
from cppy import *
import numpy

# Problem data.
demands = [8, 10, 7, 12, 4, 4]
slots = len(demands)

# Construct the model.
x = IntVar(0,sum(demands), slots)

objective = Minimise(sum(x)) # number of buses

constr_demand = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
constr_midnight = [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

model = Model(objective, constr_demand, constr_midnight)
print(model)

# Solve it.
stats = model.solve()
print("Value:", objective.value)
print("Solution:", x.value)
