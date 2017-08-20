#!/usr/bin/python
"""
Bus scheduling in CPPY.

Problem from Taha "Introduction to Operations Research", page 58.
This is a slightly more general model than Taha's.

Based on the Numberjack model of Hakan Kjellerstrand

"""
from cppy import *
import numpy

# Problem data.
demands = [8, 10, 7, 12, 4, 4]
slots = len(demands)

# Construct the model.
x = IntVar(0, sum(demands), size=slots)

objective = Minimise(Sum(x)) # number of buses

c_demand = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
c_midnight = [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

model = Model(objective, c_demand, c_midnight)

# Solve it.
stats = model.solve()
print objective.value
print x.value
