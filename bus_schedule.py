#!/usr/bin/python3
"""
Bus scheduling in CPpy

Based on the Numberjack model of Hakan Kjellerstrand:
Problem from Taha "Introduction to Operations Research", page 58.
This is a slightly more general model than Taha's.
"""
from cppy import *
import numpy

# data
demands = [8, 10, 7, 12, 4, 4]
slots = len(demands)

# variables
x = IntVar(0,sum(demands), slots)

constraint  = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
constraint += [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

objective = Minimise(sum(x)) # number of buses

model = Model(constraint, objective)
stats = model.solve()
print("Value:", objective.value)
print("Solution:", x.value)
