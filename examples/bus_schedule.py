#!/usr/bin/python3
"""
Bus scheduling in CPMpy

Based on the Numberjack model of Hakan Kjellerstrand:
Problem from Taha "Introduction to Operations Research", page 58.
This is a slightly more general model than Taha's.
"""
from cpmpy import *
import numpy

# data
demands = [8, 10, 7, 12, 4, 4]
slots = len(demands)

# variables
x = IntVar(0,sum(demands), slots)

constraint  = [x[i] + x[i+1] >= demands[i] for i in range(0,slots-1)]
constraint += [x[-1] + x[0] == demands[-1]] # 'around the clock' constraint

objective = sum(x) # number of buses

model = Model(constraint, minimize=objective)
stats = model.solve()
print("Solution:", x.value())
print("Value:", sum(x.value()))
# TODO, should we support objective.value() to compute this sum? e.g. that each operator knows how to compute its value?
