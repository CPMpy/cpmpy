#!/usr/bin/python3
"""
Send more money in CPMpy

   SEND
 + MORE
 ------
  MONEY

"""
from cpmpy import *
import numpy as np

# Construct the model.
s,e,n,d,m,o,r,y = intvar(0,9, shape=8)

model = Model(
    AllDifferent([s,e,n,d,m,o,r,y]),
    (    sum(   [s,e,n,d] * np.array([       1000, 100, 10, 1]) ) \
       + sum(   [m,o,r,e] * np.array([       1000, 100, 10, 1]) ) \
      == sum( [m,o,n,e,y] * np.array([10000, 1000, 100, 10, 1]) ) ),
    s > 0,
    m > 0,
)

print(model)
print("") # blank line

# Solve and print
if model.solve():
    print("  S,E,N,D =   ", [x.value() for x in [s,e,n,d]])
    print("  M,O,R,E =   ", [x.value() for x in [m,o,r,e]])
    print("M,O,N,E,Y =", [x.value() for x in [m,o,n,e,y]])
else:
    print("No solution found")
