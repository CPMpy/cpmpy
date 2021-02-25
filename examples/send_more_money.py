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
s,e,n,d,m,o,r,y = IntVar(0,9, 8)

constraint = []
constraint += [ alldifferent([s,e,n,d,m,o,r,y]) ]
constraint += [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                 + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
                == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) ) ]
constraint += [ s > 0, m > 0 ]

model = Model(constraint)
print(model)

stats = model.solve()
print("  S,E,N,D =  ", [x.value() for x in [s,e,n,d]])
print("  M,O,R,E =  ", [x.value() for x in [m,o,r,e]])
print("M,O,N,E,Y =", [x.value() for x in [m,o,n,e,y]])
