#!/usr/bin/python
"""
Send more money in CPpy

   SEND
 + MORE
 ------
  MONEY

"""
from cppy import *
import numpy as np

# Construct the model.
s,e,n,d,m,o,r,y = IntVar(0,9, 8)

constr_alldiff = alldifferent([s,e,n,d,m,o,r,y])
constr_sum = [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
               == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) )
             ]
constr_0 = [s > 0, m > 0]

model = Model(constr_alldiff, constr_sum, constr_0)
print(model)

stats = model.solve()
print("  S,E,N,D =  ", [x.value for x in [s,e,n,d]])
print("  M,O,R,E =  ", [x.value for x in [m,o,r,e]])
print("M,O,N,E,Y =", [x.value for x in [m,o,n,e,y]])
