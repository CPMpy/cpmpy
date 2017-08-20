#!/usr/bin/python
"""
Send more money in CPPY.

   SEND
 + MORE
 ------
  MONEY

"""
from cppy import *
import numpy

# Construct the model.
s,e,n,d,m,o,r,y = IntVar(0, 9, size=8)

c_adiff = alldifferent([s,e,n,d,m,o,r,y])
c_math = [ Sum(   numpy.flip([s,e,n,d]) * numpy.power(10, range(0,4)) ) +
           Sum(   numpy.flip([m,o,r,e]) * numpy.power(10, range(0,4)) ) ==
           Sum( numpy.flip([m,o,n,e,y]) * numpy.power(10, range(0,5)) )
         ]
c_0 = [s > 0, m > 0]

model = Model(c_adiff, c_math, c_0)

stats = model.solve()
print "  S,E,N,D =", [x.value for x in [s,e,n,d]]
print "  M,O,R,E =", [x.value for x in [m,o,r,e]]
print "M,O,N,E,Y =", [x.value for x in [m,o,n,e,y]]
