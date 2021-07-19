#!/usr/bin/python3
"""
Balanced Incomplete Block Design (BIBD) in CPMpy

CSPlib prob028

Problem description from the numberjack example:
A BIBD is defined as an arrangement of v distinct objects into b blocks such
that each block contains exactly k distinct objects, each object occurs in
exactly r different blocks, and every two distinct objects occur together in
exactly lambda blocks.
Another way of defining a BIBD is in terms of its
incidence matrix, which is a v by b binary matrix with exactly r ones per row,
k ones per column, and with a scalar product of lambda 'l' between any pair of
distinct rows.
"""

# load the libraries
import numpy as np
from cpmpy import *

# Data
v,b = 7,7
r,k = 3,3
l = 1


# Variables, incidence matrix
block = boolvar(shape=(v,b), name="block")

# Constraints on incidence matrix
m = Model(
        [sum(row) == r for row in block],
        [sum(col) == k for col in block.T],
)

# the scalar product of every pair of distinct rows sums up to `l`
for row_a in range(v):
    for row_b in range(row_a+1,v):
        m += sum(block[row_a,:] * block[row_b,:]) == l


if m.solve():
    # pretty print
    print(f"BIBD: {b} obj, {v} blocks, r={r}, k={k}, l={l}")
    for (i,row) in enumerate(block.value()):
        srow = "".join('X ' if e else '  ' for e in row)
        print(f"Object {i+1}: [ {srow}]")
else:
    print("No solution found")
