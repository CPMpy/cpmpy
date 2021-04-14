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
b,v = 7,7
r,k = 3,3
l = 1

# Variables, incidence matrix
block = BoolVar(shape=(v,b))

# Constraints on incidence matrix
m = Model([
        [sum(row) == r for row in block],
        [sum(col) == k for col in block.T],
        # the scalar product of every pair of columns adds up to l
        [sum([(row[col_i] * row[col_j]) for row in block]) == l
            for col_i in range(v) for col_j in range(col_i)],
    ])

if m.solve():
    # pretty print
    print(f"BIBD: {b} obj, {v} blocks, r={r}, k={k}, l={l}")
    for (i,row) in enumerate(block.value()):
        srow = "".join('X ' if e else '  ' for e in row)
        print(f"Object {i+1}: [ {srow}]")
else:
    print("No solution found")
