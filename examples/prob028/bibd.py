"""
Balanced Incomplete Block Design (BIBD) in cpmpy.

This is a port of Numberjack example Bibd.py:
'''
Balanced Incomplete Block Design (BIBD) --- CSPLib prob028

A BIBD is defined as an arrangement of v distinct objects into b blocks such
that each block contains exactly k distinct objects, each object occurs in
exactly r different blocks, and every two distinct objects occur together in
exactly lambda blocks. Another way of defining a BIBD is in terms of its
incidence matrix, which is a v by b binary matrix with exactly r ones per row,
k ones per column, and with a scalar product of lambda 'l' between any pair of
distinct rows.
'''

Model created by Hakan Kjellerstrand, hakank@hakank.com
See also my cpmpy page: http://www.hakank.org/cpmpy/

Modified by Ignace Bleukx
"""
import numpy as np
from cpmpy import *
from cpmpy.expressions.utils import all_pairs

def get_model(v, b, r, k, l):
    matrix = boolvar(shape=(v, b),name="matrix")

    model = Model()

    model += [sum(row) == r for row in matrix],    # every row adds up to r
    model += [sum(col) == k for col in matrix.T],  # every column adds up to k

    # the scalar product of every pair of columns adds up to l
    model += [np.dot(row_i, row_j) == l for row_i, row_j in all_pairs(matrix)]
    return matrix, model


def solve(param):
    matrix, model = get_model(param['v'], param['b'], param['r'], param['k'], param['l'])
    # break symmetry
    # lexicographic ordering of rows
    for r in range(param['v']-1):
        b = boolvar(param['b']+1)
        model += b[0] == 1
        model += b == ((matrix[r] <= matrix[r + 1]) &
                       ((matrix[r] < matrix[r + 1])) | b[1:] == 1)
        model += b[-1] == 0
    # lexicographic ordering of cols
    for c in range(param['b']-1):
        b = boolvar(param['v']+1)
        model += b[0] == 1
        model += b == ((matrix.T[c] <= matrix.T[c + 1]) &
                       ((matrix.T[c] < matrix.T[c + 1])) | b[1:] == 1)
        model += b[-1] == 0

    # find all solutions of model
    num_solutions = model.solveAll(solution_limit=param['num_sols'],
                                   display = lambda: print(matrix.value(), end="\n\n"))

    print(f"{num_solutions=}")


default = {'v': 7, 'b': 7, 'r': 3, 'k': 3, 'l': 1, 'num_sols':0}
solve(default)