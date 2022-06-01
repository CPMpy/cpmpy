"""
Diamond-free Degree Sequences in cpmpy.

This is a port of the Numberjack example DiamondfreeDegreeSequences.py:
'''
Fill in a binary matrix of size n * n in such a way that
- For every grouping of four rows, the sum of their non-symmetrical
  values is less than or equal to 4,
- No rows contain just zeroes,
- Every row has a sum modulo 3,
- The sum of the matrix is modulo 12.
- No row R contains a 1 in its Rth column.

Note on first constraint in model:
A group of four nodes can have at most four edges between them.
Since the matrix for this model will represent the adjacency
matrix of the graph, we need to take into consideration the fact
that the matrix will be symmetrical.

CSPLib Problem 050 - http://www.csplib.org/Problems/prob050/
'''

Model created by Hakan Kjellerstrand, hakank@hakank.com
See also my cpmpy page: http://www.hakank.org/cpmpy/

Modified by Ignace Bleukx, ignace.bleukx@kuleuven.be

"""
import sys
import numpy as np
from cpmpy import *
from cpmpy.solvers import *
from itertools import combinations

def diamond_free(N=10):
    # By definition a and b will have the same cardinality:
    matrix = boolvar(shape=(N, N), name="matrix")

    model = Model()

    # No rows contain just zeroes.
    model += [sum(row) > 0 for row in matrix] # can be written cleaner, see issue #117
    # Every row has a sum modulo 3.
    model += [sum(row) % 3 == 0 for row in matrix]
    # The sum of the matrix is modulo 12.
    model += sum(matrix) % 12 == 0
    # No row R contains a 1 in its Rth column.
    model += [matrix[np.diag_indices(N)] == 0]

    # Every grouping of 4 rows can have at most a sum of 4 between them.
    for a, b, c, d in combinations(range(N), 4):
        model += sum([matrix[a][b], matrix[a][c], matrix[a][d],
                      matrix[b][c], matrix[b][d], matrix[c][d]]) <= 4

    # Undirected graph
    model += matrix == matrix.T

    # Symmetry breaking
    # break symmetry
    # lexicographic ordering of rows
    for r in range(N - 1):
        b = boolvar(N + 1)
        model += b[0] == 1
        model += b == ((matrix[r] <= matrix[r + 1]) &
                       ((matrix[r] < matrix[r + 1]) | b[1:] == 1))
        model += b[-1] == 0
    # lexicographic ordering of cols
    for c in range(N - 1):
        b = boolvar(N + 1)
        model += b[0] == 1
        model += b == ((matrix.T[c] <= matrix.T[c + 1]) &
                       ((matrix.T[c] < matrix.T[c + 1]) | b[1:] == 1))
        model += b[-1] == 0

    return model, matrix

def print_sol(matrix):
    print(matrix.value())
    print("Degree sequence:", end=" ")
    print(matrix.value().sum(axis=0))
    print()

if __name__ == "__main__":

    n = 10
    num_sol = 0 # find all solutions
    if len(sys.argv) > 1:
        n = sys.argv[1]
    if len(sys.argv) > 2:
        num_sol = sys.argv[2]


    model, matrix = diamond_free(n)
    num_sol = model.solveAll(
        solution_limit=num_sol,
        display=lambda : print_sol(matrix))
    print("num_solutions:",num_sol)

