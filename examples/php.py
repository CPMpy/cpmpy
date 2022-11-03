#!/usr/bin/python3
"""
Pigeonhole principle (PHP) in CPMpy

Fit P pigeons into N holes, such that no two pigeons share the same hole.
Relevant proof complexity problem (only has exponential resolution proofs, but has short cutting plane proofs).
"""

# load the libraries
import numpy as np
from cpmpy import *
from cpmpy.solvers import CPM_ortools

def php(P,H):

    p2h = boolvar(shape=(P,H), name="in hole")

    # Constraints on incidence matrix
    m = Model(
        [sum(holes) == 1 for holes in p2h],
        [sum(pigeons) <= 1 for pigeons in p2h.T],
    )
    
    return (m, p2h)

def php2(P,H):

    p2h = boolvar(shape=(P,H), name="in hole")
    print(p2h)

    # Constraints on incidence matrix
    m = Model()
    m += [sum(holes) == 1 for holes in p2h]
    for h in range(0,H):
        for p1 in range(0,P):
            for p2 in range(p1+1,P):
                m += (~p2h[p1][h] | ~p2h[p2][h])

    return (m, p2h)

def solve(P,H, prettyprint=True):
    (m, p2h) = php(P,H)

    # s = SolverLookup.get("pysat:cadical",m)
    s = SolverLookup.get("exact",m)

    #if s.solve(num_search_workers=1, log_search_progress=True, symmetry_level=0, linearization_level=3):
    print(s.solveAll())
    #if s.solve():
        #print(s.status())

        #if prettyprint:
            ## pretty print
            #for p in range(0,P):
                #line = ""
                #for h in range(0,H):
                    #line+=str(p2h[p,h].value())
                #print(line)
    #else:
        #print("No solution found")

if __name__ == "__main__":
    print(SolverLookup.solvernames())
    solve(5,5)
