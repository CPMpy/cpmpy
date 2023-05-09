#!/usr/bin/python3
"""
Knapsack problem in CPMpy
 
Example of using float variables with gurobi, through directvar
"""
import numpy as np
from cpmpy import *
from cpmpy.expressions.variables import directvar

def run():
    # Problem data
    n = 10
    np.random.seed(1)
    values = np.random.randint(0,10, n)
    weights = np.random.randint(1,5, n)
    capacity = np.random.randint(sum(weights)*.2, sum(weights)*.5)

    # Direct Variable: grbAddVar(lb,ub,coef,type,name)
    from gurobipy import GRB
    x = directvar("grbAddVar", (0.0,1.0,0.0,GRB.BINARY), insert_name_at_index=4, name="x", shape=n)
    # to test float vars: GRB.CONTINUOUS

    s = SolverLookup.get("gurobi")
    s += sum(x*weights) <= capacity
    s.maximize(x*values)

    s.solve()
    print("Value:", model.objective_value())
    print("Items (continuous)", [v.x for v in s.solver_vars(x)])  # v.x is gurobi's way fo getting the value

if __name__ == "__main__":
    try:
        # skip if not supported
        s = SolverLookup.get("gurobi")
        run()
    except:
        print("Gurobi solver not installed? skipping")
        pass