from cppy.solver_interfaces.ortools_python import ORToolsPython
from cppy import IntVar
from cppy import Model
from cppy.globalconstraints import alldifferent
import numpy as np

def model_sendmoremoney():
    s,e,n,d,m,o,r,y = vars = IntVar(0,9, 8)

    constraint = []
    constraint += [ alldifferent([s,e,n,d,m,o,r,y]) ]
    constraint += [    sum(   [s,e,n,d] * np.flip(10**np.arange(4)) )
                    + sum(   [m,o,r,e] * np.flip(10**np.arange(4)) )
                    == sum( [m,o,n,e,y] * np.flip(10**np.arange(5)) ) ]
    constraint += [ s > 0, m > 0 ]

    model = Model(constraint)
    return model, vars

def test_ortools_python():
    try:
        import ortools
    except ImportError as e:
       return # ignore the test

    # TODO: complete testing of ortool suite
    model = model_sendmoremoney()
    ortools_solver = ORToolsPython()
    stats = model.solve(solver=ortools_solver)
    # TODO: check stats + variables values





