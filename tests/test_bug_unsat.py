from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_model

b = boolvar(name="b")
i = intvar(lb=0, ub=1, name="i")
k = intvar(lb=0, ub=1, name="k")

m = Model()
m += (b == False) & (b != (k == 1))
m += (i == 0) == (b != (k == 1))
m += i == 0

m.solve(solver="gurobi")
print(m.status())
m.solve(solver="ortools")
print(m.status())