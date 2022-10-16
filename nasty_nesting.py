from cpmpy import *
m = SolverLookup.get("Glasgow Constraint Solver")

# Variables
b = boolvar(name="b")
x = intvar(1,10, shape=3, name="x")

# Constraints
m += (x[0].any([1, 2, 3]))
m.maximize(sum(x) + 100*b)
print(m)
print(m.solve(), x.value(), b.value())
