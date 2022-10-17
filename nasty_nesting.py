from cpmpy import *
m = SolverLookup.get("Glasgow Constraint Solver")

# Variables
b = boolvar(name="b")
x = intvar(1,10, shape=3, name="x")

# Constraints
m += (x[0] == 1)
m += AllDifferent(x)
m += b.implies(x[1] + x[2] > 5)

m.maximize(sum(x) + 100*b)
print(m)
print(m.solve(), x.value(), b.value())
