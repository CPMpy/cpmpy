import cpmpy as cp
x, y, z = cp.intvar(0, 3, shape=3, name=["x", "y", "z"])

# A: one add() call with a list
s = cp.SolverLookup.get("highs")
s += [cp.AllDifferent([x, z]), y <= 2, y >= 1]
print(s.native_model.getNumRow())

# B: one add() call per constraint
s = cp.SolverLookup.get("highs")
s += cp.AllDifferent([x, z]); print(s.native_model.getNumRow())
s += y <= 2;                  print(s.native_model.getNumRow())
s += y >= 1;                  print(s.native_model.getNumRow())
