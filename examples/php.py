import cpmpy as cp
n=5
x = cp.boolvar(shape=(n,n-1), name="x")
model = cp.Model()
model += cp.cpm_array(x.sum(axis=1)) >= 1
model += cp.cpm_array(x.sum(axis=0)) <= 1

assert model.solve() is False

