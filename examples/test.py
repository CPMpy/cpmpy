import cpmpy as cp 

m = cp.Model()

intvars = cp.intvar(0, 10, shape=3, name="intvars")

m += 8//(intvars[0]**2) + 5*intvars[1] + 3*intvars[2] <= 20

m.solve()

print("Solution:")
print(intvars.value())