import cpmpy as cp

a = cp.intvar(-2, 9)

model = cp.Model(0 < a < 3)

print(model.constraints)




