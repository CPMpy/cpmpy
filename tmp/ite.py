import cpmpy as cp


a,b,c = cp.boolvar(name=tuple("abc"), shape=3)

ite = (a.implies(b)) & ((~a).implies(c))

neg_ite = (a.implies(c)) & ((~a).implies(b))

pos_sols = cp.Model(ite).solveAll(display=[a,b,c])
neg_sols = cp.Model(neg_ite).solveAll(display=[a,b,c])

print("Total number of sols:", pos_sols + neg_sols)

print("Both pos and neg?")
print(cp.Model(ite, neg_ite).solve())

print(cp.Model(cp.IfThenElse(a,b,c), ~cp.IfThenElse(a,b,c)).solve())