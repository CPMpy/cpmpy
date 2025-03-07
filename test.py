#
# import cpmpy as cp
# import sys
# sys.path.insert(0, "/home/ignace/software/IBM/cpoptimizer/bin/x86-64_linux")
#
# s = cp.intvar(0, 10, shape=3, name="start")
# e = cp.intvar(0, 10, shape=3, name="end")
# dur = [1, 4, 3]
# demand = cp.intvar(1,3, shape=3, name="demand")
# cap = 10
# expr = cp.Cumulative(s, dur, e, demand, cap)
#
# model = cp.Model(expr)
# model.minimize(cp.sum(demand))
#
# if model.solve(solver="cpo"):
#     print(s.value(), e.value())
#     print(demand.value())
# else:
#     print("Model is UNSAT!")


# import docplex.cp as docp
# import docplex.cp.model as dom
#
# model = dom.CpoModel()
# start = dom.integer_var(0,10, name="start")
# end = dom.integer_var(0,10, name="end")
#
# var = dom.interval_var((0,10),(0,10),length=3, name="var")
#
# model.add(dom.start_of(var) == start)
# model.add(dom.end_of(var) == end)
#
# height = dom.pulse(interval=var, height=(1,5))
#
# print(model.solve())

import cpmpy as cp
from cpmpy.expressions.core import Operator
from cpmpy.expressions.utils import argvals

x,y,d,r = cp.intvar(-5, 5, shape=4,name=['x','y','d','r'])

vars = [x,y,d,r]
m = cp.Model()
# modulo toplevel
m += x / y == d
m += x % y == r
sols = set()

transformed = cp.SolverLookup.get("gurobi").transform(m.constraints)

sols = set()
res = cp.Model(transformed).solveAll(solver="ortools", solution_limit=200, display=lambda: sols.add(tuple(argvals(vars))))
print(res)
for sol in sols:
    xv, yv, dv, rv = sol
    assert dv * yv + rv == xv
    assert (Operator('div', [xv, yv])).value() == dv
    assert (Operator('mod', [xv, yv])).value() == rv
