import builtins

from cpmpy import *
from cpmpy.solvers.pysat import CPM_pysat
import numpy as np
import math
from cpmpy.transformations.to_cnf import to_cnf

from cpmpy.transformations.to_bool import intvar_to_boolvar

x1 = intvar(0, 4, name="x1")

x = intvar(0, 5, shape=3, name="x")

x2 = intvar(0, 2, name="x2")
x3 = intvar(0, 3, name="x3")
x = [x1, x2, x3]
a = [3, 2, 5]
a0 = 5

m = Model(
    x1 == 4,
    x1 == x2
)
print("\n -----Make CNF ------")

s = CPM_pysat(m)

print("\n -----Solve ------")
s.solve()