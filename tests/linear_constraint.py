from networkx.algorithms.bipartite.matching import INFINITY
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables

import numpy
import math

from cpmpy.model import BoolModel

def order_encode(x):
    """
        Let y be an integer variable with domain [0,d].
        The order encoding [19, 7] (sometimes called ladder or regular )
        introduces Boolean variables y_i for 0 <= i < d.
        A variable y_i is true iff y <= i.
        The encoding also introduces the clauses:
            yi → yi+1 for 0 <= i < d −1
    """
    clauses = []
    new_vars = boolvar(shape=(x.ub-x.lb), name=x.name)
    i2b_map = {}
    for id, xi in enumerate(range(x.lb, x.ub)):
        print(new_vars[id], "<=",  xi)
        i2b_map[xi] = new_vars[id]
    for id, xi in enumerate(range(x.lb, x.ub-1)):
        clauses.append(new_vars[id].implies(new_vars[id+1]))
        print(new_vars[id].implies(new_vars[id+1]))
    return new_vars, clauses, i2b_map

def log_val(y, base=2):
    return sum(yi.value() * (base**i) for i, yi in enumerate(y))

def log_encode(x, base=2):
    '''
    Let y be an integer variable with domain [0,d].
    y = sum (y_i * 2^i) for i in [0, log(d)]
    '''
    new_vars = boolvar(shape=int(math.log(x, base)), name=x.name)
    return new_vars

def multi_decision_diagram(con):
    """
    A directed acyclic graph is called an ordered Multi Decision Diagram
    if it satisfies the following properties:
        – It has two terminal nodes, namely T (true) and F (false).
        – Each non-terminal node is labeled by an integer variable {x1,x2,··· ,xn}.
        This variable is called selector variable.
        – Every node labeled by xi has the same number of outgoing edges, namely
        di + 1.
        – If an edge connects a node with a selector variable xi and a node with a
        selector variable xj, then j > i
    """
    print(get_variables(con))

    return []



def MDDCreate(a, x, a0):
    '''
    input: Constraint C: a_1 * x_1 + .. + a_n * x_n <= a_0
    output: MDD of C
    '''
    n = len(a)
    L = {}
    for i in range(1, n+1):
        L[i] = None
    L[n+1] = (((-INFINITY,-1), 0), ((0, INFINITY), 1))
    ((beta, gamma), M) = MDDConstruction(1,a, x, a0, L)
    return M

def search(a0, Li):
    """
    searches whether there exists a pair ([β,γ],M) ∈ Li with K ∈ [β,γ]
    """
    ((beta, gamma), M) = ((None, None), None)
    return ((beta, gamma), M)

def MDDConstruction(i, a, x, a0, L):
    ((beta, gamma), M) = search(a0, Li)
    if (beta, gamma) != (None, None):
        return ((beta, gamma), M)
    
    for j in range(0, di+1):
        ((beta_j, gamma_j), M_j) = MDDConstruction(i+1, a[j])

    return ((beta, gamma), M)

def reduce_constraint(con):
    print(con, flatten_constraint(con))

x = intvar(0, 5, name="x")
y = intvar(3, 6, name="y")
order_encode(x)
order_encode(y)

## test cases
c1 = (x + y == 3)
c2 = (2 * x + y == 5)
c3 = (2 * x + y < 5)
c4 = (2 * x + y <= 5)
c5 = (2 * x + 3 * y <= 10)

reduce_constraint(c1)
multi_decision_diagram(c1)

reduce_constraint(c2)
reduce_constraint(c3)
reduce_constraint(c4)
reduce_constraint(c5)



# m = BoolModel(
#     x >= 4, # x = 4, 5
#     y <= 4, # y = 3, 4
# )

# print(m)
# m.solve()
# print(x.value(), y.value())

