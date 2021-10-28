import builtins
from networkx.algorithms.bipartite.matching import INFINITY
from cpmpy import *
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.get_variables import get_variables

import numpy as np
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
        L[i] = list()
    L[n+1] = (((-INFINITY,-1), 0), ((0, INFINITY), 1))
    ((beta, gamma), M) = MDDConstruction(1, a, x, a0, L)
    return M + (([beta], [gamma]),)

def search(K, Li):
    """
    searches whether there exists a pair ([β,γ],M) ∈ Li with K ∈ [β,γ]
    """

    for ((beta_j, gamma_j), M_j) in Li:
        if beta_j <= K and K <= gamma_j:
            return ((beta_j, gamma_j), M_j)
    return None

def test_intersect():
    all_beta = [0, 0, -INFINITY, -INFINITY]
    all_gamma = [INFINITY, INFINITY, -1, -1]
    di = 3
    a = [3, 2, 5]
    i = 3
    assert (5, 9) == intersect(all_beta=all_beta, all_gamma=all_gamma, di=di, a=a, i=i)

def intersect(all_beta, all_gamma, di, a, i):
    beta, gamma = -INFINITY, INFINITY
    a_j = a[i-1]
    for beta_j,gamma_j, d_j in zip(all_beta, all_gamma,range(0, di+1)):
        # print(beta_j,gamma_j, a_j, d_j)
        # print("\t\t\tbeta", beta)
        # print("\t\t\tgamma", gamma)
        beta = builtins.max([beta, beta_j + a_j * d_j])
        gamma = builtins.min([gamma, gamma_j + a_j * d_j])

    return beta, gamma

def MDDConstruction(i, a, x, a0, L):
    '''
        a * x = a1 (3) * x1 + 2 * x2 + 5 * x2 <= a0 = 5

    '''
    assert i <= len(L), "Never exceed size of L"
    print("\n", f"x{i}\t {'+'.join([f'{ai}*{xi}' for ai, xi in zip(a,x)])} <= {a0}")

    mdd = search(a0, L[i])
    print(f"\ta0: {a0}\tL[{i}]={L[i]}")

    if mdd is not None:
        return mdd

    #BOTTOM-UP construction of MDD
    di = x[i-1].ub

    print(f"di= [0, {di}]")
    MDDs = []
    all_beta, all_gamma = [], []

    for j in range(0, di+1):
        # COnstruction(i+1, a_(i+1)
        ((beta_j, gamma_j), M_j) = MDDConstruction(i+1, a, x, a0 - j * a[i-1], L)
        # keeping track of bounds and mdds
        # print(f"\t\tbeta_j={beta_j}")
        # print(f"\t\tgamma_j={gamma_j}")
        # print(f"\t\tM_j={M_j}")
        MDDs.append(M_j)
        all_beta.append(beta_j)
        all_gamma.append(gamma_j)

    M = ((x[i-1], MDDs, (all_beta, all_gamma)))
    (beta, gamma) = intersect(all_beta, all_gamma, x[i-1].ub, a, i)
    L[i].append(((beta, gamma), M))

    return ((beta, gamma), M)


x1 = intvar(0, 4, name="x1")

x2 = intvar(0, 2, name="x2")
x3 = intvar(0, 3, name="x3")
x = [x1, x2, x3]
a = [3, 2, 5]
a0 = 5


print(f"{' + '.join([f'{ai}*{xi}' for ai, xi in zip(a,x)])} <= {a0}\n")

mdd = MDDCreate(a=a, x=x, a0=a0)

def print_mdd(mdd, prev="root", level=0, beta=[], gamma=[]):
    # print(mdd)
    if level == 0:
        print("\n")
    xi, mdd_i, (all_beta, all_gamma) = mdd
    if type(mdd_i[0]) is tuple:
        print("\n", "\t"*level, f"{prev}->{xi}")
        for mdd_j in mdd_i:
            print_mdd(mdd_j, prev=xi, level=level+1)
    else:
        print("\t"*level, f"{prev}->{xi}", mdd_i)
print("\n\n")
print(mdd)
print_mdd(mdd)
test_intersect()

# multi_decision_diagram(c1)

# reduce_constraint(c2)
# reduce_constraint(c3)
# reduce_constraint(c4)
# reduce_constraint(c5)



# m = BoolModel(
#     x >= 4, # x = 4, 5
#     y <= 4, # y = 3, 4
# )

# print(m)
# m.solve()
# print(x.value(), y.value())

