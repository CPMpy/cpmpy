"""
    CSP model for template design problem implemented in CPMpy.
    Model created by Ignace Bleukx and based on Minizinc implementation of CSPlib.
"""


from cpmpy import *

from math import ceil
import sys


def get_data(fname, data_name):

    exec(open(fname).read())
    return eval(f"{data_name}()")

def print_result(production, layout):
    print("Pressings Layout")
    for t, l in zip(production, layout):
        print("{:<10}{}".format(t,l))
    print()
    print(f"Total pressings: {sum(production)}")




if __name__ == "__main__":

    # get data
    n_slots, n_templates, n_var, demand = get_data(*sys.argv[1:])

    lb = ceil(min(demand / (n_slots * n_templates)))
    ub = max(demand)


    # create model
    m = Model()

    # decision variables
    production = intvar(1, ub, shape=n_templates, name="production")
    layout = intvar(0,n_var, shape=(n_templates,n_var), name="layout")

    # all slots are populated in a template
    m += all(sum(layout[i]) == n_slots for i in range(n_templates))

    # meet demand
    for var in range(n_var):
        m += sum(production * layout[:,var]) >= demand[var]

    # break symmetry
    # equal demand
    for i in range(n_var-1):
        if demand[i] == demand[i+1]:
            m += layout[0,i] <= layout[0,i+1]
            for j in range(n_templates-1):
                m += (layout[j,i] == layout[j,i+1]).implies \
                        (layout[j+1,i] <= layout[j+1,i+1] )

    # distinguish templates
    for i in range(n_templates-1):
        m += production[i] <= production[i+1]

    # pseudo symmerty
    for i in range(n_var-1):
        if demand[i] < demand[i+1]:
            m += sum(production * layout[:,i]) <= sum(production * layout[:,i+1])

    # minimize number of printed sheets
    m.minimize(sum(production))

    # solve model
    if m.solve(solver="gurobi"):
        print_result(
            production.value(),
            layout.value()
        )
    else:
        print("Model is unsatisfiable")