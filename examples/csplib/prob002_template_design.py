"""
    CSP model for template design problem implemented in CPMpy.
    Model created by Ignace Bleukx and based on Minizinc implementation of CSPlib.
"""
import sys
import requests
import json

from cpmpy import *

def template_design(n_slots, n_templates, n_var, demand,**kwargs):

    ub = max(demand)

    # create model
    model = Model()

    # decision variables
    production = intvar(1, ub, shape=n_templates, name="production")
    layout = intvar(0,n_var, shape=(n_templates,n_var), name="layout")

    # all slots are populated in a template
    model += all(sum(layout[i]) == n_slots for i in range(n_templates))

    # meet demand
    for var in range(n_var):
        model += sum(production * layout[:,var]) >= demand[var]

    # break symmetry
    # equal demand
    for i in range(n_var-1):
        if demand[i] == demand[i+1]:
            model += layout[0,i] <= layout[0,i+1]
            for j in range(n_templates-1):
                model += (layout[j,i] == layout[j,i+1]).implies \
                        (layout[j+1,i] <= layout[j+1,i+1] )

    # distinguish templates
    for i in range(n_templates-1):
        model += production[i] <= production[i+1]

    # pseudo symmerty
    for i in range(n_var-1):
        if demand[i] < demand[i+1]:
            model += sum(production * layout[:,i]) <= sum(production * layout[:,i+1])

    # minimize number of printed sheets
    model.minimize(sum(production))

    return model, (production, layout)


def get_data(data, pname):
    for entry in data:
        if pname in entry["name"]:
            return entry


if __name__ == "__main__":

    fname = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob002_template_design.json"
    problem_name = "catfood2"

    data = None

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        with open(fname, "r") as f:
            data = json.load(f)

    if len(sys.argv) > 2:
        problem_name = sys.argv[2]

    if data is None:
        data = requests.get(fname).json()

    params = get_data(data, problem_name)

    model, (production, layout) = template_design(**params)

    # solve model
    if model.solve(solver="ortools"):
        print("Pressings Layout")
        for t, l in zip(production.value(), layout.value()):
            print("{:<10}{}".format(t, l))
        print()
        print(f"Total pressings: {sum(production).value()}")
    else:
        print("Model is unsatisfiable")
