"""
Template design in CPMpy (prob002 in CSPlib)
https://www.csplib.org/Problems/prob002/

This problem arises from a colour printing firm which produces a variety of products from thin board,
including cartons for human and animal food and magazine inserts. Food products, for example, are often marketed as a
basic brand with several variations (typically flavours). Packaging for such variations usually has the same overall
design, in particular the same size and shape, but differs in a small proportion of the text displayed and/or in
colour. For instance, two variations of a cat food carton may differ only in that on one is printed ‘Chicken Flavour’
on a blue background whereas the other has ‘Rabbit Flavour’ printed on a green background. A typical order is for a
variety of quantities of several design variations. Because each variation is identical in dimension, we know in
advance exactly how many items can be printed on each mother sheet of board, whose dimensions are largely determined
by the dimensions of the printing machinery. Each mother sheet is printed from a template, consisting of a thin
aluminium sheet on which the design for several of the variations is etched. The problem is to decide, firstly,
how many distinct templates to produce, and secondly, which variations, and how many copies of each, to include on
each template. The following example is based on data from an order for cartons for different varieties of dry
cat-food.

Implementation based on Minizinc model in CSPlib.
Model created by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""

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
                         (layout[j+1,i] <= layout[j+1,i+1])

    # distinguish templates
    for i in range(n_templates-1):
        model += production[i] <= production[i+1]

    # static symmetry
    for i in range(n_var-1):
        if demand[i] < demand[i+1]:
            model += sum(production * layout[:,i]) <= sum(production * layout[:,i+1])

    # minimize number of printed sheets
    model.minimize(sum(production))

    return model, (production, layout)


def _get_instance(data, pname):
    for entry in data:
        if pname == entry["name"]:
            return entry
    raise ValueError(f"Problem instance with name {pname} not found, use --list-instances to get the full list.")

def _print_instances(data):
    import pandas as pd
    df = pd.json_normalize(data)
    df_str = df.to_string(columns=["name", "n_slots", "n_templates", "n_var"], na_rep='-')
    print(df_str)

if __name__ == "__main__":
    import requests
    import json
    import argparse

    import numpy as np

    # argument parsing
    url = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob002_template_design.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-instance', default="catfood2", help="Name of the problem instance found in file 'filename'")
    parser.add_argument('-filename', default=url, help="File containing problem instances, can be local file or url")
    parser.add_argument('--list-instances', help='List all problem instances', action='store_true')

    args = parser.parse_args()

    if "http" in args.filename:
        problem_data = requests.get(args.filename).json()
    else:
        with open(args.filename, "r") as f:
            problem_data = json.load(f)

    if args.list_instances:
        _print_instances(problem_data)
        exit(0)

    problem_params = _get_instance(problem_data, args.instance)
    print("Problem name:", problem_params["name"])

    model, (production, layout) = template_design(**problem_params)

    # solve model
    if model.solve(solver="ortools"):
        np.set_printoptions(linewidth=problem_params['n_var']*5)
        print("#Pressings \t Layout")
        for t, l in zip(production.value(), layout.value()):
            print("{:>10}\t {}".format(t, l))
        print()
        print(f"Total pressings: {sum(production).value()}")
    else:
        raise ValueError("Model is unsatisfiable")
