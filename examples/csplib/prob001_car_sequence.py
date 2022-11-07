"""
Car sequencing in CPMpy (prob001 in CSPlib)
A number of cars are to be produced; they are not identical, because different options are available as variants on the basic model.
The assembly line has different stations which install the various options (air-conditioning, sun-roof, etc.).
These stations have been designed to handle at most a certain percentage of the cars passing along the assembly line.
Furthermore, the cars requiring a certain option must not be bunched together, otherwise the station will not be able to cope.
Consequently, the cars must be arranged in a sequence so that the capacity of each station is never exceeded.
For instance, if a particular station can only cope with at most half of the cars passing along the line, the sequence must be built so that at most 1 car in any 2 requires that option.
The problem has been shown to be NP-complete (Gent 1999).

https://www.csplib.org/Problems/prob001/

Based on the Minizinc model car.mzn.

Data format compatible with both variations of model (with and without block constraints)
Model was created by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""

from numpy.lib.stride_tricks import sliding_window_view

from cpmpy import *

def car_sequence(n_cars, n_options, n_classes, n_cars_p_class, options, capacity=None, block_size=None, **kwargs):
    # build model
    model = Model()

    # decision variables
    slots = intvar(0, n_classes - 1, shape=n_cars, name="slots")
    setup = boolvar(shape=(n_cars, n_options), name="setup")

    # convert options to cpm_array
    options = cpm_array(options)

    # satisfy demand
    model += [sum(slots == c) == n_cars_p_class[c] for c in range(n_classes)]

    # car has correct options
    # This can be written cleaner, see issue #117 on github
    # m += [setup[s] == options[slots[s]] for s in range(n_cars)]
    for s in range(n_cars):
        model += [setup[s, o] == options[slots[s], o] for o in range(n_options)]

    if capacity is not None:
        # satisfy block capacity
        for o in range(n_options):
            setup_seq = setup[:, o]
            # get all setups within block size of each other
            blocks = sliding_window_view(setup_seq, block_size[o])
            for block in blocks:
                model += sum(block) <= capacity[o]

    return model, (slots, setup)


# Helper functions
def _get_instance(data, pname):
    for entry in data:
        if pname == entry["name"]:
            return entry
    raise ValueError(f"Problem instance with name '{pname}' not found, use --list-instances to get the full list.")


def _print_instances(data):
    import pandas as pd
    df = pd.json_normalize(data)
    df_str = df.to_string(columns=["name", "n_cars", "n_options", "n_classes", "note"], na_rep='-')
    print(df_str)


if __name__ == "__main__":
    import argparse
    import json
    import requests

    # argument parsing
    url = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob001_car_sequence.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-instance', nargs='?', default="Problem 4/72", help="Name of the problem instance found in file 'filename'")
    parser.add_argument('-filename', nargs='?', default=url, help="File containing problem instances, can be local file or url")
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

    model, (slots, setup) = car_sequence(**problem_params)

    # solve the model
    if model.solve():
        print("Class", "Options req.", sep="\t")
        for i in range(len(slots)):
            print(slots.value()[i],
                  setup.value()[i].astype(int),
                  sep="\t\t")
    else:
        raise ValueError("Model is unsatisfiable!")
