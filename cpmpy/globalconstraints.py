from .expressions import *

# in one file for easy overview, does not include interpretation
# TODO: docstrings, generic decomposition method

def alldifferent(variables):
    return GlobalConstraint("alldifferent", variables)

def decompose_alldifferent(alldiff):
    # TODO
    raise NotImplementedError()

def circuit(variables):
    return GlobalConstraint("circuit", variables)

def decompose_circuit(circ):
    # TODO, see Hakan's or-tools one
    raise NotImplementedError()

