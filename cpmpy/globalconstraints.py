from .expressions import *

# in one file for easy overview, does not include interpretation
# TODO: docstrings, generic decomposition method

def alldifferent(variables):
    return GlobalConstraint("alldifferent", variables)

def decompose_alldifferent():
    # TODO
    return NotImplementedError()

def circuit(variables):
    return GlobalConstraint("circuit", variables)

def decompose_circuit():
    # TODO, see Hakan's or-tools one
    return NotImplementedError()

