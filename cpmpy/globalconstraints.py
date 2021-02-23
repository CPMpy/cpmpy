from .expressions import *

# in one file for easy overview, does not include interpretation

def alldifferent(variables):
    return GlobalConstraint("alldifferent", [variables])

def circuit(variables):
    return GlobalConstraint("circuit", [variables])
