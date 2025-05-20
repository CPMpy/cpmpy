"""
A collection of XCSP3 solver-native global constraints.
"""

import cpmpy as cp
import numpy as np

# --------------------------------- OR-Tools --------------------------------- #

def ort_nooverlap2d(cpm_ortools, cpm_expr):
    start_x, dur_x, end_x, start_y, dur_y, end_y = cpm_ortools.solver_vars(cpm_expr.args)
    intervals_x = [cpm_ortools.ort_model.NewIntervalVar(s,d,e, f"xinterval_{s}-{d}-{d}") for s,d,e in zip(start_x,dur_x,end_x)]
    intervals_y = [cpm_ortools.ort_model.NewIntervalVar(s,d,e, f"yinterval_{s}-{d}-{d}") for s,d,e in zip(start_y,dur_y,end_y)]
    return cpm_ortools.ort_model.add_no_overlap_2d(intervals_x, intervals_y)

def ort_subcircuit(cpm_ortools, cpm_expr):
    x = cpm_expr.args
    N = len(x)
    arcvars = cp.boolvar(shape=(N,N))
    # post channeling constraints from int to bool
    cpm_ortools += [b == (x[i] == j) for (i,j),b in np.ndenumerate(arcvars)]
    # post the global constraint
    # posting arcs on diagonal (i==j) allows for subcircuits
    ort_arcs = [(i,j,cpm_ortools.solver_var(b)) for (i,j),b in np.ndenumerate(arcvars)] # Allows for empty subcircuits
    return cpm_ortools.ort_model.AddCircuit(ort_arcs)

def ort_subcircuitwithstart(cpm_ortools, cpm_expr):
    x = cpm_expr.args
    N = len(x)
    arcvars = cp.boolvar(shape=(N,N))
    # post channeling constraints from int to bool
    cpm_ortools += [b == (x[i] == j) for (i,j),b in np.ndenumerate(arcvars)]
    # post the global constraint
    # posting arcs on diagonal (i==j) allows for subcircuits
    ort_arcs = [(i,j,cpm_ortools.solver_var(b)) for (i,j),b in np.ndenumerate(arcvars) if not ((i == j) and (i == cpm_expr.start_index))] # The start index cannot self loop and thus must be part of the subcircuit.
    return cpm_ortools.ort_model.AddCircuit(ort_arcs)


# ----------------------------------- Choco ---------------------------------- #

def choco_subcircuit(cpm_choco, cpm_expr):
    # Successor variables
    succ = cpm_choco.solver_vars(cpm_expr.args)
    # Add an unused variable for the subcircuit length.
    subcircuit_length = cpm_choco.solver_var(cp.intvar(0, len(succ)))
    return cpm_choco.chc_model.sub_circuit(succ, 0, subcircuit_length)


# --------------------------------- Minizinc --------------------------------- #

def minizinc_subcircuit(cpm_minizinc, cpm_expr):
    # minizinc is offset 1, which can be problematic here...
    args_str = ["{}+1".format(cpm_minizinc._convert_expression(e)) for e in cpm_expr.args]
    return "{}([{}])".format(cpm_expr.name, ",".join(args_str))