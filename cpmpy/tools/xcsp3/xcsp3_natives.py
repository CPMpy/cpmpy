"""
A collection of XCSP3 solver-native global constraints.
"""

import numpy as np
import cpmpy as cp
from cpmpy.expressions.globalconstraints import DirectConstraint


# --------------------------------- OR-Tools --------------------------------- #

class OrtNoOverlap2D(DirectConstraint):
    def __init__(self, arguments):
        super().__init__("ortnooverlap2d", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        start_x, dur_x, end_x, start_y, dur_y, end_y = CPMpy_solver.solver_vars(self.args[0])
        intervals_x = [Native_solver.NewIntervalVar(s,d,e, f"xinterval_{s}-{d}-{d}") for s,d,e in zip(start_x,dur_x,end_x)]
        intervals_y = [Native_solver.NewIntervalVar(s,d,e, f"yinterval_{s}-{d}-{d}") for s,d,e in zip(start_y,dur_y,end_y)]
        return Native_solver.add_no_overlap_2d(intervals_x, intervals_y)

class OrtSubcircuit(DirectConstraint):
    def __init__(self, arguments):
        super().__init__("ortsubcircuit", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        N = len(self.args[0])
        arcvars = cp.boolvar(shape=(N,N))
        # post channeling constraints from int to bool
        CPMpy_solver.add([b == (self.args[0][i] == j) for (i,j),b in np.ndenumerate(arcvars)], internal=True)
        # post the global constraint
        #   posting arcs on diagonal (i==j) allows for subcircuits
        ort_arcs = [(i,j, CPMpy_solver.solver_var(b)) for (i,j),b in np.ndenumerate(arcvars)] # Allows for empty subcircuits

        return Native_solver.AddCircuit(ort_arcs)

class OrtSubcircuitWithStart(DirectConstraint):
    def __init__(self, arguments, start_index:int=0):
        super().__init__("ortsubcircuitwithstart", (arguments, start_index))

    def callSolver(self, CPMpy_solver, Native_solver):
        N = len(self.args[0])
        arcvars = cp.boolvar(shape=(N,N))
        # post channeling constraints from int to bool
        CPMpy_solver.add([b == (self.args[0][i] == j) for (i,j),b in np.ndenumerate(arcvars)], internal=True)
        # post the global constraint
        # posting arcs on diagonal (i==j) allows for subcircuits
        ort_arcs = [(i,j,CPMpy_solver.solver_var(b)) for (i,j),b in np.ndenumerate(arcvars) if not ((i == j) and (i == self.args[1]))] # The start index cannot self loop and thus must be part of the subcircuit.

        return Native_solver.AddCircuit(ort_arcs)
        

# ----------------------------------- Choco ---------------------------------- #

class ChocoSubcircuit(DirectConstraint):
    def __init__(self, arguments):
        super().__init__("chocosubcircuit", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        # Successor variables
        succ = CPMpy_solver.solver_vars(self.args[0])
        # Add an unused variable for the subcircuit length.
        subcircuit_length = CPMpy_solver.solver_var(cp.intvar(0, len(succ)))
        return Native_solver.sub_circuit(succ, 0, subcircuit_length)

# --------------------------------- Minizinc --------------------------------- #

class MinizincSubcircuit(DirectConstraint):
    def __init__(self, arguments):
        super().__init__("minizincsubcircuit", arguments)

    def callSolver(self, CPMpy_solver, Native_solver):
        # minizinc is offset 1, which can be problematic here...
        args_str = ["{}+1".format(CPMpy_solver._convert_expression(e)) for e in self.args[0]]
        return "{}([{}])".format("subcircuit", ",".join(args_str))
        