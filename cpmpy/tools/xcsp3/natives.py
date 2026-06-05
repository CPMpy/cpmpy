"""
Solver-native constraint substitution for XCSP3.

After parsing an XCSP3 instance into a CPMpy model, certain global constraints
have more efficient solver-native implementations than the generic CPMpy
decomposition.  This module maps constraint names to their native equivalents
per solver and applies the substitution in-place on the model's constraint list.
"""

from cpmpy.tools.xcsp3 import globals as xcsp3_globals

# Map: solver → {constraint_name → native_constructor}
SOLVER_NATIVE_MAP = {
    "ortools": {
        "no_overlap2d": xcsp3_globals.OrtNoOverlap2D,
        "subcircuit": xcsp3_globals.OrtSubcircuit,
        "subcircuitwithstart": lambda args: xcsp3_globals.OrtSubcircuitWithStart(args[:-1], args[-1]),
    },
    "choco": {
        "subcircuit": xcsp3_globals.ChocoSubcircuit,
    },
    "minizinc": {
        "subcircuit": xcsp3_globals.MinizincSubcircuit,
        "subcircuitwithstart": xcsp3_globals.MinizincSubcircuitWithStart,
    },
}


def apply_solver_native_constraints(model, solver: str) -> None:
    """
    Replace supported global constraints in *model* with solver-native equivalents.

    Modifies ``model.constraints`` in-place.  Raises on any substitution error
    so that the caller gets a hard failure rather than silent wrong behaviour.
    """
    native_map = SOLVER_NATIVE_MAP.get(solver, {})
    if not native_map:
        return
    for i, constraint in enumerate(model.constraints):
        if hasattr(constraint, "name") and constraint.name in native_map:
            model.constraints[i] = native_map[constraint.name](constraint.args)
