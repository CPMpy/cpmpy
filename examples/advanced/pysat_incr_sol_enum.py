"""
Example solution enumeration for incremental/repeated pysat solve calls.

"""

from cpmpy import *
from cpmpy.solvers.pysat import CPM_pysat


def enum_models(solver, assumptions=None, solution_limit=None, solution_callback=None, **kwargs):
    """
        Call the PySAT solver and repeatedly enumerate models until no more model
        exist.

        A new indicator boolean variable is introduced in order to block model, 
        when we are done with finding new non-intersecting models we ensure the models 
        do not interfere with future solve calls by adding its negation making these models
        always true.

        Arguments:
        - time_limit:  maximum solve time in seconds (float, optional)
        - assumptions: list of CPMpy Boolean variables that are assumed to be true.
                        For use with s.get_core(): if the model is UNSAT, get_core() returns a small subset of assumption variables that are unsat together.
                        Note: the PySAT interface is statefull, so you can incrementally call solve() with assumptions and it will reuse learned clauses
    """
    bi = boolvar(name="pysat-blocking-var")
    solution_count = 0
    if not assumptions:
        assumptions = []

    while(solver.solve(assumptions=assumptions + [bi], **kwargs)):

        if solution_callback:
            solution_callback()

        # count and stop
        solution_count += 1
        if solution_count == solution_limit:
            break

        ### This does not work ...
        # solver += bi.implies(any(v != v.value() for v in solver.user_vars))
        solver += bi.implies(any([~v if v.value() else v for v in solver.user_vars] ))

    solver += (~bi)
    return solution_count

if __name__ == "__main__":
    # Construct the model.
    (mayo, ketchup, curry) = boolvar(3)
    model = Model(
        (~mayo | ketchup),
        (~ketchup| curry)
    )
    ps2 = CPM_pysat(model)

    enum_models(
        ps2, 
        solution_callback=lambda: print(f"mayo={mayo.value()}, ketchup={ketchup.value()}, curry={curry.value()}")
    )