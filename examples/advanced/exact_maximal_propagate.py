"""
Example showing direct solver access to Exact solver object.

Solver-native implementation of the Maximal-Propagation algorithm as found in cpmpy.tools.maximal_propagate
Can be used for finding the maximal consequence of a set of constraints under assumptions.
"""

from cpmpy import *
from cpmpy.expressions.variables import NegBoolView
from cpmpy.solvers import CPM_exact
from cpmpy.expressions.utils import is_any_list, is_num, flatlist


class PropagationSolver(CPM_exact):

    def __init__(self, cpm_model=None, subsolver=None):
        # first need to set the encoding before adding constraints and variables
        super(PropagationSolver, self).__init__(cpm_model=None, subsolver=subsolver)
        self.encoding = "onehot" # required for use of pruneDomains function

        if cpm_model is not None:
            # post all constraints at once, implemented in __add__()
            self += cpm_model.constraints

            # post objective
            if cpm_model.objective_ is not None:
                if cpm_model.objective_is_min:
                    self.minimize(cpm_model.objective_)
                else:
                    self.maximize(cpm_model.objective_)

    def maximal_propagate(self, assumptions=[], vars=None):
        """
            Wrapper for the `pruneDomains` method of Exact
            Automatically converts CPMpy variables to Exact variables and supports assumptions
            :param: assumptions: list of Boolean variables (or their negation) assumed to be True
            :param: vars: list of variables to propagate
        """
        if self.solver_is_initialized is False:
            self.solve() # this initializes solver and necessary datastructures

        self.xct_solver.clearAssumptions() # clear any old assumptions

        assump_vals = [int(not isinstance(v, NegBoolView)) for v in assumptions]
        assump_vars = [self.solver_var(v._bv if isinstance(v, NegBoolView) else v) for v in assumptions]
        for var, val in zip(assump_vars, assump_vals):
            self.xct_solver.setAssumption(var, [val])

        if vars is None:
            vars = flatlist(self.user_vars)

        domains = self.xct_solver.pruneDomains(flatlist(self.solver_vars(vars)))

        return {var : {int(v) for v in dom} for var, dom in zip(vars, domains)}



if __name__ == "__main__":
    import cpmpy as cp
    from cpmpy.tools.explain.utils import make_assump_model

    x = cp.intvar(1, 5, shape=5, name="x")

    c1 = cp.AllDifferent(x)
    c2 = x[0] == cp.min(x)
    c3 = x[-1] == 1

    model, cons, assump = make_assump_model([c1,c2,c3])

    propsolver = PropagationSolver(model)

    print(f"Propagating constraints {c1} and {c2}:")
    possible_vals = propsolver.maximal_propagate(assumptions=assump[:2])
    for var, values in possible_vals.items():
        print(f"{var}: {sorted(values)}")

    print("\n\n")
    print(f"Propagating constraints {c1}, {c2} and {c3}:")
    possible_vals = propsolver.maximal_propagate(assumptions=assump)
    for var, values in possible_vals.items():
        print(f"{var}: {sorted(values)}")
