"""
    Interface to PySDD's API

    PySDD is a knowledge compilation to SDD library... TODO
    https://TODO

    This solver can ONLY be used for solution enumeration over Boolean variables!
    That is, only logical constraints (and,or,xor,implies,==,!=) (and cardinality constraints later).

    Documentation of the solver's own Python API:
    https://TODO


    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pysdd
"""
from functools import reduce
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.utils import is_any_list
from ..transformations.get_variables import get_variables
from ..transformations.to_cnf import to_cnf

class CPM_pysdd(SolverInterface):
    """
    Interface to pysdd's API

    Requires that the 'PySDD' python package is installed:
    $ pip install pysdd

    See detailed installation instructions at:
    https://TODO

    Creates the following attributes (see parent constructor for more):
    pysdd_vtree: a pysdd.sdd.Vtree
    pysdd_manager: a pysdd.sdd.SddManager
    pysdd_root: a pysdd.sdd.SddNode
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            from pysdd.sdd import SddManager
            return True
        except ImportError as e:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        pysdd vtree, manager and (True) root node

        Only supports satisfaction problems and solution enumeration

        Arguments:
        - cpm_model: Model(), a CPMpy Model(), optional
        - subsolver: None
        """
        if not self.supported():
            raise Exception("CPM_pysdd: Install the python 'pysdd' package to use this solver interface")
        if cpm_model and cpm_model.objective_ is not None:
            raise Exception("CPM_pysdd: only satisfaction, does not support an objective function")

        # initialise the native solver object, or at least their existence
        self.pysdd_vtree = None
        self.pysdd_manager = None
        self.pysdd_root = None

        # initialise everything else and post the constraints/objective
        super().__init__(name="pysdd", cpm_model=cpm_model)


    def solve(self, time_limit=None, assumptions=None):
        """
            See if an arbitrary model exists

            This is a knowledge compiler:
                - building it is the (computationally) hard part
                - checking for a solution is trivial
        """
        has_sol = True
        if self.pysdd_root is not None:
            # if root node is false (empty), no solutions
            has_sol = not self.pysdd_root.is_false()

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = 0.0

        # translate exit status
        if has_sol:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE

        # get solution values (of user specified variables only)
        if has_sol and self.pysdd_root is not None:
            sol = next(self.pysdd_root.models())
            # fill in variable values
            for cpm_var in self.user_vars:
                lit = self.solver_var(cpm_var).literal
                if lit in sol:
                    cpm_var._value = bool(sol[lit])
                else:
                    cpm_var._value = None  # not specified...

        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            WARNING: setting 'display' will SIGNIFICANTLY slow down solution counting...

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit, solution_limit, kwargs: not used

            Returns: number of solutions found
        """
        if time_limit is not None:
            raise NotImplementedError("PySDD.solveAll(), time_limit not (yet?) supported")
        if solution_limit is not None:
            raise NotImplementedError("PySDD.solveAll(), solution_limit not (yet?) supported")

        if self.pysdd_root is None:
            return 0

        if display is None:
            # the desired, fast computation
            return self.pysdd_root.model_count()
        else:
            # manually walking over the tree, much slower...
            solution_count = 0
            for sol in self.pysdd_root.models():
                # fill in variable values
                for cpm_var in self.user_vars:
                    lit = self.solver_var(cpm_var).literal
                    if lit in sol:
                        cpm_var._value = bool(sol[lit])
                    else:
                        cpm_var._value = None

                # display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display()  # callback

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            # just a view, get actual var identifier, return -id
            return -self.solver_var(cpm_var._bv)

        # create if it does not exit
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                # make new var, add at end (what is best here??)
                self.pysdd_manager.add_var_after_last()
                n = self.pysdd_manager.var_count()
                revar = self.pysdd_manager.vars[n]
            else:
                raise NotImplementedError(f"CPM_pysdd: variable {cpm_var} not supported")
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]


    def __add__(self, cpm_con):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver

        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver

        For PySDD, it can be beneficial to post a big model (collection of constraints) at once...

        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # add new user vars to the set
        self.user_vars.update(get_variables(cpm_con))

        if self.pysdd_root is None:
            # initialize (arbitrary) vtree from vars
            from pysdd.sdd import SddManager, Vtree

            self.pysdd_vtree = Vtree(var_count=len(self.user_vars), vtree_type="balanced")
            self.pysdd_manager = SddManager.from_vtree(self.pysdd_vtree)
            self.pysdd_root = self.pysdd_manager.true()

        # apply transformations, then post internally
        cpm_cons = to_cnf(cpm_con)
        for con in cpm_cons:
            self._post_constraint(con)

        return self

    def _post_constraint(self, cpm_expr):
        """
            Post a primitive CPMpy constraint to the native solver API
        """
        if isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var
            self.pysdd_root &= self.solver_var(cpm_expr)
        elif isinstance(cpm_expr, Operator):
            if cpm_expr.name == 'or':
                # not sure about this way of making a clause...
                clause = reduce(self.pysdd_manager.disjoin, self.solver_vars(cpm_expr.args))
                self.pysdd_root &= clause
            else:
                raise NotImplementedError(
                    f"Automatic conversion of Operator {cpm_expr} to CNF not yet supported, please report on github.")
        #elif isinstance(cpm_expr, Comparison):
        else:
            raise NotImplementedError(f"Constraint {cpm_expr} not supported by CPM_pysdd")


    def dot(self):
        """
            Returns a graphviz Dot object

            Display (in a notebook) with:
            import graphviz
            graphviz.Source(m.dot())
        """
        if self.pysdd_root is None:
            from pysdd.sdd import SddManager
            SddManager().true().dot()
        return self.pysdd_root.dot()