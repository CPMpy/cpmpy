"""
    Interface to PySDD's API

    PySDD is a knowledge compilation package for Sentential Decision Diagrams (SDD)
    https://pysdd.readthedocs.io/en/latest/

    This solver can ONLY be used for solution checking and enumeration over Boolean variables!
    That is, only logical constraints (and,or,implies,==,!=) and Boolean global constraints.

    Documentation of the solver's own Python API:
    https://pysdd.readthedocs.io/en/latest/classes/SddManager.html


    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pysdd
"""
from functools import reduce
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import is_any_list, is_bool
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list, simplify_boolean

class CPM_pysdd(SolverInterface):
    """
    Interface to pysdd's API

    Requires that the 'PySDD' python package is installed:
    $ pip install pysdd

    See detailed installation instructions at:
    https://pysdd.readthedocs.io/en/latest/usage/installation.html

    Creates the following attributes (see parent constructor for more):
        - pysdd_vtree: a pysdd.sdd.Vtree
        - pysdd_manager: a pysdd.sdd.SddManager
        - pysdd_root: a pysdd.sdd.SddNode (changes whenever a formula is added)

    The `DirectConstraint`, when used, calls a function on the `pysdd_manager` object and replaces the root node with a conjunction of the previous root node and the result of this function call.
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
            raise NotSupportedError("CPM_pysdd: only satisfaction, does not support an objective function")

        # these will be loaded once a first formula is added
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
                - checking for a solution is trivial after that
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

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            WARNING: setting 'display' will SIGNIFICANTLY slow down solution counting...

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit, solution_limit, kwargs: not used
                - call_from_model: whether the method is called from a CPMpy Model instance or not

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
                solution_count += 1
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
            return solution_count

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
        """
        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            # just a view, get actual var identifier, return -id
            return -self.solver_var(cpm_var._bv)

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                # make new var, add at end (what is best here??)
                self.pysdd_manager.add_var_after_last()
                n = self.pysdd_manager.var_count()
                revar = self.pysdd_manager.vars[n]
            else:
                raise NotImplementedError(f"CPM_pysdd: non-Boolean variable {cpm_var} not supported")
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

            For PySDD, it can be beneficial to add a big model (collection of constraints) at once...

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # works on list of nested expressions
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons,supported={'xor'}, supported_reified={'xor'}) #keep unsupported xor for error message purposes.
        cpm_cons = simplify_boolean(cpm_cons)  # for cleaning (BE >= 0) and such
        return cpm_cons

    def __add__(self, cpm_expr):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """

        newvars = get_variables(cpm_expr)

        # check only Boolean variables
        # XXX a bit redundant, `solver_var()` already does this too
        for v in newvars:
            if not isinstance(v, _BoolVarImpl):
                raise NotSupportedError(f"CPM_pysdd: only Boolean variables allowed -- {type(v)}: {v}")
        # add new user vars to the set
        self.user_vars |= set(newvars)

        # if needed initialize (arbitrary) vtree from all user-specified vars
        # we waited till here to already have some vars... beneficial?
        if self.pysdd_root is None:
            from pysdd.sdd import SddManager, Vtree

            cnt = len(self.user_vars)
            if cnt == 0:
                cnt = 1  # otherwise segfault
            self.pysdd_vtree = Vtree(var_count=cnt, vtree_type="balanced")
            self.pysdd_manager = SddManager.from_vtree(self.pysdd_vtree)
            self.pysdd_root = self.pysdd_manager.true()

        # transform and post the constraints
        # XXX the order in the for loop will matter on runtime efficiency...
        for cpm_con in self.transform(cpm_expr):
            # replace root by conjunction of itself and the con expression
            self.pysdd_root = self.pysdd_manager.conjoin(self.pysdd_root,
                                                self._pysdd_expr(cpm_con))

        return self

    def _pysdd_expr(self, cpm_con):
        """
            PySDD supports nested expressions: each expression
            (variable or subexpression) is a node...
            so we recursively translate our expressions to theirs.

            input: Expression or const
            output: pysdd Node
        """
        if isinstance(cpm_con, _BoolVarImpl):
            # base case, just var or ~var
            return self.solver_var(cpm_con)

        elif is_bool(cpm_con) or isinstance(cpm_con, BoolVal):
            # base case: Boolean value
            if cpm_con:
                return self.pysdd_manager.true()
            else:
                return self.pysdd_manager.false()

        elif not isinstance(cpm_con, Expression):
            # a number or so
            raise NotImplementedError(f"CPM_pysdd: Non supported object {cpm_con}")

        elif cpm_con.name == 'and':
            # conjoin the nodes corresponding to the args
            # also here the order might matter on runtime efficiency...
            return reduce(self.pysdd_manager.conjoin, [self._pysdd_expr(a) for a in cpm_con.args])

        elif cpm_con.name == 'or':
            # disjoin the nodes corresponding to the args
            # also here the order might matter on runtime efficiency...
            return reduce(self.pysdd_manager.disjoin, [self._pysdd_expr(a) for a in cpm_con.args])

        elif cpm_con.name == 'not':
            return self.pysdd_manager.negate(self._pysdd_expr(cpm_con.args[0]))

        elif cpm_con.name == '->':
            a0 = self._pysdd_expr(cpm_con.args[0])
            a1 = self._pysdd_expr(cpm_con.args[1])
            # ~a0 | a1
            return self.pysdd_manager.disjoin(self.pysdd_manager.negate(a0), a1)

        elif cpm_con.name == '==':
            a0 = self._pysdd_expr(cpm_con.args[0])
            a1 = self._pysdd_expr(cpm_con.args[1])
            # (~a0 | a1) & (~a1 | a0)
            return self.pysdd_manager.conjoin(
                        self.pysdd_manager.disjoin(self.pysdd_manager.negate(a0), a1),
                        self.pysdd_manager.disjoin(self.pysdd_manager.negate(a1), a0),
                   )

        elif cpm_con.name == '!=':
            # ~(a0 == a1)
            equiv = self._pysdd_expr(cpm_con.args[0] == cpm_con.args[1])
            return self.pysdd_manager.negate(equiv)

        # a direct constraint, call on manager
        # WARNING: will only work when all args are variables or constants!
        # if unwanted, repeated some of the logic of callSolver here
        elif isinstance(cpm_con, DirectConstraint):
            return cpm_con.callSolver(self, self.pysdd_manager)

        else:
            raise NotImplementedError(f"CPM_pysdd: Non supported constraint {cpm_con}")

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
