#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pysdd.py
##
"""
    Interface to PySDD's API

    PySDD is a knowledge compilation package for Sentential Decision Diagrams (SDD).
    (see https://pysdd.readthedocs.io/en/latest/)

    .. warning::    
        This solver can ONLY be used for solution checking and enumeration over Boolean variables!
        It does not support optimization.

    Always use :func:`cp.SolverLookup.get("pysdd") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'PySDD' python package is installed:

    .. code-block:: console

        $ pip install PySDD

    See detailed installation instructions at:
    https://pysdd.readthedocs.io/en/latest/usage/installation.html

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pysdd

    ==============
    Module details
    ==============
"""
from functools import reduce
from typing import Optional, List

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, BoolVal
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import is_bool, argval, argvals
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list, simplify_boolean
from ..transformations.safening import no_partial_functions


class CPM_pysdd(SolverInterface):
    """
    Interface to PySDD's API.

    Creates the following attributes (see parent constructor for more):

    - ``pysdd_vtree`` : a pysdd.sdd.Vtree
    - ``pysdd_manager`` : a pysdd.sdd.SddManager
    - ``pysdd_root`` : a pysdd.sdd.SddNode (changes whenever a formula is added)

    The :class:`~cpmpy.expressions.globalconstraints.DirectConstraint`, when used, calls a function on the ``pysdd_manager`` object and replaces the root node with a conjunction of the previous root node and the result of this function call.

    Documentation of the solver's own Python API:
    https://pysdd.readthedocs.io/en/latest/classes/SddManager.html
    """

    supported_global_constraints = frozenset({"xor"})
    supported_reified_global_constraints = frozenset({"xor"})

    @staticmethod
    def supported():
        # try to import the package
        try:
            from pysdd.sdd import SddManager
            return True
        except ModuleNotFoundError:
            return False
        except Exception as e:
            raise e
        
    @staticmethod
    def version() -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version('pysdd')
        except PackageNotFoundError:
            return None


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Requires a CPMpy model as input, and will create the corresponding
        pysdd vtree, manager and (True) root node

        Only supports satisfaction problems and solution enumeration

        Arguments:
            cpm_model: Model(), a CPMpy Model(), optional
            subsolver: None
        """
        if not self.supported():
            raise Exception("CPM_pysdd: Install the python package 'pysdd' to use this solver interface")
        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError("CPM_pysdd: only satisfaction, does not support an objective function")

        from pysdd.sdd import SddManager, Vtree

        cnt = 1
        self.pysdd_vtree = Vtree(var_count=cnt, vtree_type="balanced")
        self.pysdd_manager = SddManager.from_vtree(self.pysdd_vtree)
        self.pysdd_root = self.pysdd_manager.true()

        # initialise everything else and post the constraints/objective
        super().__init__(name="pysdd", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.pysdd_root

    def solve(self, time_limit:Optional[float]=None, assumptions:Optional[List[_BoolVarImpl]]=None):
        """
            See if an arbitrary model exists

            This is a knowledge compiler:

            - building it is the (computationally) hard part
            - checking for a solution is trivial after that
        """

        if time_limit is not None:
            raise NotImplementedError("PySDD.solve(), time_limit not (yet?) supported")
        
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(boolvar() == True)

        has_sol = True
        if self.pysdd_root is not None:
            # if root node is false (empty), no solutions
            has_sol = not self.pysdd_root.is_false()

        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = 0.0

        # translate exit status
        if has_sol:
            # Only CSP (does not support COP)
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            for cpm_var in self.user_vars:
                cpm_var._value = None

        # get solution values (of user specified variables only)
        if has_sol and self.pysdd_root is not None:
            sol = next(self.pysdd_root.models())
            # fill in variable values
            for cpm_var in self.user_vars:
                lit = self.solver_var(cpm_var).literal
                if lit in sol:
                    cpm_var._value = bool(sol[lit])
                else:
                    cpm_var._value = cpm_var.get_bounds()[0] # dummy value - TODO: ensure Pysdd assigns an actual value
                    # cpm_var._value = None  # not specified...

        return has_sol

    def solveAll(self, display:Optional[Callback]=None, time_limit:Optional[float]=None, solution_limit:Optional[int]=None, call_from_model=False, **kwargs):
        """
            Compute all solutions and optionally display the solutions.

            .. warning::
                WARNING: setting 'display' will SIGNIFICANTLY slow down solution counting...

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit, solution_limit, kwargs: not used
                - call_from_model: whether the method is called from a CPMpy Model instance or not

            Returns: 
                number of solutions found            
        """
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

        # edge case, empty model, ensure the solver has something to solve
        if not len(self.user_vars):
            self.add(boolvar() == True)

        if time_limit is not None:
            raise NotImplementedError("PySDD.solveAll(), time_limit not (yet?) supported")
        if solution_limit is not None:
            raise NotImplementedError("PySDD.solveAll(), solution_limit not (yet?) supported")

        if self.pysdd_root is None or self.pysdd_root.is_false():
            # clear user vars if no solution found
            for var in self.user_vars:
                var._value = None
            return 0

        sddmodels = [x for x in self.pysdd_root.models()]
        if len(sddmodels) != self.pysdd_root.model_count:
            #pysdd doesn't always have correct solution count..
            projected_sols = set()
            for sol in sddmodels:
                projectedsol = []
                for cpm_var in self.user_vars:
                    lit = self.solver_var(cpm_var).literal
                    projectedsol.append(bool(sol[lit]))
                projected_sols.add(tuple(projectedsol))
        else:
            projected_sols = set(sddmodels)

        if projected_sols:
            if projected_sols == solution_limit:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            else:
                # time limit not (yet) supported -> always all solutions found
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        else:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE

        # display if needed
        if display is not None:
            # manually walking over the tree, much slower...
            for sol in projected_sols:
                # fill in variable values
                for i, cpm_var in enumerate(self.user_vars):
                    cpm_var._value = sol[i]

                if isinstance(display, Expression):
                    print(argval(display))
                elif isinstance(display, list):
                    print(argvals(display))
                else:
                    display()  # callback
        
        return len(projected_sols)

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

            See the ':ref:`Adding a new solver` docs on readthedocs for more information.

            For PySDD, it can be beneficial to add a big model (collection of constraints) at once...

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: list of Expression
        """
        # works on list of nested expressions
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons, safen_toplevel={"div", "mod", "element"})
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints,
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        cpm_cons = simplify_boolean(cpm_cons)  # for cleaning (BE >= 0) and such
        return cpm_cons

    def add(self, cpm_expr):
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

        # add new user vars to the set
        self.user_vars |= set(newvars)

        # transform and post the constraints
        # XXX the order in the for loop will matter on runtime efficiency...
        for cpm_con in self.transform(cpm_expr):
            # replace root by conjunction of itself and the con expression
            self.pysdd_root = self.pysdd_manager.conjoin(self.pysdd_root,
                                                self._pysdd_expr(cpm_con))

        return self
    __add__ = add  # avoid redirect in superclass

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

            .. code-block:: python

                import graphviz
                graphviz.Source(m.dot())
        """
        if self.pysdd_root is None:
            from pysdd.sdd import SddManager
            SddManager().true().dot()
        return self.pysdd_root.dot()
