#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## pindakaas.py
##
"""
    Interface to Pindakaas (`pkl`) API

    Requires that the `pkl` library python package is installed:

        $ pip install pindakaas

    `pkl` is a library to transform pseudo-Boolean and integer constraints into conjunctive normal form.
    See https://github.com/pindakaashq/pindakaas.

    This solver can be used if the model only has PB constraints.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_pindakaas

    ==============
    Module details
    ==============
"""
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, NegBoolView, boolvar
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.utils import is_int, flatlist
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint
from ..transformations.normalize import toplevel_list, simplify_boolean
from ..transformations.reification import only_implies, only_bv_reifies


class CPM_pindakaas(SolverInterface):
    """
    Interface to TEMPLATE's API

    Requires that the 'TEMPLATEpy' python package is installed:
    $ pip install TEMPLATEpy

    See detailed installation instructions at:
    <URL to detailed solver installation instructions, if any>

    Creates the following attributes (see parent constructor for more):
    - tpl_model: object, TEMPLATE's model object
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            import pindakaas as pkl
            return True
        except ImportError as e:
            return False


    @staticmethod
    def solvernames():
        # """
        #     Returns solvers supported by `pkl` on your system
        # """
        # from pysat.solvers import SolverNames
        # names = []
        # for name, attr in vars(SolverNames).items():
        #     # issue with cryptosat, so we don't include it in our https://github.com/msoos/cryptominisat/issues/765
        #     if not name.startswith('__') and isinstance(attr, tuple) and not name == 'cryptosat':
        #         if name not in attr:
        #             name = attr[-1]
        #         names.append(name)
        # TODO [?] @jip discuss solver interface
        return ["encode"]


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        name="pindakaas"
        if not self.supported():
            raise Exception(f"CPM_{name}: Install the Pindakaas python library `pindakaas` (e.g. `pip install pindakaas`) package to use this solver interface")
        if cpm_model and cpm_model.objective_ is not None:
            raise NotSupportedError(f"CPM_{name}: only satisfaction, does not support an objective function")

        import pindakaas as pkl
        self.pkl_cnf = pkl.CNF()
        # TODO initialize solver

        # initialise everything else and post the constraints/objective
        super().__init__(name=name, cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        raise NotSupportedError(f"{self.name}: sub-solvers not yet supported, encode-only")


    def solve(self, time_limit=None, assumptions=None):
        """
            Call the `pkl` solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            # [GUIDELINE] Please document key solver arguments that the user might wish to change
            #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
            # [GUIDELINE] Add link to documentation of all solver parameters
        """
        print(f"{self.pkl_cnf}")
        raise NotSupportedError(f"{self.name}: TODO solving interface")

        # ensure all vars are known to solver
        # TODO little unclear about interaction of solver/user vars?
        self.solver_vars(list(self.user_vars))

        if assumptions is not None:
            raise NotSupportedError(f"{self.name}: assumptions currently unsupported")
        # if assumptions is None:
        #     pysat_assum_vars = [] # default if no assumptions
        # else:
        #     pysat_assum_vars = self.solver_vars(assumptions)
        #     self.assumption_vars = assumptions

        # import time
        # # set time limit?
        # if time_limit is not None:
        #     from threading import Timer
        #     t = Timer(time_limit, lambda s: s.interrupt(), [self.pysat_solver])
        #     t.start()
        #     my_status = self.pysat_solver.solve_limited(assumptions=pysat_assum_vars, expect_interrupt=True)
        #     # ensure timer is stopped if early stopping
        #     t.cancel()
        #     ## this part cannot be added to timer otherwhise it "interrups" the timeout timer too soon
        #     self.pysat_solver.clear_interrupt()
        # else:
        #     my_status = self.pysat_solver.solve(assumptions=pysat_assum_vars)

        # # new status, translate runtime
        # self.cpm_status = SolverStatus(self.name)
        # self.cpm_status.runtime = self.pysat_solver.time()

        # # translate exit status
        # if my_status is True:
        #     self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        # elif my_status is False:
        #     self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        # elif my_status is None:
        #     # can happen when timeout is reached...
        #     self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        # else:  # another?
        #     raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # # True/False depending on self.cpm_status
        # has_sol = self._solve_return(self.cpm_status)

        # # translate solution values (of user specified variables only)
        # if has_sol:
        #     sol = frozenset(self.pysat_solver.get_model())  # to speed up lookup
        #     # fill in variable values
        #     for cpm_var in self.user_vars:
        #         lit = self.solver_var(cpm_var)
        #         if lit in sol:
        #             cpm_var._value = True
        #         elif -lit in sol:
        #             cpm_var._value = False
        #         else: # not specified, dummy val
        #             cpm_var._value = True

        # return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created

            Transforms cpm_var into CNF literal using self.pkl_cnf
            (positive or negative integer)
        """
        if isinstance(cpm_var, NegBoolView): # negative literal
             # get inner variable and return its negated solver var
            return self.solver_var(cpm_var._bv).negate()
        elif isinstance(cpm_var, _BoolVarImpl): # positive literal
             # insert if new
            if cpm_var not in self._varmap:
                self._varmap[cpm_var.name] = self.pkl_cnf.new_var()
            return self._varmap[cpm_var.name]
        else:
            raise NotImplementedError(f"{self.name}: variable {cpm_var} not supported")


    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons)
        cpm_cons = simplify_boolean(cpm_cons)
        cpm_cons = flatten_constraint(cpm_cons)
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_implies(cpm_cons)
        return cpm_cons

    def __add__(self, cpm_expr_orig):
      """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            What 'supported' means depends on the solver capabilities, and in effect on what transformations
            are applied in `transform()`.

      """

      # add new user vars to the set
      get_variables(cpm_expr_orig, collect=self.user_vars)

      # transform and post the constraints
      for cpm_expr in self.transform(cpm_expr_orig):
        if cpm_expr.name == 'or':
            self.pkl_cnf.add_clause(self.solver_vars(cpm_expr.args))

        elif cpm_expr.name == '->':  # BV -> BE only thanks to only_bv_reifies
            a0,a1 = cpm_expr.args

            # BoolVar() -> BoolVar()
            if isinstance(a1, _BoolVarImpl):
                args = [~a0, a1]
                self.pkl_cnf.add_clause(self.solver_vars(args))
            elif isinstance(a1, Operator) and a1.name == 'or':
                args = [~a0]+a1.args
                self.pkl_cnf.add_clause(self.solver_vars(args))
            elif hasattr(a1, 'decompose'):  # implied global constraint
                self += a0.implies(cpm_expr.decompose())
            elif isinstance(a1, Comparison) and a1.args[0].name == "sum":  # implied sum comparison (a0->sum(bvs)<>val)
                # convert sum to clauses
                sum_clauses = self._pkl_cardinality(a1)
                # implication of conjunction is conjunction of individual implications
                # i.e.: (l1 /\ ... /\ ln) -> a == (l1 -> a) /\ ... /\ (ln -> a)
                nimplvar = [self.solver_var(~a0)]
                clauses = [nimplvar+c for c in sum_clauses]
                self.pysat_solver.append_formula(clauses)

        elif isinstance(cpm_expr, Comparison):
            # only handle cardinality encodings (for now)
            if isinstance(cpm_expr.args[0], Operator) and cpm_expr.args[0].name == "sum":
                # convert to clauses and post
                clauses = self._pkl_cardinality(cpm_expr)
                self.pysat_solver.append_formula(clauses)
            else:
                raise NotImplementedError(f"Non-operator constraint {cpm_expr} not supported by CPM_pysat")

        elif isinstance(cpm_expr, BoolVal):
            # base case: Boolean value
            if cpm_expr.args[0] is False:
                self.pkl_cnf.add_clause([])

        elif isinstance(cpm_expr, _BoolVarImpl):
            # base case, just var or ~var
            self.pkl_cnf.add_clause([self.solver_var(cpm_expr)])

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            raise NotImplementedError(f"TODO")
            cpm_expr.callSolver(self, self.pysat_solver)

        else:
            raise NotImplementedError(f"{self.name}: Non supported constraint {cpm_expr}")

      return self

    def _pkl_cardinality(self, cpm_compsum):
        """ convert CPMpy comparison of sum into PySAT list of clauses """
        raise NotImplementedError(f"TODO add adder encoding")
        # we assume transformations are applied such that the below is true
        if not isinstance(cpm_compsum, Comparison):
            raise NotSupportedError(f"PySAT card: input constraint must be Comparison -- {cpm_compsum}")
        if not is_int(cpm_compsum.args[1]):
            raise NotSupportedError(f"PySAT card: sum must have constant at rhs not {cpm_compsum.args[1]} -- {cpm_compsum}")
        if not cpm_compsum.args[0].name == "sum":
            raise NotSupportedError(f"PySAT card: input constraint must be sum, got {cpm_compsum.args[0].name} -- {cpm_compsum}")
        if not (all(isinstance(v, _BoolVarImpl) for v in cpm_compsum.args[0].args)):
            raise NotSupportedError(f"PySAT card: sum must be over Boolvars only -- {cpm_compsum.args[0]}")

        from pysat.card import CardEnc

        lits = self.solver_vars(cpm_compsum.args[0].args)
        bound = cpm_compsum.args[1]

        if cpm_compsum.name == "<":
            return CardEnc.atmost(lits=lits, bound=bound-1, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == "<=":
            return CardEnc.atmost(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == ">=":
            return CardEnc.atleast(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == ">":
            return CardEnc.atleast(lits=lits, bound=bound+1, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == "==":
            return CardEnc.equals(lits=lits, bound=bound, vpool=self.pysat_vpool).clauses
        elif cpm_compsum.name == "!=":
            # special cases with bounding 'hardcoded' for clarity
            if bound <= 0:
                return CardEnc.atleast(lits=lits, bound=bound+1, vpool=self.pysat_vpool).clauses
            elif bound >= len(lits):
                return CardEnc.atmost(lits=lits, bound=bound-1, vpool=self.pysat_vpool).clauses
            else:
                ## add implication literals for (strict) atleast/atmost, one must be true
                is_atleast = self.solver_var(boolvar())
                is_atmost = self.solver_var(boolvar())
                clauses = [[is_atleast, is_atmost]]
                clauses += [atl + [-is_atleast] for atl in
                            CardEnc.atleast(lits=lits, bound=bound+1, vpool=self.pysat_vpool).clauses]
                clauses += [atm + [-is_atmost] for atm in
                            CardEnc.atmost(lits=lits, bound=bound-1, vpool=self.pysat_vpool).clauses]
                return clauses

        raise NotImplementedError(f"Non-operator constraint {cpm_compsum} not supported by CPM_pysat")
