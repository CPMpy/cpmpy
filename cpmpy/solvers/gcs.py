#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## gcs.py
##
"""
    Interface to the Glasgow Constraint Solver's API for the CPMpy library.

    See:
    https://github.com/ciaranm/glasgow-constraint-solver

    The key feature of this CP solver is the ability to produce proof logs.

    Always use :func:`cp.SolverLookup.get("gcs") <cpmpy.solvers.utils.SolverLookup.get>` to instantiate the solver object.

    ============
    Installation
    ============

    Requires that the 'gcspy' python package is installed:

    .. code-block:: console

        $ pip install gcspy

    Source installation instructions:

    - Ensure you have C++20 compiler such as GCC 10.3  / clang 15
    - (on Debian-based systems, see https://apt.llvm.org for easy installation)
    - If necessary ``export CXX=<your up to date C++ compiler (e.g. clang++-15)>``
    - Ensure you have Boost installed
    - ``git clone https://github.com/ciaranm/glasgow-constraint-solver.git``
    - ``cd glasgow-constraint-solver/python``
    - ``pip install .``

    .. note::
        If for any reason you need to retry the build, ensure you remove glasgow-constraints-solver/generator before rebuilding.

    For the verifier functionality, the 'veripb' tool is also required.
    See https://gitlab.com/MIAOresearch/software/VeriPB#installation for installation instructions of veripb. 

    The rest of this documentation is for advanced users.

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_gcs
"""
import warnings
from typing import Optional

from packaging.version import Version

from cpmpy.transformations.comparison import only_numexpr_equality
from cpmpy.transformations.reification import reify_rewrite, only_bv_reifies
from ..exceptions import NotSupportedError, GCSVerificationException
from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, _IntVarImpl, _NumVarImpl, NegBoolView, boolvar, intvar
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_num, argval, argvals
from ..transformations.decompose_global import decompose_in_tree, decompose_objective
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, get_or_make_var
from ..transformations.safening import no_partial_functions

from ..transformations.normalize import toplevel_list

# For proof file handling and verifying
import sys
from os import path
from shutil import which
import subprocess

class CPM_gcs(SolverInterface):
    """
    Interface to Glasgow Constraint Solver's API.

    Creates the following attributes (see parent constructor for more):

    - ``gcs`` : the gcspy solver object
    - ``objective_var`` : optional: the variable used as objective
    - ``proof_location`` : location of the last proof produced by the solver
    - ``proof_name`` : name of the last proof (means <proof_name>.opb and <proof_name>.pbp will be present at the proof location)
    - ``veripb_return_code`` : return code from the last VeriPB check.
    - ``proof_check_timeout`` : whether the last VeriPB check timed out.

    Documentation of the solver's own Python API is sparse, but example usage can be found at:
    https://github.com/ciaranm/glasgow-constraint-solver/blob/main/python/python_test.py
    """

    supported_global_constraints = frozenset({"alldifferent", "table", "negative_table", "inverse", "circuit", "xor",
                                              "min", "max", "abs", "div", "mod", "pow", "element", "count", "nvalue"})
    supported_reified_global_constraints = frozenset()

    @staticmethod
    def supported():
        # try to import the package
        try:
            import gcspy
            gcs_version = CPM_gcs.version()
            if Version(gcs_version) < Version("0.1.8"):
                warnings.warn(f"CPMpy requires GCS version >=0.1.8 but you have version "
                              f"{gcs_version}, beware exact>=2.1.0 requires Python 3.10 or higher.")
                return False
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
            return version('gcspy')
        except PackageNotFoundError:
            return None

    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: None (not supported)
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_gcs: Install the python package 'cpmpy[gcs]' to use this solver interface.")

        import gcspy

        assert(subsolver is None) # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        self.gcs = gcspy.GCS()
        self.objective_var = None
        self.proof_location = None
        self.proof_name = None
        self.proof_check_timeout = True
        self.veripb_return_code = False

        # initialise everything else and post the constraints/objective
        super().__init__(name="Glasgow Constraint Solver", cpm_model=cpm_model)

    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).
        """
        return self.gcs
    
    def has_objective(self):
        return self.objective_var is not None
    
    def solve(self, time_limit:Optional[float]=None, prove=False, proof_name:Optional[str]=None, proof_location:Optional[str]=".",
              verify=False, verify_time_limit=None, veripb_args = [], display_verifier_output=True, **kwargs):
        """
            Run the Glasgow Constraint Solver, get just one (optimal) solution.

            Arguments:
                time_limit (float, optional):   maximum solve time in seconds.
                prove:                          whether to produce a VeriPB proof (.opb model file and .pbp proof file).
                proof_name:                     name for the the proof files.
                proof_location:                 location for the proof files (default to current working directory).
                verify:                         whether to verify the result of the solve run (overrides prove if prove is False)
                verify_time_limit:              time limit for verification (ignored if verify=False) 
                veripb_args:                    list of command line arguments to pass to veripb e.g. ``--trace --useColor`` (run ``veripb --help`` for a full list)
                display_verifier_output:        whether to print the output from VeriPB
                **kwargs:                       currently GCS does not support any additional keyword arguments.

            Returns: 
                whether a solution was found.
        """
        # ensure all user vars are known to solver
        self.solver_vars(list(self.user_vars))
        
        # If we're verifying we must be proving
        prove |= verify
        # Set default proof name to name of file containing __main__
        if prove and proof_name is None:
            if hasattr(sys.modules['__main__'], "__file__"):
                self.proof_name = path.splitext(path.basename(sys.modules['__main__'].__file__))[0]
            else:
                self.proof_name = "gcs_proof"
        else:
            self.proof_name = proof_name
        self.proof_location = proof_location
     
        # set time limit
        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be positive")
                 
        # call the solver, with parameters    
        self.gcs_result = self.gcs.solve(
            all_solutions=self.has_objective(), 
            timeout=time_limit,
            callback=None,
            prove=prove,
            proof_name=self.proof_name,
            proof_location=proof_location,
            **kwargs)

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.gcs_result["solve_time"]

        # translate exit status
        if self.gcs_result['solutions'] != 0:
            if self.gcs_result['completed'] and self.has_objective():
                self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            else:
                self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif not self.gcs_result['completed']:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if isinstance(cpm_var, _BoolVarImpl):
                    # Convert back to bool
                    cpm_var._value = bool(self.gcs.get_solution_value(sol_var, self.gcs_result['solutions']-1))
                else:
                    cpm_var._value = self.gcs.get_solution_value(sol_var, self.gcs_result['solutions']-1)

            # translate objective, for optimisation problems only
            if self.has_objective():
                self.objective_value_ = self.gcs.get_solution_value(self.solver_var(self.objective_var))

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var._value = None

        # Verify proof, if requested
        if verify:

            # set time limit
            if verify_time_limit is not None:
                if verify_time_limit <= 0:
                    raise ValueError("Time limit for verifying must be positive")

            self.verify(name=self.proof_name, location=proof_location, time_limit=verify_time_limit,
                        veripb_args=veripb_args, display_output=display_verifier_output)
            
            if self.veripb_return_code > 0:
                raise GCSVerificationException("Glasgow Constraint Solver: Proof failed to verify.")
            
        return has_sol

    def solveAll(self, time_limit:Optional[float]=None, display:Optional[Callback]=None, solution_limit:Optional[int]=None, call_from_model=False,
                 prove=False, proof_name:Optional[str]=None, proof_location:Optional[str]=".",
                 verify=False, verify_time_limit=None, veripb_args = [], display_verifier_output=True, **kwargs):
        """
            Run the Glasgow Constraint Solver, and get a number of solutions, with optional solution callbacks. 

            Arguments:
                display:                        either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                                                default/None: nothing displayed
                solution_limit:                 stop after this many solutions (default: None)
                time_limit (float, optional):   maximum solve time in seconds (default: None)
                call_from_model:                whether the method is called from a CPMpy Model instance or not
                prove:                          whether to produce a VeriPB proof (.opb model file and .pbp proof file).
                proof_name:                     name for the the proof files.
                proof_location:                 location for the proof files (default to current working directory).
                verify:                         whether to verify the result of the solve run (overrides prove if prove is False)
                verify_time_limit:              time limit for verification (ignored if verify=False) 
                veripb_args:                    list of command line arguments to pass to veripb e.g. ``--trace --useColor`` (run ``veripb --help`` for a full list)
                display_verifier_output:        whether to print the output from VeriPB
                **kwargs:                       currently GCS does not support any additional keyword arguments.

            Returns: 
                number of solutions found
        """
        if self.has_objective():
            raise NotSupportedError("Glasgow Constraint Solver: does not support finding all optimal solutions.")
        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))

         # If we're verifying we must be proving
        prove |= verify
        # Set default proof name to name of file containing __main__
        if prove and proof_name is None:
            if hasattr(sys.modules['__main__'], "__file__"):
                self.proof_name = path.splitext(path.basename(sys.modules['__main__'].__file__))[0]
            else:
                self.proof_name = "gcs_proof"
        self.proof_location = proof_location

        # Set display callback
        def display_callback(solution_map):
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                if isinstance(cpm_var, _BoolVarImpl):
                    # Convert back to bool
                    cpm_var._value = bool(solution_map[sol_var])
                else:
                    cpm_var._value = solution_map[sol_var]

            if isinstance(display, Expression):
                print(argval(display))
            elif isinstance(display, list):
                # explicit list of expressions to display
                print(argvals(display))
            elif callable(display):
                display()
            else:
                raise NotImplementedError("Glasgow Constraint Solver: Unknown display type.".format(cpm_var))
            return 
        sol_callback = None
        if display:
            sol_callback=display_callback

        self.gcs_result = self.gcs.solve(
            all_solutions=True, 
            timeout=time_limit, 
            solution_limit=solution_limit, 
            callback=sol_callback, 
            prove=prove, 
            proof_name=proof_name, 
            proof_location=proof_location, **kwargs)

        # new status, get runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.gcs_result["solve_time"]

        num_sols = self.gcs_result["solutions"]
        if self.gcs_result["completed"] and num_sols >= 1:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
        elif self.gcs_result["completed"] and num_sols == 0:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif num_sols >= 1:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        else: # maybe unsat, maybe not (maybe a timeout)
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        # clear user vars if no solution found
        if self._solve_return(self.cpm_status, self.objective_value_) is False:
            for var in self.user_vars:
                var._value = None

        # Verify proof, if requested
        if verify:
            self.verify(name=self.proof_name, location=proof_location, time_limit=verify_time_limit, 
                        veripb_args=veripb_args, display_output=display_verifier_output)

        return num_sols

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return self.gcs.create_integer_constant(cpm_var)

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            # gcs only works with integer variables, so not(x) = -x + 1
            return self.gcs.add_constant(self.gcs.negate(self.solver_var(cpm_var._bv)), 1)

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                # Bool vars are just int vars with [0, 1] domain
                revar = self.gcs.create_integer_variable(0, 1, str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.gcs.create_integer_variable(cpm_var.lb, cpm_var.ub, str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        return self._varmap[cpm_var]

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize.

            ``objective()`` can be called multiple times, only the last one is stored.

            .. note::
                technical side note: any constraints created during conversion of the objective
                are permanently posted to the solver
        """

        # save variables
        get_variables(expr, collect=self.user_vars)

        # transform objective
        obj, decomp_cons = decompose_objective(expr,
                                               supported=self.supported_global_constraints,
                                               supported_reified=self.supported_reified_global_constraints,
                                               csemap=self._csemap)
        obj_var, obj_cons = get_or_make_var(obj) # do not pass csemap here, we will still transform obj_var == obj...
        self.add(decomp_cons + obj_cons)

        self.objective_var = obj_var

        if minimize:
            self.gcs.minimise(self.solver_var(obj_var))
        else:
            self.gcs.maximise(self.solver_var(obj_var))

    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the :ref:`Adding a new solver` docs on readthedocs for more information.

            :param cpm_expr: CPMpy expression, or list thereof
            :type cpm_expr: Expression or list of Expression

            :return: list of Expression
        """
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = no_partial_functions(cpm_cons)
        cpm_cons = decompose_in_tree(cpm_cons,
                                     supported=self.supported_global_constraints,
                                     supported_reified=self.supported_reified_global_constraints,
                                     csemap=self._csemap)
        cpm_cons = flatten_constraint(cpm_cons, csemap=self._csemap)  # flat normal form

        # NB: GCS supports full reification for linear equality and linear inequaltiy constraints
        # but no reification for linear not equals and not half reification for linear equality. 
        # Maybe a future transformation (or future work on the GCS solver).
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['==']), csemap=self._csemap)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum"]), csemap=self._csemap)  # supports >, <, !=

        # NB: GCS supports a small number of simple expressions as the reifying term
        # e.g. (x > 3) -> constraint could in principle be supported in the future.
        cpm_cons = only_bv_reifies(cpm_cons, csemap=self._csemap)
        str_rep = ""
        for c in cpm_cons:
            str_rep += str(c) + '\n'
        return cpm_cons

    def verify(self, name=None, location=".", time_limit=None, display_output=False, veripb_args=[]):
        """
        Verify a solver-produced proof using VeriPB.

        Requires that the 'veripb' tool is installed and on system path. 
        See https://gitlab.com/MIAOresearch/software/VeriPB#installation for installation instructions. 

        Arguments:
            - name:             name for the the proof files (default to self.proof_name)
            - location:         location for the proof files (default to current working directory).
            - time_limit:       time limit for verification (ignored if verify=False) 
            - veripb_args:      list of command line arguments to pass to veripb e.g. ``--trace --useColor`` (run ``veripb --help`` for a full list)
            - display_output:   whether to print the output from VeriPB
        """
        if not which("veripb"):
            raise Exception("Unable to run VeriPB: check it is installed and on system path - see https://gitlab.com/MIAOresearch/software/VeriPB#installation.")

        if name is None:
            name = self.proof_name
        
        if name is None: # Still None?
            raise ValueError("No proof to verify")
        
        if not isinstance(veripb_args, list):
            raise ValueError("veripb_args should be a list")
        
        opb_file = path.join(location, name +".opb")
        pbp_file = path.join(location, name +".pbp")

        if not path.isfile(opb_file):
            raise FileNotFoundError("Can't find " + opb_file)
        if not path.isfile(pbp_file):
            raise FileNotFoundError("Can't find " + pbp_file)
        
        try:
            result = subprocess.run(["veripb"] + veripb_args + [opb_file, pbp_file], 
                                    capture_output=True, text=True, timeout=time_limit)
            self.proof_check_timeout = False
            self.veripb_return_code = result.returncode
            if display_output:
                print(result.stdout)
                print(result.stderr)
        except subprocess.TimeoutExpired:
            self.proof_check_timeout = True
            self.veripb_return_code = 0

        return self.veripb_return_code
    
    def add(self, cpm_cons):
        """
        Post a (list of) CPMpy constraints(=expressions) to the solver
        Note that we don't store the constraints in a cpm_model,
        we first transform the constraints into primitive constraints,
        then post those primitive constraints directly to the native solver
        :param cpm_con CPMpy constraint, or list thereof
        :type cpm_con (list of) Expression(s)
        """
        # add new user vars to the set
                # add new user vars to the set
        get_variables(cpm_cons, collect=self.user_vars)

        for con in self.transform(cpm_cons):
            cpm_expr = con
            if isinstance(cpm_expr, _BoolVarImpl):
                # base case, just var or ~var
                self.gcs.post_or([self.solver_var(cpm_expr)])
            elif isinstance(cpm_expr, BoolVal):
                if not cpm_expr:
                    # bit a hack, empty clause does not work (issue #73 on gcs github)
                    a = boolvar()
                    self.gcs.post_and(self.solver_vars([a,~a]))
            elif isinstance(cpm_expr, Operator) or \
                (cpm_expr.name == '==' and isinstance(cpm_expr.args[0], _BoolVarImpl) \
                and not isinstance(cpm_expr.args[1], _NumVarImpl)): 
                # ^ Somewhat awkward, but want to deal with full and half reifications
                # in one go here, and then deal with regular == comparisons later.l

                # 'and'/n, 'or'/n, '->'/2
                if cpm_expr.name == 'and':
                    self.gcs.post_and(self.solver_vars(cpm_expr.args))
                elif cpm_expr.name == 'or':
                    self.gcs.post_or(self.solver_vars(cpm_expr.args))

                # Reified constraint: BoolVar -> Boolexpr or BoolVar == Boolexpr
                # LHS must be boolvar due to only_bv_reifies
                elif cpm_expr.name == '->' or cpm_expr.name == '==':
                    fully_reify = (cpm_expr.name == '==')
                    assert(isinstance(cpm_expr.args[0], _BoolVarImpl))  
                    bool_lhs = cpm_expr.args[0]
                    reif_var = self.solver_var(bool_lhs)
                    bool_expr = cpm_expr.args[1]

                    # Just a plain implies or equals between two boolvars
                    if isinstance(bool_expr, _BoolVarImpl): # bv1 -> bv2
                        (self.gcs.post_equals if fully_reify else self.gcs.post_implies)\
                            (*self.solver_vars([bool_lhs, bool_expr]))
                    elif isinstance(bool_expr, Operator): # bv -> and(...), bv -> or(...)  bv -> [->]
                        if bool_expr.name == 'and':
                            self.gcs.post_and_reif(self.solver_vars(bool_expr.args), reif_var, fully_reify)
                        elif bool_expr.name == 'or':
                            self.gcs.post_or_reif(self.solver_vars(bool_expr.args), reif_var, fully_reify)
                        elif bool_expr.name == '->':
                            self.gcs.post_implies_reif(self.solver_vars(bool_expr.args), reif_var, fully_reify)
                        else:
                            # Shouldn't happen if reify_rewrite worked?
                            raise NotImplementedError("Not currently supported by Glasgow Constraint Solver API '{}' {}".format)
                    
                    # Reified Comparison
                    elif isinstance(bool_expr, Comparison):
                        lhs = bool_expr.args[0]
                        rhs = bool_expr.args[1]
                        if bool_expr.name == '==':
                            self.gcs.post_equals_reif(*self.solver_vars([lhs, rhs]), reif_var, fully_reify)
                        elif bool_expr.name == '<=':
                            self.gcs.post_less_than_equal_reif(*self.solver_vars([lhs, rhs]), reif_var, fully_reify)
                        elif bool_expr.name == '<':
                            self.gcs.post_less_than_reif(*self.solver_vars([lhs, rhs]), reif_var, fully_reify)
                        elif bool_expr.name == '>=':
                            self.gcs.post_greater_than_equal_reif(*self.solver_vars([lhs, rhs]), reif_var, fully_reify)
                        elif bool_expr.name == '>':
                            self.gcs.post_greater_than_reif(*self.solver_vars([lhs, rhs]), reif_var, fully_reify)
                        elif bool_expr.name == '!=':
                            # Note: GCS doesn't currently support reified NotEquals, so we need this ugly workaround for now:
                            # bv -> x != y can be written as 
                            # bv -> OR(lt, gt) with lt, gt being BoolVars and the additional constraints
                            # lt == x < y
                            # gt == x > y
                            lt_bool, gt_bool = boolvar(shape=2)
                            self += (lhs < rhs) == lt_bool
                            self += (lhs > rhs) == gt_bool
                            if fully_reify:
                                self += (~bool_lhs).implies(lhs == rhs)
                            self.gcs.post_or_reif(self.solver_vars([lt_bool, gt_bool]), reif_var, False)
                        else:
                            raise NotImplementedError("Not currently supported by Glasgow Constraint Solver API '{}' {}".format)
                    else:
                        # Shouldn't happen if reify_rewrite worked
                        raise NotImplementedError("Not currently supported by Glasgow Constraint Solver API '{}' {}".format)
            
            # Normal comparison     
            elif isinstance(cpm_expr, Comparison):
                lhs = cpm_expr.args[0]
                rhs = cpm_expr.args[1]

                # Due to only_numexpr_equality we can only have '!=', "<=", etc.
                # when the lhs is a variable, sum or wsum
                if isinstance(lhs, _NumVarImpl) or lhs.name == 'sum' or lhs.name == 'wsum':
                    if lhs.name == 'sum' or lhs.name == 'wsum':
                        if lhs.name == 'sum':
                            summands = self.solver_vars(lhs.args)
                            summands.append(self.solver_var(rhs))
                            coeffs = [1]*len(lhs.args) + [-1]
                        else:
                            summands = self.solver_vars(lhs.args[1])
                            summands.append(self.solver_var(rhs))
                            coeffs = list(lhs.args[0]) + [-1]

                        if cpm_expr.name == '==':
                            self.gcs.post_linear_equality(summands, coeffs, 0)
                        elif cpm_expr.name == '!=':
                            self.gcs.post_linear_not_equal(summands, coeffs, 0)
                        elif cpm_expr.name == '<=':
                            self.gcs.post_linear_less_equal(summands, coeffs, 0)
                        elif cpm_expr.name == '<':
                            self.gcs.post_linear_less_equal(summands, coeffs, -1)
                        elif cpm_expr.name == '>=':
                            self.gcs.post_linear_greater_equal(summands, coeffs, 0)
                        elif cpm_expr.name == '>':
                            self.gcs.post_linear_greater_equal(summands, coeffs, 1)
                        else:
                            raise NotImplementedError("Not currently supported by Glasgow Constraint Solver API '{}'".format(cpm_expr))
                    else:
                        if cpm_expr.name == '==':
                            self.gcs.post_equals(*self.solver_vars([lhs, rhs]))
                        elif cpm_expr.name == '!=':
                            self.gcs.post_not_equals(*self.solver_vars([lhs, rhs]))
                        elif cpm_expr.name == '<=':
                            self.gcs.post_less_than_equal(*self.solver_vars([lhs, rhs]))
                        elif cpm_expr.name == '<':
                            self.gcs.post_less_than(*self.solver_vars([lhs, rhs]))
                        elif cpm_expr.name == '>=':
                            self.gcs.post_greater_than_equal(*self.solver_vars([lhs, rhs]))
                        elif cpm_expr.name == '>':
                            self.gcs.post_greater_than(*self.solver_vars([lhs, rhs]))
                        else:
                            raise NotImplementedError("Not currently supported by Glasgow Constraint Solver API '{}'".format(cpm_expr))

                # If the comparison is '==' we can have a NumExpr on the lhs
                elif cpm_expr.name == '==':
                    if lhs.name == 'abs':
                        assert(len(lhs.args) == 1) # Should not have a nested expression inside abs
                        self.gcs.post_abs(*self.solver_vars(list(lhs.args) + [rhs]))
                    elif lhs.name in ['mul', 'div', 'pow', 'mod']:
                        self.gcs.post_arithmetic(*self.solver_vars(list(lhs.args) + [rhs]), lhs.name)
                    elif lhs.name == 'sub':
                        var1 = self.solver_var(lhs.args[0])
                        nVar2 = self.gcs.negate(self.solver_var(lhs.args[1]))
                        self.gcs.post_arithmetic(var1, nVar2, self.solver_var(rhs), 'sum')
                    elif lhs.name == 'sum' and len(lhs.args) == 2:
                        var1 = self.solver_var(lhs.args[0])
                        var2 = self.solver_var(lhs.args[1])
                        self.gcs.post_arithmetic(var1, var2, self.solver_var(rhs), 'sum')
                    elif lhs.name == 'sum' and len(lhs.args) > 2:
                        summands = self.solver_vars(lhs.args)
                        summands.append(self.gcs.negate(self.solver_var(rhs)))
                        self.gcs.post_linear_equality(summands, [1]*len(summands), 0)
                    elif lhs.name == 'wsum':
                        summands = self.solver_vars(lhs.args[1])
                        summands.append(self.gcs.negate(self.solver_var(rhs)))
                        self.gcs.post_linear_equality(summands, list(lhs.args[0]) + [1], 0)
                    elif lhs.name == 'max':
                        self.gcs.post_max(self.solver_vars(lhs.args), self.solver_var(rhs))
                    elif lhs.name == 'min':
                        self.gcs.post_min(self.solver_vars(lhs.args), self.solver_var(rhs))   
                    elif lhs.name == 'element':
                        self.gcs.post_element(self.solver_var(rhs), self.solver_vars(lhs.args[1]), self.solver_vars(lhs.args[0])) 
                    elif lhs.name == 'count':
                        self.gcs.post_count(self.solver_vars(lhs.args[0]), self.solver_var(lhs.args[1]), self.solver_var(rhs))
                    elif lhs.name == 'nvalue':
                        self.gcs.post_nvalue(self.solver_var(rhs), self.solver_vars(lhs.args))
                    else:
                        # Think that's all the possible NumExprs?
                        raise NotImplementedError("Not currently supported by Glasgow Constraint Solver API '{}'".format(cpm_expr))
                else:
                    raise NotImplementedError("Not currently supported by Glasgow Constraint Solver API '{}'".format(cpm_expr))
            
            # rest: base (Boolean) global constraints
            elif cpm_expr.name == 'xor':
                self.gcs.post_xor(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'circuit':
                self.gcs.post_circuit(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'inverse':
                self.gcs.post_inverse(self.solver_vars(cpm_expr.args[0]), self.solver_vars(cpm_expr.args[1]))
            elif cpm_expr.name == 'alldifferent':
                self.gcs.post_alldifferent(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'table':
                self.gcs.post_table(self.solver_vars(cpm_expr.args[0]), cpm_expr.args[1])
            elif cpm_expr.name == 'negative_table':
                self.gcs.post_negative_table(self.solver_vars(cpm_expr.args[0]), cpm_expr.args[1])
            elif isinstance(cpm_expr, GlobalConstraint):
                # GCS also has SmartTable, Regular Language Membership, Knapsack constraints
                # which could be added in future. 
                self += cpm_expr.decompose()  # assumes a decomposition exists...
            else:
                # Hopefully we don't end up here.
                raise NotImplementedError(cpm_expr)

        return self
    __add__ = add  # avoid redirect in superclass


        
