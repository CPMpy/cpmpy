"""
    Solver interfaces have the same API as :class:`Model <cpmpy.model.Model>`.
    However some solvers are **incremental**, meaning that after solving a problem, you can add constraints or change
    the objective, and the next solve will reuse as much information from the previous solve as possible.
    Some clause learning solvers also support solving with **assumptions**, meaning you can solve the same problem with
    different assumption variables toggled on/off, and the solver will reuse information from the previous solves.

    See :ref:`supported-solvers` for the list of solvers and their capabilities.

    To benefit from incrementality, you have to instantiate the solver object and reuse it, rather than working on a Model object.
    Solvers must be instantiated throught the static :class:`cp.SolverLookup <cpmpy.solvers.utils.SolverLookup>` class:

    - :meth:`cp.SolverLookup.solvernames() <cpmpy.solvers.utils.SolverLookup.solvernames>` -- List all installed solvers (including subsolvers).
    - :meth:`cp.SolverLookup.get(solvername, model=None) <cpmpy.solvers.utils.SolverLookup.get>` -- Initialize a specific solver.

    For example creating a CPMpy solver object for OR-Tools:

    .. code-block:: python

        import cpmpy as cp
        s = cp.SolverLookup.get("ortools")
        # can now use solver object 's' over and over again 


    =========================
    List of solver submodules
    =========================
    .. autosummary::
        :nosignatures:

        ortools
        choco
        gcs
        minizinc
        cpo
        gurobi
        exact
        z3
        pysat
        pysdd
        pindakaas
        pumpkin
        cplex
        hexaly

    =========================
    List of helper submodules
    =========================
    .. autosummary::
        :nosignatures:

        solver_interface
        utils

    =================
    List of functions
    =================
    .. autosummary::
        :nosignatures:

        param_combinations
"""

from .utils import builtin_solvers, get_supported_solvers, param_combinations
from .ortools import CPM_ortools
from .pysat import CPM_pysat
from .minizinc import CPM_minizinc
from .gurobi import  CPM_gurobi
from .pysdd import CPM_pysdd
from .z3 import CPM_z3
from .exact import CPM_exact
from .choco import CPM_choco
from .gcs import CPM_gcs
from .cpo import CPM_cpo
from .pindakaas import CPM_pindakaas
from .pumpkin import CPM_pumpkin
from .cplex import CPM_cplex
from .hexaly import CPM_hexaly
