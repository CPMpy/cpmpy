"""
Local search methods built on top of CPMpy solvers.

Implements Large Neighborhood Search (LNS): iteratively destroy part of a
solution and re-optimize the resulting neighborhood with a CP solver.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    large_neighborhood_search
"""
from typing import Optional, Callable
import inspect
import time

import cpmpy as cp
from cpmpy.expressions.variables import _NumVarImpl
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.transformations.normalize import toplevel_list


def large_neighborhood_search(model: cp.Model, 
                              destroy: Callable,
                              max_iterations: Optional[int] = None,
                              total_time_limit: Optional[float] = None,
                              incremental = True,
                              verbose = 1,
                              solver = "ortools",
                              **kwargs):
    """
    Improve a solution of an optimization model using Large Neighborhood Search (LNS).

    Each iteration calls ``destroy`` to unassign part of the current solution, then re-solves
    the resulting neighborhood. The best assignment found is stored in the model's variable values.

    All decision variables in the model must already be assigned, e.g. from a previous
    :meth:`solve() <cpmpy.model.Model.solve>` call or a manually constructed solution.

    At least one of ``max_iterations`` or ``total_time_limit`` should be set, otherwise the
    search runs indefinitely.

    Arguments:
        - model (:class:`~cpmpy.model.Model`): CPMpy optimization model
        - destroy (Callable): callback invoked each iteration; should unassign part of the current solution
            by calling `_NumVarImpl.clear()` on the variables to unassign. Invoked after each solver call.
        - max_iterations (int, optional): maximum number of destroy-and-repair iterations
        - total_time_limit (float, optional): overall time budget in seconds
        - incremental (bool, optional): whether to use assumption variables to fix the values of non-destroyed variables.
            Avoids re-initializing the solver between iterations, but can introduce overhead.
        - verbose (int, optional): how much information to print (0=none, default: 1)
        - solver (str): name of a solver to use, defaults to "ortools"
        - **kwargs: extra keyword arguments passed to the solver (e.g., ``time_limit`` to set the time limit per iteration)

    Example:
        .. code-block:: python

            # knapsack model
            weights = [...]
            vals = [...]
            capacity = ...

            x = cp.boolvar(shape=len(weights))
            model = cp.Model(cp.sum(weights * x) <= capacity, 
                             maximize=cp.sum(vals * x))

            # greedy solution, assign variables with highest values until capacity is reached
            sorted_vars = sorted(enumerate(x), key=lambda v: vals[v[0]] / weights[v[0]], reverse=True)
            for i, var in sorted_vars:
                if used_capacity + weights[i] <= capacity:
                    var._value = 1
                    used_capacity += weights[i]
                else:
                    var._value = 0

            def destroy(): # clear first half of the variables
                for i in range(len(x)//2):
                    x[i].clear()

            large_neighborhood_search(model, destroy, max_iterations=10, time_limit=1)
    """

    if model.objective_ is None:
        raise ValueError("LNS only supports optimization models")
    if max_iterations is None and total_time_limit is None:
        raise ValueError("At least one of max_iterations or total_time_limit must be set")

    model_vars = sorted(get_variables_model(model), key=str)

    s = cp.SolverLookup.get(solver, model)
    incremental = incremental and "assumptions" in inspect.signature(s.solve).parameters

    # keep track of best solution and objective value
    start = time.time()

    best_sol = [var.value() for var in model_vars]
    best_obj = model.objective_value()
    assert best_obj is not None, "Not all variables are assigned, LNS requires a full solution to start from."
    sol = [var.value() for var in model_vars]
    if verbose >= 1:
        print(f"Starting LNS ({len(model_vars)} vars, solver={solver})")
        print(f" - initial objective: {best_obj}")
        if verbose >= 2:
            print(f" - incremental: {incremental}")

    num_iter = 0
    while 1:

        num_iter += 1
        if total_time_limit is not None and time.time() - start > total_time_limit:
            if verbose >= 1:
                print(f"Stopping: total time limit reached ({total_time_limit}s)")
            break
        if max_iterations is not None and num_iter > max_iterations:
            if verbose >= 1:
                print(f"Stopping: max iterations reached ({max_iterations})")
            break

        # destroy (part of) the solution
        destroy()
        if verbose >= 2:
            n_destroyed = sum(1 for var in model_vars if var.value() is None)
            print(f"Iteration {num_iter}: destroyed {n_destroyed}/{len(model_vars)} variables")

        # fix the variables that are not destroyed
        if incremental:
            bv = cp.boolvar()
            s += bv.implies(cp.all(var == sol[i] for i, var in enumerate(model_vars) if var.value() is not None))
            kwargs["assumptions"] = [bv]
        else: # not incremental, re-initialize solver
            s = cp.SolverLookup.get(solver, model)
            s += cp.all(var == sol[i] for i, var in enumerate(model_vars) if var.value() is not None)

        # re-solve
        res = s.solve(**kwargs)
        if res is False:
            raise RuntimeError(f"Solver returned False, no solution found after destroy-and-repair (exit status: {s.status()})")

        sol = [var.value() for var in model_vars]
        obj = model.objective_value()
        assert obj is not None # to make mypy happy
        if model.objective_is_min:
            improved = obj < best_obj
        else:
            improved = obj > best_obj
        if improved:
            best_sol = sol
            best_obj = obj

        if verbose >= 1:
            extra = " (new best)" if improved else ""
            print(f"Iteration {num_iter}: objective {obj}, best {best_obj}{extra}")
        if verbose >= 2:
            print(f" - solve: {s.status()}")

    # done, restore best solution
    for var, val in zip(model_vars, best_sol):
        var._value = val
    if verbose >= 1:
        print(f"LNS finished after {num_iter - 1} iterations ({time.time() - start:.1f}s), best objective: {best_obj}")