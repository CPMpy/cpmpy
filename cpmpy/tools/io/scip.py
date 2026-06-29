#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## scip.py
##
"""
This file implements helper functions for converting CPMpy models to and from various data 
formats supported by the SCIP optimization suite.

============
Installation
============

The 'pyscipopt' optional dependency must be installed separately through `pip`:

.. code-block:: console
    
    $ pip install cpmpy[io.scip]

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_scip
    write_scip
"""


import argparse
import math
import os
import sys
import tempfile
from io import TextIOBase
import numpy as np
import cpmpy as cp
import warnings
import builtins
from typing import Union, Optional, Callable, TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import pyscipopt

from cpmpy.solvers.scip import CPM_scip
from cpmpy.tools.io.utils import _create_header, _derive_format, get_extension


def load_scip(instance: Union[str, os.PathLike, TextIO], open:Callable = builtins.open, assume_integer:bool=False, type: Optional[str]=None) -> cp.Model:
    """
    Load a SCIP-compatible model from a file and return a CPMpy model.

    Arguments:
        instance (str or os.PathLike or TextIO): The path to the SCIP-compatible file to read, a string containing the model content directly, or a TextIO object already open for reading.
        open (Callable): The function to use to open the file. (SCIP does not require this argument, will be ignored)
        assume_integer (bool): Whether to assume that all variables are integer.
        type (str, optional): Type of the inline string or unnamed TextIO input. Required when ``instance`` is a raw string.

    Warning:
        Setting assumes_integer to True will cause CPMpy to assume that all variables are integer, 
        even if they are not explicitly declared as such in the problem file. Use with caution.

    Returns:
        cp.Model: A CPMpy model.
    """

    # Check if SCIP is installed
    if not _SCIPWriter.supported():
        raise Exception("SCIP: Install SCIP IO dependencies: cpmpy[io.scip]")

    from pyscipopt import Model

    content = None
    if isinstance(instance, TextIOBase):
        content = instance.read()
        if type is None:
            try:
                type = _derive_format(getattr(instance, "name", ""))
            except (KeyError, ValueError):
                raise ValueError("Type must be provided when loading a SCIP model from an unnamed TextIO object.")
    elif isinstance(instance, str) and not os.path.exists(instance):
        if type is None:
            raise ValueError("Type must be provided when loading a SCIP model from a string.")
        content = instance

    # SCIP's parser only supports file paths
    # -> write content to a temporary file
    tmp_fname = None
    if content is not None:
        try:
            suffix = "." + get_extension(type)
        except KeyError as e:
            raise ValueError(f"Unsupported SCIP file type: {type}") from e

        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as tmp:
            tmp.write(content)
            tmp_fname = tmp.name
        instance = tmp_fname

    # Load file into pyscipopt model
    scip = Model()
    try:
        scip.hideOutput() # suppress SCIP output
        scip.readProblem(filename=instance)
        scip.hideOutput(quiet=False)
    finally:
        if tmp_fname is not None:
            os.remove(tmp_fname)

    # 1) translate variables
    scip_vars = scip.getVars()
    var_map = {}
    for var in scip_vars:
        name = var.name         # name of the variable
        vtype = var.vtype()     # type of the variable
        if vtype == "BINARY":
            var_map[name] = cp.boolvar(name=name)
        elif vtype == "INTEGER":
            lb = int(var.getLbOriginal())
            ub = int(var.getUbOriginal())
            var_map[name] = cp.intvar(lb, ub, name=name)
        elif vtype == "CONTINUOUS":
            if assume_integer:
                lb = int(math.ceil(var.getLbOriginal()))
                ub = int(math.floor(var.getUbOriginal()))
                if lb != var.getLbOriginal() or ub != var.getUbOriginal():
                    warnings.warn(f"Continuous variable {name} has non-integer bounds {var.getLbOriginal()} - {var.getUbOriginal()}. CPMpy will assume it is integer.")
                var_map[name] = cp.intvar(lb, ub, name=name)
            else:
                raise ValueError(f"CPMpy does not support continious variables: {name}")
        else:
            raise ValueError(f"Unsupported variable type: {vtype}")
        

    model = cp.Model()

    # 2) translate constraints
    scip_cons = scip.getConss()
    for cons in scip_cons:
        ctype = cons.getConshdlrName()  # type of the constraint

        if ctype == "linear":
            cons_vars = scip.getConsVars(cons)  # variables in the constraint (x)
            cons_coeff = scip.getConsVals(cons) # coefficients of the variables (A)

            cpm_vars = [var_map[v.name] for v in cons_vars] # convert to CPMpy variables
            cpm_sum = cp.sum(var*coeff for (var,coeff) in zip(cpm_vars, cons_coeff)) # Ax

            lhs = scip.getLhs(cons) # lhs of the constraint
            rhs = scip.getRhs(cons) # rhs of the constraint

            # convert to integer bounds
            _lhs = int(math.ceil(lhs))
            _rhs = int(math.floor(rhs))
            if _lhs != int(lhs) or _rhs != int(rhs):
                if assume_integer:
                    warnings.warn(f"Constraint {cons.name} has non-integer bounds. CPMpy will assume it is integer.")
                else:
                    raise ValueError(f"Constraint {cons.name} has non-integer bounds. CPMpy does not support non-integer bounds.")

            # add the constraint to the model
            model += _lhs <= cpm_sum
            model += cpm_sum <= _rhs

        else: 
            raise ValueError(f"Unsupported constraint type: {ctype}")

    # 3) translate objective
    scip_objective = scip.getObjective()
    direction = scip.getObjectiveSense()

    n_terms = len(scip_objective.terms)
    obj_vars = cp.cpm_array([None] * n_terms)  # type: ignore[list-item]
    obj_coeffs = np.zeros(n_terms, dtype=int)

    for i, (term, coeff) in enumerate(scip_objective.terms.items()): # terms is a dictionary mapping terms to coefficients
        if len(term.vartuple) > 1:
            raise ValueError(f"Unsupported objective term: {term}") # TODO <- assumes linear, support higher-order terms
        cpm_var = var_map[term.vartuple[0].name] # TODO <- assumes linear
        obj_vars[i] = cpm_var
        
        _coeff = int(math.floor(coeff))
        if _coeff != int(coeff):
            if assume_integer:
                warnings.warn(f"Objective term {term} has non-integer coefficient. CPMpy will assume it is integer.")
            else:
                raise ValueError(f"Objective term {term} has non-integer coefficient. CPMpy does not support non-integer coefficients.")
        obj_coeffs[i] = _coeff

    if direction == "minimize":
        model.minimize(cp.sum(obj_vars * obj_coeffs))
    elif direction == "maximize":
        model.maximize(cp.sum(obj_vars * obj_coeffs))
    else:
        raise ValueError(f"Unsupported objective sense: {direction}")

    return model

class _SCIPWriter(CPM_scip):
    """
    A helper class aiding in translating CPMpy models to SCIP models.

    Builds on top of the CPMpy SCIP solver interface.
    """
    def __init__(self, model: cp.Model, problem_name: Optional[str] = None):
        if not self.supported():
            raise Exception(
                "SCIP: Install SCIP IO dependencies: cpmpy[io.scip]")

        super().__init__(model)
        self.scip_model.setProbName(problem_name)

def _add_header(fname: Union[str, os.PathLike], format: str, header: Optional[str] = None):
    """
    Add a header to a file.

    Arguments:
        fname (str or os.PathLike): The path to the file to add the header to.
        format (str): The format of the file.
        header (Optional[str]): The header to add.
    """

    if header is None:
        header = ""

    with open(fname, "r") as f:
        lines = f.readlines()

    if format == "mps":
        header_lines = ["* " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines
        
    elif format == "lp":
        header_lines = ["\\ " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    elif format == "cip":
        header_lines = ["# " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    elif format == "fzn":
        header_lines = ["% " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    elif format == "gms":
        header_lines = ["* " + line + "\n" for line in header.splitlines()]
        lines = [lines[0]] + header_lines + lines[1:] # handle first line: $OFFLISTING

    elif format == "pip":
        header_lines = ["\\ " + line + "\n" for line in header.splitlines()]
        lines = header_lines + lines

    else:
        warnings.warn(f"Unsupported format for header: {format}")
        return

    with open(fname, "w") as f:
        f.writelines(lines)


def write_scip(
        model: cp.Model, 
        fname: Optional[str] = None, 
        format: str = "mps", 
        header: Optional[str] = None, 
        verbose: bool = False, 
        open: Callable = builtins.open
    ) -> str:
    """
    Write a CPMpy model to file using the SCIP solver.

    Supported formats include: 
    - "mps"
    - "lp"
    - "cip"
    - "fzn"
    - "gms"
    - "pip"

    More formats can be supported upon the installation of additional dependencies (like SIMPL).
    For more information, see the SCIP documentation: https://pyscipopt.readthedocs.io/en/latest/tutorials/readwrite.html

    Arguments:
        model (cp.Model): CPMpy model to write.
        fname (Optional[str]): Path to write to. If None, the file content is returned as a string.
        format (str): Output format (e.g. "mps", "lp", "cip", "fzn", "gms", "pip").
        header (Optional[str]): Optional header text to prepend (format-dependent comment style).
            If None, a default CPMpy header is created only when writing to ``fname``.
            Pass an empty string to skip adding a header.
        verbose (bool): If True, allow SCIP to print progress.
        open (Callable): Callable to open the file for writing (default: builtin ``open``).
            Called as ``open(fname, "w")``. Mirrors the ``open=`` argument in loaders and
            allows custom compression or I/O (e.g.
            ``lambda p, mode='w': lzma.open(p, 'wt')``).

    Returns:
        str: The file content as a string (whether written to ``fname`` or not).
    """

    writer = _SCIPWriter(model, problem_name="CPMpy Model")
    if header is None:
        header = _create_header(format=format) if fname is not None else None
    elif header == "":
        header = None

    # Always write via SCIP to a temp file, then add header and get content
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp:
        tmp_fname = tmp.name
    try:
        if not verbose:
            writer.scip_model.hideOutput()
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        os.dup2(devnull, 1)
        try:
            writer.scip_model.writeProblem(tmp_fname, verbose=verbose)
        finally:
            os.dup2(old_stdout, 1)
            os.close(devnull)
            os.close(old_stdout)
        if not verbose:
            writer.scip_model.hideOutput(quiet=False)
        _add_header(tmp_fname, format, header)
        with builtins.open(tmp_fname, "r") as f:
            content = f.read()
        if fname is not None:
            with open(fname, "w") as f:
                f.write(content)
        return content
    finally:
        os.remove(tmp_fname)

def main():
    parser = argparse.ArgumentParser(description="Parse and solve a SCIP compatible model using CPMpy")
    parser.add_argument("model", help="Path to a SCIP compatible file (or raw string if --string is given)")
    parser.add_argument("-s", "--solver", default=None, help="Solver name to use (default: CPMpy's default)")
    parser.add_argument("--string", action="store_true", help="Interpret the first argument (model) as a raw OPB string instead of a file path")
    parser.add_argument("-t", "--time-limit", type=int, default=None, help="Time limit for the solver in seconds (default: no limit)")
    args = parser.parse_args()

    # Build the CPMpy model
    try:
        if args.string:
            model = load_scip(args.model)
        else:
            model = load_scip(os.path.expanduser(args.model))
    except Exception as e:
        sys.stderr.write(f"Error reading model: {e}\n")
        sys.exit(1)

    # Solve the model
    try:
        if args.solver:
            result = model.solve(solver=args.solver, time_limit=args.time_limit)
        else:
            result = model.solve(time_limit=args.time_limit)
    except Exception as e:
        sys.stderr.write(f"Error solving model: {e}\n")
        sys.exit(1)

    # Print results
    print("Status:", model.status())
    if result is not None:
        if model.has_objective():
            print("Objective:", model.objective_value())
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
