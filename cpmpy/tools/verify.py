#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## verify.py
##
"""
Verify solver-generated proofs with an external checker.

Runs a checker such as ``drat-trim`` or ``veripb`` as a subprocess on the
proof files produced by a solver. Solver interfaces call this from their
:meth:`verify` method; it can also be used directly on existing proof files.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    verify
"""
import sys
import time
import subprocess
from os import path
from shutil import which


def verify_prooflog(verifier, proof_files, time_limit=None, display_output=False, verifier_args=[]):
    """
    Verify proof files using an external CLI tool.

    The checker is invoked as ``verifier [verifier_args...] <proof_files...>``.
    For example ``drat-trim formula.cnf proof.drat`` or ``veripb model.opb proof.pbp``.

    Arguments:
        - verifier (str):           name or path of the proof checker executable (must be on the system path if a name)
        - proof_files (list[str]):  proof files to pass to the checker, in the order the checker expects
        - time_limit (float):       time limit for verification in seconds (default: None)
        - display_output (bool):    whether to print the output from the checker (default: False)
        - verifier_args (list[str]): extra command line arguments to pass to the checker (default: [])

    Returns:
        - status (dict): result and statistics of the verification run.
            Keys:
                - "result": True if the proof is valid, False otherwise.
                - "runtime": Time taken for verification.
                - "verifier_args": List of command line arguments that were passed to the checker.
                - "error_message": Error message from the checker if the proof is invalid.
                - "timeout": True if the verification timed out, False otherwise.
    """
    if not which(verifier):
        raise Exception(f"Unable to run {verifier}: make sure it is installed and on system path.")

    status = dict(verifier_args=verifier_args, timeout=False)
    try:
        t0 = time.time()
        result = subprocess.run([verifier] + verifier_args + list(proof_files),
                                 timeout=time_limit,
                                 capture_output=True)
        if display_output:
            # keep raw bytes (some checkers prefix status lines with '\r')
            if result.stdout:
                sys.stdout.buffer.write(result.stdout)
            if result.stderr:
                sys.stderr.buffer.write(result.stderr)
    except subprocess.TimeoutExpired:
        status["result"] = False
        status["timeout"] = True
        status["runtime"] = time.time() - t0
        return status

    status["runtime"] = time.time() - t0
    status["result"] = result.returncode == 0
    if result.returncode != 0:
        status["error_message"] = result.stderr.decode() if result.stderr else ""

    return status
