"""
CPMpy tools for writing models to files.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    write
    write_formats

==============
Module details
==============
"""

import inspect
from typing import Callable, Optional, List
from functools import partial

import cpmpy as cp
from cpmpy.tools.dimacs import write_dimacs
from cpmpy.tools.io.scip import write_scip

# mapping format names to appropriate writer functions
_writer_map = {
    "mps": partial(write_scip, format="mps"),
    "lp": partial(write_scip, format="lp"),
    "cip": partial(write_scip, format="cip"),
    # "cnf": partial(write_scip, format="cnf"),      # requires SIMPL, not included in pip package
    # "diff": partial(write_scip, format="diff"),    # requires SIMPL, not included in pip package
    "fzn": partial(write_scip, format="fzn"),
    "gms": partial(write_scip, format="gms"),
    # "opb": partial(write_scip, format="opb"),      # requires SIMPL, not included in pip package
    # "osil": partial(write_scip, format="osil"),
    "pip": partial(write_scip, format="pip"),
    # "sol": partial(write_scip, format="sol"),      # requires SIMPL, not included in pip package
    # "wbo": partial(write_scip, format="wbo"),      # requires SIMPL, not included in pip package   
    # "zpl": partial(write_scip, format="zpl"),      # requires SIMPL, not included in pip package
    "dimacs": write_dimacs,
    # "wcnf": write_wcnf,                            # currently not supported
}

def _get_writer(format: str) -> Callable:
    """
    Get the writer function for a given format.

    Arguments:
        format (str): The name of the format to get a writer for.

    Raises:
        ValueError: If the format is not supported.

    Returns:
        A callable that writes a model to a file.
    """

    if format not in _writer_map:
        raise ValueError(f"Unsupported format: {format}")

    return _writer_map[format]

def write_formats() -> List[str]:
    """
    List of supported write formats.

    Each can be used as the `format` argument to the `write` function.
    E.g.:

    .. code-block:: python

        from cpmpy.tools.io import write, write_formats, get_extension
        write(model, format=write_formats()[0])
        write(model, format=write_formats()[1], file_path=f"model.{get_extension(write_formats()[1])}")
    """
    return list(_writer_map.keys())

def _create_header(format: str) -> str:
    """
    Default header for a file.
    """
    header = "-"*100 + "\n"
    header += "File written by CPMpy\n"
    header += f"    Format: '{format}'\n"
    header += f"    CPMpy Version: {cp.__version__}\n"
    header += "-"*100 + "\n"
    return header

def write(model: cp.Model, format: str, file_path: Optional[str] = None, verbose: bool = False, header: Optional[str] = None, **kwargs) -> str:
    """
    Write a model to a file.

    Arguments:
        model (cp.Model): The model to write.
        format (str): The format to write the model in.
        file_path (Optional[str]): The path to the file to write the model to. If None, only a string containing the model will be returned.
        verbose (bool): Whether to print verbose output.
        header (Optional[str]): The header to put at the top of the file. If None, a default header will be created. Pass an empty string to skip adding a header.
        **kwargs: Additional arguments to pass to the writer.
    """

    writer = _get_writer(format)

    kwargs["verbose"] = verbose

    # keep only kwargs the writer accepts
    sig = inspect.signature(writer)
    allowed = sig.parameters
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in allowed
    }

    # create header if not provided
    if header is None:
        header = _create_header(format)
    if header == "":
        header = None

    return writer(model, fname=file_path, header=header, **filtered_kwargs)