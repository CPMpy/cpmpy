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
from .dimacs import write_dimacs
from cpmpy.tools.io.scip import write_scip
from cpmpy.tools.io.opb import write_opb
from cpmpy.tools.io.utils import get_format

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
    "opb": write_opb,
    "wcnf": write_dimacs,
}

# Maps each format to the external packages its writer depends on.
# Used by writer_dependencies() to record provenance in sidecar metadata.
_writer_deps = {
    "mps": ["pyscipopt"],
    "lp": ["pyscipopt"],
    "cip": ["pyscipopt"],
    "fzn": ["pyscipopt"],
    "gms": ["pyscipopt"],
    "pip": ["pyscipopt"],
    "dimacs": ["pindakaas"],
    "wcnf": ["pindakaas"],
    "opb": [],
}


def writer_dependencies(format: str) -> dict:
    """Return a dict of ``{package_name: version}`` for the writer's external deps.

    Arguments:
        format: target format name (e.g., ``"mps"``, ``"dimacs"``, ``"opb"``).

    Returns:
        dict mapping package names to installed version strings.
        Packages that are not installed are omitted.
    """
    from importlib.metadata import version, PackageNotFoundError

    deps = _writer_deps.get(format, [])
    result = {}
    for pkg in deps:
        try:
            result[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return result


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
        write(model, format=write_formats()[0])  # Returns string
        write(model, f"model.{get_extension(write_formats()[1])}")  # Writes to file, format auto-detected
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

def _derive_format(file_path: str) -> str:
    """
    Derive the format of a file from its path.

    Arguments:
        file_path (str): The path to the file to derive the format from.

    Raises:
        ValueError: If the format could not be derived from the file path.

    Returns:
        The name of the format.

    Example:
        >>> _derive_format("output.mps")
        "mps"
        >>> _derive_format("output.lp.xz")
        "lp"
    """

    # Iterate over the file path extensions in reverse order
    for ext in file_path.split(".")[::-1]:
        try:
            return get_format(ext)
        except (ValueError, KeyError):
            continue

    raise ValueError(f"No file format provided and could not derive format from file path: {file_path}")

def write(model: cp.Model, file_path: Optional[str] = None, format: Optional[str] = None, verbose: bool = False, header: Optional[str] = None, **kwargs) -> str:
    """
    Write a model to a file.

    Arguments:
        model (cp.Model): The model to write.
        file_path (Optional[str]): The path to the file to write the model to. If None, only a string containing the model will be returned.
        format (Optional[str]): The format to write the model in. If None and file_path is provided, the format will be derived from the file path extension.
        verbose (bool): Whether to print verbose output.
        header (Optional[str]): The header to put at the top of the file. If None, a default header will be created. Pass an empty string to skip adding a header.
        **kwargs: Additional arguments to pass to the writer.

    Raises:
        ValueError: If the format is not supported or could not be derived from the file path.

    Example:
        >>> write(model, "output.opb")  # Format auto-detected from .opb
        >>> write(model, "output.txt", format="opb")  # Format explicitly specified
        >>> write(model, format="opb")  # Returns string, format must be specified
    """

    # Derive format from file_path if not provided
    if format is None:
        if file_path is None:
            raise ValueError("Either 'format' or 'file_path' must be provided")
        format = _derive_format(file_path)

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