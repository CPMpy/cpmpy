"""
CPMpy tools for loading models from files.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load
    read_formats
"""

from typing import Callable, List, Optional

import cpmpy as cp
from cpmpy.tools.dimacs import load_dimacs
from cpmpy.tools.io.scip import load_scip
from cpmpy.tools.io.wcnf import load_wcnf
from cpmpy.tools.io.opb import load_opb
from cpmpy.tools.io.utils import get_format

# mapping format names to appropriate loader functions
_reader_map = {
    "mps": load_scip,
    "lp": load_scip,
    "cip": load_scip,
    "fzn": load_scip,
    "gms": load_scip,
    "pip": load_scip,
    "dimacs": load_dimacs,
    "opb": load_opb,
    "wcnf": load_wcnf,
}


def _get_reader(format: str) -> Callable[[str], cp.Model]:
    """
    Get the reader function for a given format.

    Arguments:
        format (str): The name of the format to get a reader for.

    Raises:
        ValueError: If the format is not supported.

    Returns:
        A callable that reads a model from a file.
    """

    if format not in _reader_map:
        raise ValueError(f"Unsupported format: {format}")

    return _reader_map[format]

def read_formats() -> List[str]:
    """
    List of supported load formats.

    Each can be used as the `format` argument to the `load` function.
    E.g.:

    .. code-block:: python      

        from cpmpy.tools.io import load
        model = load(file_path, format="mps")
        model = load(file_path, format="lp")
    """
    return list(_reader_map.keys())

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
        >>> _derive_format("instance.mps")
        "mps"
        >>> _derive_format("instance.lp.xz")
        "lp"
    """

    # Iterate over the file path extensions in reverse order
    for ext in file_path.split(".")[::-1]:
        try:
            return get_format(ext)
        except ValueError:
            continue

    raise ValueError(f"No file format provided and could not derive format from file path: {file_path}")

def load(file_path: str, format: Optional[str] = None) -> cp.Model:
    """
    Load a model from a file.

    Arguments:
        file_path (str): The path to the file to load.
        format (Optional[str]): The format of the file to load. If None, the format will be derived from the file path.

    Raises:
        ValueError: If the format is not supported.

    Returns:
        A CPMpy model.
    """

    if format is None:
        format = _derive_format(file_path)

    reader = _get_reader(format)
    return reader(file_path)

# Backward compatibility alias
read = load