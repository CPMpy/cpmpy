"""
CPMpy tools for loading models from files.

=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load
    load_formats
"""

from typing import Callable, List, Optional

import cpmpy as cp
from .dimacs import load_dimacs
from cpmpy.tools.io.scip import load_scip
from cpmpy.tools.io.wcnf import load_wcnf
from cpmpy.tools.io.opb import load_opb
from cpmpy.tools.io.utils import _derive_format

# mapping format names to appropriate loader functions
_loader_map: dict[str, Callable[..., cp.Model]] = {
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


def _get_loader(format: str) -> Callable[..., cp.Model]:
    """
    Get the loader function for a given format.

    Arguments:
        format (str): The name of the format to get a loader for.

    Raises:
        ValueError: If the format is not supported.

    Returns:
        A callable that loads a model from a file in the given format.
    """

    if format not in _loader_map:
        raise ValueError(f"Unsupported format: {format}")

    return _loader_map[format]

def load_formats() -> List[str]:
    """
    List of supported load formats.

    Each can be used as the `format` argument to the `load` function.
    E.g.:

    .. code-block:: python      

        from cpmpy.tools.io import load, load_formats
        assert "mps" in load_formats()
        model = load(file_path_to_mps, format="mps")
        assert "lp" in load_formats()
        model = load(file_path_to_lp, format="lp")
    """
    return list(_loader_map.keys())



def load(file_path: str, format: Optional[str] = None) -> cp.Model:
    """
    Load a model from a file.

    Arguments:
        file_path (str): The path to the file to load.
        format (Optional[str]): The format of the file to load. If None, the format will be derived from the file path (best effort). 
                                Might raise a ValueError if the format could not be derived from the file path, or if the format is not supported.

    Raises:
        ValueError: If the format is not supported or could not be derived from the file path.

    Returns:
        A CPMpy model.
    """

    if format is None:
        format = _derive_format(file_path)

    loader = _get_loader(format)
    return loader(file_path)