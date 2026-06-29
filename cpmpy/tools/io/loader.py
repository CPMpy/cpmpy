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

from typing import Callable, List, Optional, Union, TextIO
from functools import partial
import os
from pathlib import Path
import cpmpy as cp
from .dimacs import load_dimacs
from cpmpy.tools.io.scip import load_scip
from cpmpy.tools.io.wcnf import load_wcnf
from cpmpy.tools.io.opb import load_opb
from cpmpy.tools.io.utils import _derive_format, _is_potential_path

# mapping format names to appropriate loader functions
_loader_map: dict[str, Callable[..., cp.Model]] = {
    "mps": partial(load_scip, type="mps"),
    "lp": partial(load_scip, type="lp"),
    "cip": partial(load_scip, type="cip"),
    "fzn": partial(load_scip, type="fzn"),
    "gms": partial(load_scip, type="gms"),
    "pip": partial(load_scip, type="pip"),
    "dimacs": load_dimacs,
    "opb": load_opb,
    "cnf": partial(load_dimacs, type="cnf"),
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

def load(instance: Union[str, os.PathLike, TextIO], format: Optional[str] = None) -> cp.Model:
    """
    Load an instance from a file into a CPMpy model..

    Arguments:
        instance (str or os.PathLike or TextIO): The path to the instance file to load, the instance itself as a string, or a TextIO object.
        format (Optional[str]): The format of the file to load. If None, the format will be derived from the file path (best effort). 
                                Might raise a ValueError if the format could not be derived from the file path, or if the format is not supported.

    Raises:
        ValueError: If the format is not supported or could not be derived from the file path.

    Returns:
        A CPMpy model.
    """
    
    if format is not None:
        return _get_loader(format)(instance)

    else:
        if isinstance(instance, (str, os.PathLike)) and _is_potential_path(instance):
            path = Path(instance)
            if path.exists():
                format = _derive_format(instance)
                return _get_loader(format)(instance)
            else:
                raise FileNotFoundError(instance)
        else:
            raise ValueError("Format must be provided when loading instance from a string.")
