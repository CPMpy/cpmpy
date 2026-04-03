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
from cpmpy.tools.io.utils import get_format
from cpmpy.tools.io.xcsp3 import load_xcsp3

# mapping format names to appropriate loader functions
_loader_map = {
    "xcsp3": load_xcsp3,
}


def _get_loader(format: str) -> Callable[[str], cp.Model]:
    """
    Get the loader function for a given format.

    Arguments:
        format (str): The name of the format to get a loader for.

    Raises:
        ValueError: If the format is not supported.

    Returns:
        A callable that loads a model from a file.
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

        from cpmpy.tools.io import load
        model = load(file_path, format="mps")
        model = load(file_path, format="lp")
    """
    return list(_loader_map.keys())

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

    reader = _get_loader(format)
    return reader(file_path)