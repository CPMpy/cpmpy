import warnings
import os
from contextlib import contextmanager
from io import StringIO, TextIOBase
from pathlib import Path
from typing import Union, TextIO, Callable, Iterator, cast
import builtins

# mapping file extensions to appropriate format names
_format_map = {
    "mps"   : "mps",
    "lp"    : "lp",
    "cip"   : "cip",
    "fzn"   : "fzn",
    "gms"   : "gms",
    "pip"   : "pip",
    "wcnf"  : "wcnf",
    "cnf"   : "cnf",
    "opb"   : "opb",
    # Corpus files use ``.sdk.txt``; map the ``sdk`` suffix to the sudoku loader.
    "sdk"   : "sudoku",
}

_extension_map: dict[str, list[str]] = {}
for extension, format in _format_map.items():
    _extension_map[format] = _extension_map.get(format, []) + [extension]

def get_extension(format: str) -> str:
    """
    Get the file extension for a given format.
    """
    if len(_extension_map[format]) > 1:
        warnings.warn(f"Multiple extensions found for format {format}: {_extension_map[format]}. Using the first one: {_extension_map[format][0]}")
    
    return _extension_map[format][0]

def get_format(extension: str) -> str:
    """
    Get the format for a given file extension.
    """
    return _format_map[extension]

def _create_header(format: str) -> str:
    """
    Default header for an exported file.
    """
    import cpmpy as cp

    header = "-"*100 + "\n"
    header += "File written by CPMpy\n"
    header += f"    Format: '{format}'\n"
    header += f"    CPMpy Version: {cp.__version__}\n"
    header += "-"*100 + "\n"
    return header

def _derive_format(file_path: Union[str, os.PathLike]) -> str:
    """
    Derive the format of a file from its path by looking at its file extension.

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
        >>> _derive_format("instance.cnf")
        "dimacs"
    """

    # Iterate over the file path extensions in reverse order
    for ext in str(file_path).split(".")[::-1]:
        try:
            return get_format(ext)
        except KeyError:
            continue

    raise ValueError(f"No file format provided and could not derive format from file path: {file_path}")

def _is_potential_path(instance: Union[str, os.PathLike]) -> bool:
    """
    Check if a given instance is a potential path.
    """

    if isinstance(instance, os.PathLike):
        return True

    if not isinstance(instance, str):
        raise ValueError("Instance must be a string or a path-like object")

    if isinstance(instance, str):
        if ("\n" in instance or "\r" in instance): # typical indicator of inline contents, not present in a path string
            return False
        return True

    return False

@contextmanager
def _handle_loader_input(
    source: Union[str, os.PathLike, TextIO],
    open: Callable = builtins.open,
) -> Iterator[TextIO]:
    """
    Context manager yielding a line-iterable source for format loaders.

    - path / PathLike: opened with ``open`` (closed on exit)
    - TextIO: yielded as-is (not closed)
    - inline str: yielded as ``StringIO`` (no close needed)

    Arguments:
        source (str or os.PathLike or TextIO): The source to handle.
        open (Callable): callable to open the file for reading (default: builtin ``open``).
            Use for decompression, e.g. ``lambda p: lzma.open(p, 'rt')`` for ``.cnf.xz``.

    Returns:
        A text stream over the source.

    Raises:
        ValueError: If the source is not a string, os.PathLike, or TextIO.
        FileNotFoundError: If the source is a path-like object and does not exist.
    """
    f: TextIO
    should_close = False

    if isinstance(source, TextIOBase):
        f = cast(TextIO, source)
    elif isinstance(source, (str, os.PathLike)) and _is_potential_path(source):
        path = Path(source)
        if path.exists():
            # ``open`` is documented as a single-argument, path-only callable, so
            # custom decompressing openers (e.g. ``lambda p: lzma.open(p, 'rt')``)
            # and dataset ``open`` methods work without accepting a mode argument.
            f = open(path)
            should_close = True
        elif isinstance(source, str):
            # e.g. "p cnf 0 0" — no newlines, not an existing path: inline content
            f = StringIO(source)
        else:
            raise FileNotFoundError(path)
    elif isinstance(source, str):
        f = StringIO(source)
    else:
        raise ValueError(f"Expected a string, os.PathLike, or TextIO, but got {type(source)}")

    try:
        yield f
    finally:
        if should_close:
            f.close()
