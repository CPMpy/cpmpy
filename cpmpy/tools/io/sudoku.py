#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## sudoku.py
##
"""
Loader for the Sudoku puzzle format used by the
`scaled-sudoku-instances <https://github.com/zayenz/scaled-sudoku-instances>`_ corpus.

Specification
-------------

The format is documented in that repository's README
(https://github.com/zayenz/scaled-sudoku-instances). Each puzzle starts with a
header line ``# size=N box=WxH``, followed by ``N`` rows of length ``N``. Empty
cells are ``.``; filled cells use the alphabet
``123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0`` (value ``k`` is the ``k``-th character,
1-indexed). Files typically use the ``.sdk.txt`` extension.

This is related to, but not the same as, the classic SadMan / SudoCue ``.sdk``
format (fixed 9x9 grids with optional ``#A``/``#L``-style metadata lines; see
https://www.sudocue.net/fileformats.php). The corpus format adds an explicit
size/box header and supports grids from ``6x6`` up to ``36x36``.


=================
List of functions
=================

.. autosummary::
    :nosignatures:

    load_sudoku
    parse_sudoku
"""


import os
import builtins
import re
from typing import Union, Callable, TextIO, Any

import numpy as np
import cpmpy as cp

from cpmpy.expressions.variables import NDVarArray
from cpmpy.tools.io.utils import _handle_loader_input


# Cell values 1..36 map to these symbols (empty cells are '.').
_ALPHABET = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0"
_HEADER_RE = re.compile(
    r"#\s*size=(?P<size>\d+)\s+box=(?P<box_width>\d+)x(?P<box_height>\d+)"
)


def load_sudoku(sudoku: Union[str, os.PathLike, TextIO], open: Callable = builtins.open) -> cp.Model:
    """
    Loader for the Sudoku puzzle format. Loads an instance and returns its matching CPMpy model.

    Arguments:
        sudoku (str or os.PathLike or TextIO):
            - A file path to a Sudoku puzzle file (typically ``.sdk.txt``), or
            - A string containing the puzzle content directly, or
            - A TextIO object already open for reading
        open (Callable):
            If sudoku is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        cp.Model: The CPMpy model of the Sudoku instance.
    """
    data = parse_sudoku(sudoku, open=open)
    model, _ = _model_sudoku(**data)
    return model


def parse_sudoku(instance: Union[str, os.PathLike, TextIO], open: Callable = builtins.open) -> dict[str, Any]:
    """
    Parse a Sudoku puzzle instance file.

    Arguments:
        instance (str or os.PathLike or TextIO):
            - A file path to a Sudoku puzzle file (typically ``.sdk.txt``), or
            - A string containing the puzzle content directly, or
            - A TextIO object already open for reading
        open (Callable):
            If instance is the path to a file, a callable to "open" that file (default=python standard library's 'open').

    Returns:
        dict[str, Any]: Parsed puzzle data with keys ``grid``, ``size``,
            ``box_width``, and ``box_height``. ``grid`` is a 2D numpy array of
            integers (``0`` for empty cells, ``1..size`` for givens).
    """
    with _handle_loader_input(instance, open=open) as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0 or not lines[0].startswith("#"):
        raise ValueError("Missing Sudoku header line (expected '# size=N box=WxH')")

    match = _HEADER_RE.match(lines[0])
    if match is None:
        raise ValueError(f"Invalid Sudoku header: {lines[0]!r}")

    size = int(match.group("size"))
    box_width = int(match.group("box_width"))
    box_height = int(match.group("box_height"))

    if box_width * box_height != size:
        raise ValueError(
            f"Box {box_width}x{box_height} is incompatible with grid size {size}"
        )

    body = lines[1:]
    if len(body) != size:
        raise ValueError(f"Expected {size} grid rows, got {len(body)}")

    cells = []
    for row in body:
        if len(row) != size:
            raise ValueError(f"Expected row width {size}, got {len(row)}")
        cells.append([_decode_symbol(ch) for ch in row])

    grid = np.array(cells, dtype=int)
    if np.any(grid > size):
        raise ValueError(f"Cell values must be in 0..{size}, got max {grid.max()}")

    return {
        "grid": grid,
        "size": size,
        "box_width": box_width,
        "box_height": box_height,
    }


def _decode_symbol(ch: str) -> int:
    """
    Decode a single puzzle symbol to an integer value (0 for empty).
    """
    if ch == ".":
        return 0
    idx = _ALPHABET.find(ch)
    if idx < 0:
        raise ValueError(f"Invalid puzzle symbol {ch!r}")
    return idx + 1


def _model_sudoku(grid: np.ndarray, size: int, box_width: int, box_height: int) -> tuple[cp.Model, NDVarArray]:
    """
    Model a Sudoku instance from a given grid and box shape.

    Arguments:
        grid (np.ndarray): 2D array of givens (0 = empty).
        size (int): Grid order (number of rows/columns).
        box_width (int): Width of each block.
        box_height (int): Height of each block.

    Returns:
        tuple[cp.Model, NDVarArray]: The model and the cell variables.
    """
    puzzle = cp.intvar(1, size, shape=grid.shape, name="puzzle")

    model = cp.Model(
        # Fix given cells
        puzzle[grid != 0] == grid[grid != 0],
        # Rows and columns
        [cp.AllDifferent(row) for row in puzzle],
        [cp.AllDifferent(col) for col in puzzle.T],
    )

    # Blocks
    for i in range(0, size, box_height):
        for j in range(0, size, box_width):
            model += cp.AllDifferent(puzzle[i:i + box_height, j:j + box_width])

    return model, puzzle
