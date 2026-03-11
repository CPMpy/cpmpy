Writers (:mod:`cpmpy.tools.io`)
================================

CPMpy can serialize models to various file formats for use with external solvers
or for format translation. All writers accept an optional ``file_path``; omitting
it (or passing ``None``) returns the result as a string.

Basic Usage
-----------

A unified ``write()`` function auto-detects the format from the file extension:

.. code-block:: python

    import cpmpy as cp
    from cpmpy.tools.io import write

    x = cp.intvar(0, 10, name="x")
    y = cp.intvar(0, 10, name="y")
    model = cp.Model([x + y <= 5], minimize=x + y)

    write(model, "output.opb")          # format from extension
    write(model, "output.mps")
    write(model, "output.cnf")

    # Explicit format
    write(model, "output.txt", format="opb")

    # Write to string (no file)
    opb_string = write(model, format="opb")

Supported Formats
-----------------

.. list-table::
   :header-rows: 1

   * - **Format**
     - **Extension**
     - **Writer function**
     - **Dependencies**
   * - OPB
     - ``.opb``
     - :func:`write_opb <cpmpy.tools.io.opb.write_opb>`
     - —
   * - DIMACS
     - ``.cnf``
     - :func:`write_dimacs <cpmpy.tools.io.dimacs.write_dimacs>`
     - —
   * - MPS
     - ``.mps``
     - :func:`write_scip <cpmpy.tools.io.scip.write_scip>`
     - pyscipopt
   * - LP
     - ``.lp``
     - :func:`write_scip <cpmpy.tools.io.scip.write_scip>`
     - pyscipopt
   * - CIP
     - ``.cip``
     - :func:`write_scip <cpmpy.tools.io.scip.write_scip>`
     - pyscipopt
   * - FZN
     - ``.fzn``
     - :func:`write_scip <cpmpy.tools.io.scip.write_scip>`
     - pyscipopt
   * - GMS
     - ``.gms``
     - :func:`write_scip <cpmpy.tools.io.scip.write_scip>`
     - pyscipopt
   * - PIP
     - ``.pip``
     - :func:`write_scip <cpmpy.tools.io.scip.write_scip>`
     - pyscipopt

Format-Specific Writers
-----------------------

All writers return the serialized string when no ``file_path`` (or ``fname``) is
given, making them suitable for use inside dataset transforms.

.. code-block:: python

    from cpmpy.tools.io.opb import write_opb

    write_opb(model, "output.opb")          # write to file
    opb_string = write_opb(model)           # return as string

.. code-block:: python

    from cpmpy.tools.io.dimacs import write_dimacs

    write_dimacs(model, "output.cnf")
    cnf_string = write_dimacs(model)

Compressed output
-----------------

Writers mirror the loader convention: many format-specific writers accept an
optional ``open`` callable. This allows you to write compressed output (or use
any custom I/O) without CPMpy guessing what compression you want.

.. code-block:: python

    import lzma
    from cpmpy.tools.io.opb import write_opb

    xz_text = lambda path, mode="w": lzma.open(path, "wt")
    write_opb(model, "output.opb.xz", open=xz_text)

.. code-block:: python

    import lzma
    from cpmpy.tools.io.dimacs import write_dimacs

    xz_text = lambda path, mode="w": lzma.open(path, "wt")
    write_dimacs(model, "output.cnf.xz", open=xz_text)

.. code-block:: python

    # MPS / LP / CIP / FZN / GMS / PIP  (require pyscipopt)
    from cpmpy.tools.io.scip import write_scip

    write_scip(model, "output.mps", format="mps")
    write_scip(model, "output.fzn", format="fzn")
    mps_string = write_scip(model, format="mps")  # return as string

    # Compressed output via open=
    import lzma
    xz_text = lambda path, mode="w": lzma.open(path, "wt")
    write_scip(model, "output.mps.xz", format="mps", open=xz_text)

Format Limitations
------------------

- **DIMACS**: Boolean variables and CNF constraints only.
- **OPB**: Linear constraints and integer variables.
- **MPS/LP**: Linear and integer constraints.
- **FZN**: MiniZinc-compatible constraints.

Models containing unsupported features will raise an exception at write time.

Checking Writer Dependencies
-----------------------------

.. code-block:: python

    from cpmpy.tools.io.writer import writer_dependencies

    print(writer_dependencies("mps"))
    # {'pyscipopt': '0.4.8'}  — package name → installed version

Listing Available Formats
--------------------------

.. code-block:: python

    from cpmpy.tools.io import write_formats
    print(write_formats())
    # ['mps', 'lp', 'cip', 'fzn', 'gms', 'pip', 'dimacs', 'opb']

Converting Between Formats
---------------------------

Load a file in one format and write it in another:

.. code-block:: python

    from cpmpy.tools.io import load, write

    model = load("input.opb")
    write(model, "output.mps")

For bulk format translation across a dataset, see :doc:`/reading_and_writing`
and the ``Translate`` transform in :doc:`datasets`.

API Reference
-------------

.. automodule:: cpmpy.tools.io.writer
    :members:
    :undoc-members:
