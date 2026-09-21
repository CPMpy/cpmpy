IO (:mod:`cpmpy.tools.io`)
==========================

.. py:module:: cpmpy.tools.io

.. automodule:: cpmpy.tools.io
    :members:
    :undoc-members:
    :inherited-members:


Available formats
-----------------

.. list-table::
   :header-rows: 1

   * - Format
     - Read
     - Write
   * - ``cip``
     - ✓
     - ✓
   * - ``cnf``
     - ✓
     - ✓
   * - ``dimacs``
     - ✓
     - ✓
   * - ``fzn``
     - ✓
     - ✓
   * - ``gms``
     - ✓
     - ✓
   * - ``lp``
     - ✓
     - ✓
   * - ``mps``
     - ✓
     - ✓
   * - ``opb``
     - ✓
     - ✓
   * - ``pip``
     - ✓
     - ✓
   * - ``wcnf``
     - ✓
     - ✓
   * - ``xcsp3``
     - ✓
     -
   * - ``jsplib``
     - ✓
     -
   * - ``nurserostering``
     - ✓
     -
   * - ``rcpsp``
     - ✓
     -
   * - ``sudoku``
     - ✓
     -


Format-specific loaders and writers:

.. toctree::
    :maxdepth: 1

    io/dimacs
    io/jsplib
    io/nurserostering
    io/opb
    io/rcpsp
    io/scip
    io/sudoku
    io/wcnf
    io/xcsp3
    
    
    