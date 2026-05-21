Benchmarks (:mod:`cpmpy.tools.benchmark`)
=====================================================

CPMpy provides a comprehensive benchmarking framework for running constraint programming benchmarks 
across multiple instances and solvers. The benchmark module allows you to systematically evaluate 
solver performance with proper resource management, error handling, and result collection.

Overview
--------

The benchmark module provides:

- **Benchmark base class**: Framework for running individual instances
- **Dataset-specific benchmarks**: Pre-configured benchmarks for XCSP3, OPB, MSE, JSPLib, PSPLib, etc.
- **Resource management**: Time and memory limits with proper cleanup
- **Solver configuration**: Automatic solver parameter configuration
- **Result tracking**: Structured output and intermediate solution reporting

Basic Usage
-----------

The simplest way to run a benchmark:

.. code-block:: python

    from cpmpy.tools.benchmark import Benchmark
    from cpmpy.tools.io.opb import read_opb
    
    # Create a benchmark with a reader
    bm = Benchmark(reader=read_opb)
    
    # Run a single instance
    bm.run(
        instance="instance.opb",
        solver="ortools",
        time_limit=30,
        mem_limit=1024,
        verbose=True
    )

Available Benchmarks
--------------------

CPMpy provides pre-configured benchmarks for various datasets:

.. list-table::
   :header-rows: 1

   * - **Benchmark Class**
     - **Dataset**
     - **Reader**
     - **Description**
   * - :class:`XCSP3Benchmark <cpmpy.tools.benchmark.xcsp3.XCSP3Benchmark>`
     - XCSP3Dataset
     - read_xcsp3
     - Benchmark for XCSP3 Competition instances
   * - :class:`OPBBenchmark <cpmpy.tools.benchmark.opb.OPBBenchmark>`
     - OPBDataset
     - read_opb
     - Benchmark for Pseudo-Boolean Competition instances
   * - :class:`MSEBenchmark <cpmpy.tools.benchmark.mse.MSEBenchmark>`
     - MaxSATEvalDataset
     - read_wcnf
     - Benchmark for MaxSAT Evaluation instances
   * - :class:`JSPLibBenchmark <cpmpy.tools.benchmark.jsplib.JSPLibBenchmark>`
     - JSPLibDataset
     - read_jsplib
     - Benchmark for Job Shop Scheduling instances
   * - :class:`PSPLibBenchmark <cpmpy.tools.benchmark.psplib.PSPLibBenchmark>`
     - PSPLibDataset
     - read_rcpsp
     - Benchmark for Project Scheduling instances
   * - :class:`NurseRosteringBenchmark <cpmpy.tools.benchmark.nurserostering.NurseRosteringBenchmark>`
     - NurseRosteringDataset
     - read_nurserostering
     - Benchmark for Nurse Rostering instances

Using Pre-configured Benchmarks
--------------------------------

Example with XCSP3 benchmark:

.. code-block:: python

    from cpmpy.tools.benchmark.xcsp3 import XCSP3Benchmark
    
    bm = XCSP3Benchmark()
    
    bm.run(
        instance="instance.xml",
        solver="ortools",
        time_limit=60,
        mem_limit=2048,
        cores=4,
        verbose=True
    )

Resource Limits
---------------

Benchmarks support both time and memory limits:

.. code-block:: python

    bm = Benchmark(reader=read_opb)
    
    bm.run(
        instance="instance.opb",
        solver="ortools",
        time_limit=300,      # 5 minutes in seconds
        mem_limit=4096,      # 4 GB in MiB
        cores=1             # Number of CPU cores
    )

Solver Configuration
---------------------

The benchmark framework automatically configures solver parameters. You can customize this:

.. code-block:: python

    class CustomBenchmark(Benchmark):
        def ortools_arguments(self, model, cores=None, seed=None, **kwargs):
            res = super().ortools_arguments(model, cores=cores, seed=seed, **kwargs)
            # Add custom OR-Tools parameters
            res[0]["use_rins_lns"] = True
            return res

Intermediate Solutions
----------------------

For optimization problems, you can enable intermediate solution reporting:

.. code-block:: python

    bm = Benchmark(reader=read_opb)
    
    bm.run(
        instance="instance.opb",
        solver="ortools",
        time_limit=300,
        intermediate=True  # Report intermediate solutions
    )

This will print intermediate objective values as they are found.

Exit Status
-----------

Benchmarks return exit statuses indicating the result:

.. code-block:: python

    from cpmpy.tools.benchmark import ExitStatus
    
    # ExitStatus.optimal: Optimal solution found (COP)
    # ExitStatus.sat: Solution found but not proven optimal (CSP/COP)
    # ExitStatus.unsat: Instance is unsatisfiable
    # ExitStatus.unsupported: Instance contains unsupported features
    # ExitStatus.unknown: Any other case

Custom Benchmarks
-----------------

Create custom benchmarks by inheriting from the Benchmark base class:

.. code-block:: python

    from cpmpy.tools.benchmark import Benchmark
    from cpmpy.tools.io.opb import read_opb
    
    class MyBenchmark(Benchmark):
        def print_result(self, s):
            # Custom result printing
            print(f"Custom result: {s.status()}")
        
        def handle_exception(self, e):
            # Custom error handling
            print(f"Custom error: {e}")
            super().handle_exception(e)

Error Handling
--------------

The benchmark framework handles various error conditions:

- **MemoryError**: When memory limit is exceeded
- **TimeoutError**: When time limit is exceeded
- **NotImplementedError**: When instance contains unsupported features
- **Other exceptions**: General error handling with stack traces

All errors are properly handled and reported through callback methods.

Signal Handling
---------------

Benchmarks properly handle system signals:

- **SIGTERM/SIGINT**: Graceful termination
- **SIGXCPU**: CPU time limit exceeded (Unix only)

API Reference
-------------

.. automodule:: cpmpy.tools.benchmark._base
    :members:
    :undoc-members:
    :inherited-members:
