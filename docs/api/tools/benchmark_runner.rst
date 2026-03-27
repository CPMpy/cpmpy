Benchmark Runner (:mod:`cpmpy.tools.benchmark.runner`)
=====================================================

The benchmark runner provides functionality to execute benchmarks across multiple instances 
in parallel, with proper resource management, result collection, and CSV output generation.

Overview
--------

The benchmark runner module provides:

- **Parallel execution**: Run multiple instances concurrently
- **Resource management**: Time and memory limits per instance
- **Result collection**: Structured CSV output with instance metadata
- **Progress tracking**: Progress bars and status reporting
- **Error isolation**: Each instance runs in isolation to prevent crashes

Basic Usage
-----------

The simplest way to run a benchmark across a dataset:

.. code-block:: python

    from cpmpy.tools.benchmark.runner import benchmark_runner
    from cpmpy.tools.benchmark.opb import OPBBenchmark
    from cpmpy.tools.datasets import OPBDataset
    
    # Load dataset
    dataset = OPBDataset(root=".", year=2023, download=True)
    
    # Create benchmark instance
    benchmark = OPBBenchmark()
    
    # Run benchmark across all instances
    output_file = benchmark_runner(
        dataset=dataset,
        instance_runner=benchmark,
        output_file="results.csv",
        solver="ortools",
        workers=4,
        time_limit=300,
        mem_limit=4096
    )

Function Signature
------------------

.. code-block:: python

    benchmark_runner(
        dataset,              # Dataset object
        instance_runner,      # Benchmark instance
        output_file,          # Output CSV file path
        solver,               # Solver name
        workers=1,            # Number of parallel workers
        time_limit=300,       # Time limit per instance (seconds)
        mem_limit=4096,       # Memory limit per instance (MiB)
        cores=1,              # CPU cores per instance
        verbose=False,        # Show solver output
        intermediate=False,  # Report intermediate solutions
        checker_path=None,    # Path to solution checker
        **kwargs             # Additional arguments
    ) -> str                 # Returns output file path

Parameters
----------

dataset
~~~~~~~

A dataset object (e.g., :class:`XCSP3Dataset`, :class:`OPBDataset`) that provides instances to benchmark.

instance_runner
~~~~~~~~~~~~~~~

A benchmark instance (e.g., :class:`XCSP3Benchmark`, :class:`OPBBenchmark`) that implements the `run()` method.

output_file
~~~~~~~~~~~

Path to the CSV file where results will be written. The file will contain columns for:
- Instance metadata (name, path, category)
- Solver status
- Runtime
- Memory usage
- Objective value (if applicable)
- Other benchmark-specific fields

solver
~~~~~~

Name of the solver to use (e.g., "ortools", "gurobi", "z3").

workers
~~~~~~~

Number of parallel processes to run instances. Default is 1 (sequential execution).

time_limit
~~~~~~~~~~

Time limit in seconds for each instance. Default is 300 (5 minutes).

mem_limit
~~~~~~~~~

Memory limit in MiB (1024 * 1024 bytes) per instance. Default is 4096 (4 GB).

cores
~~~~~

Number of CPU cores assigned per instance. Default is 1.

verbose
~~~~~~~

Whether to show solver output in stdout. Default is False.

intermediate
~~~~~~~~~~~~

Whether to report intermediate solutions if supported. Default is False.

checker_path
~~~~~~~~~~~~

Optional path to a solution checker executable for validating instance solutions.

Example: Running XCSP3 Benchmark
---------------------------------

.. code-block:: python

    from cpmpy.tools.benchmark.runner import benchmark_runner
    from cpmpy.tools.benchmark.xcsp3 import XCSP3Benchmark
    from cpmpy.tools.datasets import XCSP3Dataset
    
    # Load XCSP3 2024 CSP track dataset
    dataset = XCSP3Dataset(root=".", year=2024, track="CSP", download=True)
    
    # Create benchmark
    benchmark = XCSP3Benchmark()
    
    # Run with 4 parallel workers
    output_file = benchmark_runner(
        dataset=dataset,
        instance_runner=benchmark,
        output_file="xcsp3_2024_csp_results.csv",
        solver="ortools",
        workers=4,
        time_limit=600,      # 10 minutes per instance
        mem_limit=8192,      # 8 GB per instance
        cores=1,
        verbose=False,
        intermediate=False
    )
    
    print(f"Results written to: {output_file}")

Example: Running OPB Benchmark
------------------------------

.. code-block:: python

    from cpmpy.tools.benchmark.runner import benchmark_runner
    from cpmpy.tools.benchmark.opb import OPBBenchmark
    from cpmpy.tools.datasets import OPBDataset
    
    # Load OPB 2023 dataset
    dataset = OPBDataset(root=".", year=2023, download=True)
    
    # Create benchmark
    benchmark = OPBBenchmark()
    
    # Run benchmark
    output_file = benchmark_runner(
        dataset=dataset,
        instance_runner=benchmark,
        output_file="opb_2023_results.csv",
        solver="ortools",
        workers=8,
        time_limit=300,
        mem_limit=4096
    )

Parallel Execution
------------------

The benchmark runner uses Python's ThreadPoolExecutor for parallel execution:

- Each instance runs in a separate thread
- Instances are isolated from each other
- Results are collected as they complete
- Progress is tracked with a progress bar (if tqdm is available)

Resource Management
--------------------

Each instance execution:

- Runs in isolation with its own resource limits
- Has time and memory limits enforced
- Captures stdout/stderr separately
- Handles timeouts gracefully

Output Format
-------------

The CSV output file contains columns such as:

- **instance_name**: Name of the instance
- **instance_path**: Path to the instance file
- **solver**: Solver used
- **status**: Exit status (optimal, sat, unsat, unknown, etc.)
- **runtime**: Runtime in seconds
- **memory**: Peak memory usage in MiB
- **objective**: Objective value (if applicable)
- **timeout**: Whether instance timed out
- **error**: Error message (if any)

Additional columns may be present depending on the dataset metadata.

Error Handling
--------------

The benchmark runner handles errors gracefully:

- Failed instances don't stop the benchmark
- Errors are logged in the CSV output
- Timeouts are handled separately from crashes
- Memory errors are caught and reported

Progress Tracking
-----------------

If `tqdm` is available, the benchmark runner shows:

- Progress bar with instance count
- Estimated time remaining
- Current instance being processed

Without `tqdm`, progress is printed to stdout.

API Reference
-------------

.. automodule:: cpmpy.tools.benchmark.runner
    :members:
    :undoc-members:
    :inherited-members:
