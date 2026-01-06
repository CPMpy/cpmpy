# Benchmark Testing Tooling

python cpmpy/tools/benchmark/test/xcsp3_instance_runner.py data/2024/CSP/AverageAvoiding-20_c24.xml.lzma





This directory contains tooling for benchmarking and testing constraint satisfaction problem instances, particularly XCSP3 instances.

## Overview

The tooling provides a flexible framework for:
- Running individual problem instances with various solvers
- Managing computational resources (time, memory, CPU cores)
- Collecting detailed profiling and solution information
- Running benchmarks in parallel across multiple instances

## Components

### Core Components

- **`instance_runner.py`**: Base class for instance runners. Provides the interface for running instances with argument parsing and observer registration.

- **`xcsp3_instance_runner.py`**: Specialized runner for XCSP3 instances. Handles reading compressed (.lzma) and uncompressed XCSP3 files, and sets up appropriate observers for competition-style output.

- **`runner.py`**: Core execution engine that:
  - Reads problem instances
  - Transforms them into solver models
  - Executes solvers with resource limits
  - Manages observers for profiling, solution checking, and output formatting

- **`manager.py`**: Resource management systems:
  - `RunExecResourceManager`: Uses benchexec's RunExecutor for strict resource control (requires benchexec)
  - `PythonResourceManager`: Python-based resource management using observers

### Example Scripts

- **`main.py`**: Example of running parallel benchmarks on XCSP3 datasets with resource management
- **`bench_xcsp3.py`**: Alternative benchmarking script (deprecated, see `run_xcsp3.py`)
- **`run_xcsp3.py`**: Deprecated script (use `XCSP3InstanceRunner` instead)

## Usage

### Running a Single Instance

The simplest way to run a single XCSP3 instance:

```bash
python -m cpmpy.tools.benchmark.test.xcsp3_instance_runner <instance_path> [options]
```

**Options:**
- `--solver SOLVER`: Solver to use (default: "ortools")
- `--output_file FILE`: Output file path (default: `results/{solver}_{instance}.txt`)
- `--time_limit SECONDS`: Time limit in seconds
- `--mem_limit MB`: Memory limit in MB
- `--seed SEED`: Random seed for solver
- `--intermediate`: Print intermediate solutions
- `--cores N`: Number of CPU cores to use
- `--verbose`: Enable verbose output

**Example:**
```bash
python -m cpmpy.tools.benchmark.test.xcsp3_instance_runner instance.xml --solver ortools --time_limit 300 --seed 42
```

### Running Multiple Instances in Parallel

Use `main.py` as a template for running benchmarks on multiple instances:

```python
from cpmpy.tools.benchmark.test.xcsp3_instance_runner import XCSP3InstanceRunner
from cpmpy.tools.benchmark.test.manager import RunExecResourceManager, run_instance
from cpmpy.tools.dataset.model.xcsp3 import XCSP3Dataset
from concurrent.futures import ProcessPoolExecutor
from queue import Queue

# Load dataset
dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)

# Configure resources
time_limit = 600  # 10 minutes
workers = 4
cores_per_worker = 1
memory_limit = 8000  # MB per worker

# Initialize managers
resource_manager = RunExecResourceManager()
instance_runner = XCSP3InstanceRunner()

# Create job queue
job_queue = Queue()
for instance, metadata in dataset:
    job_queue.put((instance, metadata))

# Run with parallel workers
with ProcessPoolExecutor(max_workers=workers) as executor:
    # ... worker setup code ...
```

### Using Resource Managers

#### RunExecResourceManager (Recommended)

Uses benchexec's RunExecutor for strict resource control. Requires `benchexec` to be installed.

```python
from cpmpy.tools.benchmark.test.manager import RunExecResourceManager, XCSP3InstanceRunner

resource_manager = RunExecResourceManager()
runner = XCSP3InstanceRunner()

resource_manager.run(
    instance="instance.xml",
    runner=runner,
    time_limit=300,
    memory_limit=4000,
    cores=[0, 1]  # Use cores 0 and 1
)
```

#### PythonResourceManager

Python-based resource management using observers. Less strict but doesn't require external dependencies.

```python
from cpmpy.tools.benchmark.test.manager import PythonResourceManager, XCSP3InstanceRunner

resource_manager = PythonResourceManager()
runner = XCSP3InstanceRunner()

resource_manager.run(
    instance="instance.xml",
    runner=runner,
    time_limit=300,
    memory_limit=4000,
    cores=[0, 1]
)
```

### Using the Manager CLI

The `manager.py` script provides a command-line interface:

```bash
python -m cpmpy.tools.benchmark.test.manager \
    --instance instance.xml \
    --time_limit 300 \
    --memory_limit 4000 \
    --cores 0,1 \
    --runner xcsp3 \
    --resource_manager runexec
```

**Options:**
- `--instance PATH`: Path to instance file (required)
- `--time_limit SECONDS`: Time limit in seconds
- `--memory_limit MB`: Memory limit in MB
- `--cores LIST`: Comma-separated list of core IDs (e.g., "0,1,2")
- `--runner RUNNER`: Runner to use (default: "xcsp3")
- `--resource_manager MANAGER`: Resource manager ("runexec" or "python", default: "runexec")

## Observers

The runner system uses observers to collect information and format output:

- **`CompetitionPrintingObserver`**: Prints competition-style output (s, v, c lines)
- **`ProfilingObserver`**: Collects timing and resource usage statistics
- **`HandlerObserver`**: Handles exceptions and errors
- **`SolverArgsObserver`**: Logs solver arguments
- **`SolutionCheckerObserver`**: Validates solutions
- **`ResourceLimitObserver`**: Monitors and enforces resource limits

Observers are automatically registered by `XCSP3InstanceRunner`. To add custom observers:

```python
from cpmpy.tools.benchmark.test.runner import YourCustomObserver

runner = XCSP3InstanceRunner()
runner.register_observer(YourCustomObserver())
runner.run(instance="instance.xml")
```

## Output Format

The tooling produces competition-style output:

- `c <comment>`: Comment lines
- `s <status>`: Solution status (SATISFIABLE, UNSATISFIABLE, UNKNOWN)
- `v <values>`: Variable assignments (if solution found)
- `o <objective>`: Objective value (for optimization problems)

Output is written to the specified output file (default: `results/{solver}_{instance}.txt`).

## Supported File Formats

- **XCSP3**: XML-based constraint satisfaction problem format
- **Compressed**: Supports `.lzma` compressed XCSP3 files (automatically detected)

## Dependencies

- **Required**: cpmpy, standard Python libraries
- **Optional**: benchexec (for `RunExecResourceManager`)

## Examples

See `main.py` and `bench_xcsp3.py` for complete examples of running benchmarks.



