# CPMpy XCSP3 tools

This directory contains a collection of tools for reading, loading and solving XCSP3 instances in CPMpy.

What is included:
- utilities for reading and loading XCSP3 instances
- a PyTorch-compatible dataset class for loading (and downloading) XCSP3 instances
- cli for solving individual XCSP3 instances and outputting the result in the competition format
- cli for benchmarking CPMpy across a large collection of XCSP3 instances
- cli for processing and visualizing benchmark results

The XCSP3-Core specification version 3.2 is currently supported.

## Installation

The XCSP3 tooling has some additional dependencies on top of the base CPMpy. Install them using:

```console
pip install cpmpy[xcsp3]
```

## Utilities

We provide a basic utility for parsing and loading a XCSP3 `.xml` or compressed `.xml.lzma` file into a CPMpy model:

```python
from cpmpy.tools.xcsp3 import read_xcsp3

model = read_xcsp3("<file>")
```


## Dataset

We provide a PyTorch compatible dataset class (`XCSP3Dataset`) to easily work with XCSP3 competition instances. As an example, use the following to install the 2024 instances from the COP track:

```python
from cpmpy.tools.xcsp3 import XCSP3Dataset

dataset = XCSP3Dataset(year=2024, track="COP", download=True)
```

This will install the instances under `<cwd>/2024/COP` as `.xml.lzma` compressed files.

You can now iterate over the dataset and load the instances as CPMpy models:

```python
from cpmpy.tools.xcsp3 import XCSP3Dataset, read_xcsp3

for filename, metadata in XCSP3Dataset(year=2024, track="COP", download=True): # auto download dataset and iterate over its instances
    # Do whatever you want here, e.g. reading to a CPMpy model and solving it:
    model = read_xcsp3(filename)
    model.solve()
    print(model.status())
```

## Solving single instance

To parse, load and solve a single XCSP3 instance, we provide the `xcsp3_cpmpy` CLI.

To use the single-instance CLI:

```python
python xcsp3_cpmpy.py <benchname> --solver <solver> [-s SEED] [-l TIME_LIMIT] [-m MEM_LIMIT] [-t TMPDIR] [-c CORES] [--time-buffer TIME_BUFFER] [--intermediate]
```



## Benchmarking

For benchmarking CPMpy / a backend solver on XCSP3, we provide a CLI to run against a complete competition dataset.

To use the benchmarking CLI:

```python
python xcsp3_benchmark.py --year <YEAR> --track <TRACK> --solver <SOLVER> [--workers WORKERS] [--time-limit TIME_LIMIT] [--mem-limit MEM_LIMIT] [--output-dir OUTPUT_DIR] [--verbose] [--intermediate]
```

This will create a `.csv` file containing (performance) measurements for each of the instances. To compare the results of different solvers: 

```python
python xcsp3_analyze.py <files> [--time_limit TIME_LIMIT] [--output OUTPUT]
```

