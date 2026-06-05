# CPMpy Benchmarking

This module contains a collection of tools meant for running experiments with CPMpy in a controlled environment.
Most of this is linked to competitions (MSE, XCSP3, PB), but the runner is generic and can be used for any benchmark run, with build-in support for CPMpy datasets.


## Overview

1) `cpbenchy`

The new "observer pattern"-based benchmark runner. Through registerable observer callbacks, is flexible enough to support any custom benchmark implementation. The core of cpbenchy is just a basic harness to build upon. It contains:

- A) **adapter**: A collection of example benchmark implementations on top of cpbenchy for different competitions and CPMpy datasets
- B) **observer**: A collection of pre-made observers to register with the runner, like a solution printer
- C) **runner**: The actual runner harness, both for individual instances and for complete CPMpy datasets. Build-in support for resource managers (runexec, Python)

2) `competition`

Collection of files related to previous competition submissions.

## Usage

The runner can be found here: `cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py`.

Its interface:

```console
python cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py INSTANCE
    --runner RUNNER
    [--solver SOLVER]
    [--seed RANDOMSEED]
    [--time_limit TIMELIMIT]
    [--mem_limit MEMLIMIT]
    [--cores NBCORES]
    [--intermediate]
    [--verbose]                   # To print output to stdout
    [--output OUTPUT]
    [--observers OBSERVER ...]
    [--solution-checker]
```

`--runner` selects an **adapter**: a competition- or dataset-specific wrapper that knows how to read an instance, print results in the expected format, and register the right observers. Built-in names:

| **Runner** | **For** | **Instance format** |
| - | - | - |
| `xcsp3` (default) | [XCSP3](https://xcsp.org) competition | `.xml` / `.xml.lzma` |
| `opb` | Pseudo-Boolean (PB) competition | `.opb` / `.opb.xz` |
| `mse` | MaxSAT Evaluation (MSE) | `.wcnf` / `.wcnf.xz` |
| `nurserostering` | Nurse rostering benchmarks | schedulingbenchmarks.org format |
| `jsplib` | JSPLib job-shop scheduling | JSPLib format |
| `psplib` | PSPLib resource-constrained project scheduling | RCPSP format |

Example:

```console
python cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py instance.xml.lzma --runner xcsp3 --solver ortools --time_limit 300 --verbose
```

You can also pass a full adapter class path instead of a short name, e.g. `cpmpy.tools.benchmark.cpbenchy.adapter.opb.OPBAdapter` (when you want to use a custom adapter).

Instead of a single `INSTANCE`, you can also pass a batch file or a dataset class (exactly one input mode is required):

```console
python cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py --batch FILE ...
python cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py --dataset DATASET_CLASS ...
    [--workers WORKERS]
    [--cores_per_worker CORES_PER_WORKER]
    [--resource-manager {runexec,python}]
    [--no-pin-cores]
    [--total-memory MiB]
    [--memory-per-worker MiB]
    [--ignore-memory-check]
    [--dataset-root DATASET_ROOT]
    [--dataset-year DATASET_YEAR]
    [--dataset-track DATASET_TRACK]
    [--dataset-download]
    [--dataset-variant DATASET_VARIANT]
    [--dataset-family DATASET_FAMILY]
    [--dataset-option KEY VALUE]
```

XCSP3 dataset example (downloads the 2025 CSP track to `./data`, runs all instances in parallel):

```console
python cpmpy/tools/benchmark/cpbenchy/runner/run_benchmark.py \
    --dataset cpmpy.tools.datasets.xcsp3.XCSP3Dataset \
    --dataset-year 2025 \
    --dataset-track CSP25 \
    --dataset-download \
    --dataset-root ./data \
    --runner xcsp3 \
    --solver ortools \
    --workers 4 \
    --time_limit 300 \
    --output ./results
```