---
title: Datasets
---

# Datasets

CPMpy provides a PyTorch-style dataset interface for working with collections of
benchmark instances. Datasets handle:

- downloading and local storage
- instance discovery (files)
- per-instance metadata collection (sidecars)
- optional decompression on read
- optional transforms (load, translate, save, etc.)

The goal is that you can write experiments in a **data-loader style loop**:
each item yields `(x, y)` where `x` is the instance reference and `y` is the
metadata record.

This page starts with a quickstart, then shows common pipelines, and finally
points to the advanced authoring guides.

## Quickstart

### 1) Iterate over instances

Datasets yield `(file_path, info)` pairs:

```python
from cpmpy.tools.datasets import JSPLibDataset

ds = JSPLibDataset(root="./data", download=True)
print(len(ds), "instances")

for file_path, info in ds:
    print(info["name"], info.get("jobs"), info.get("machines"))
```

`info` is an `InstanceInfo` (a `dict` subclass) with structured properties.
See the dedicated metadata guide at [](instance_metadata.md).

### 2) Load each instance into a CPMpy model

Because datasets are PyTorch-compatible, the most direct pattern is to use the
loader as the dataset transform:

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.io import load_xcsp3

ds = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)
ds.transform = load_xcsp3

for model, info in ds:
    if model.solve():
        print(info.id, "objective:", model.objective_value() if model.has_objective() else None)
```

If files are compressed, keep decompression support by wrapping the IO loader:
`lambda p: load_xcsp3(p, open=ds.open)`.

### 3) Add computed fields via `target_transform`

Use `target_transform` when you want to enrich metadata without modifying your
loop:

```python
from cpmpy.tools.datasets import JSPLibDataset

ds = JSPLibDataset(
    root="./data",
    download=True,
    target_transform=lambda info: info | {
        "density": info["jobs"] / info["machines"],
        "has_optimum": info.get("optimum") is not None,
    },
)

for _, info in ds:
    print(info.id, info["density"], info["has_optimum"])
```

### 4) Parse-first datasets (two-step or compact)

Some datasets represent domain data, not a fixed CPMpy model. For those
datasets, enable `parse=True` and either model in a second step or pass a model
builder as `transform`.

```python
from cpmpy.tools.datasets import PSPLibDataset, model_rcpsp

# Two-step: parse first, model later
ds = PSPLibDataset(variant="rcpsp", family="j60", download=True, parse=True)
for (tasks, capacities), info in ds:
    model, (start, end, makespan) = model_rcpsp(tasks, capacities)
    model.solve()

# Compact: parse + model in dataset pipeline
ds = PSPLibDataset(
    variant="rcpsp",
    family="j60",
    download=True,
    parse=True,
    transform=model_rcpsp,
)
for (model, aux), info in ds:
    model.solve()
```

## Common pipelines

### Load → Translate → Save (format conversion)

Use transform composition:

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.datasets.transforms import Compose, Translate, SaveToFile

ds = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)

ds.transform = Compose([
    Translate(load_xcsp3, "opb", open=ds.open),          # file_path -> OPB string
    SaveToFile("./out_opb/", extension=".opb", write_metadata=True),
])

for output_path, info in ds:
    print("saved", output_path, "id=", info.id)
```

When `write_metadata=True`, a `.meta.json` sidecar is written next to each
output file. It contains portable metadata (domain fields, format fields,
model features), but never in-memory objects (see `model_objects` in
[](instance_metadata.md)).

### Load → Save → Reload from files (generic dataset)

You can translate a named dataset to a format (e.g. OPB), write instances to
a directory, and later iterate over that directory **without** a dedicated
dataset class for the translated format. Use the `from_files()` helper to build
a generic file-based dataset over any directory:

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.datasets.core import from_files
from cpmpy.tools.datasets.transforms import Compose, Translate, SaveToFile
from cpmpy.tools.io import load_opb, load_xcsp3

# 1) Load, translate to OPB, write to disk (with metadata sidecars)
ds = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)
ds.transform = Compose([
    Translate(load_xcsp3, "opb", open=ds.open),
    SaveToFile("./out_opb/", extension=".opb", write_metadata=True),
])
for out_path, info in ds:
    pass   # files written to ./out_opb/

# 2) Later: open the same directory as a generic dataset (no XCSP3 class needed)
generic = from_files("./out_opb/", extension=".opb")
generic.transform = load_opb   # or lambda p: load_opb(p, open=open)

for model, info in generic:
    print(info["name"], info.get("path"))   # minimal metadata; .meta.json can be read separately
```

`from_files(dataset_dir, extension)` returns a `FileDataset` that discovers
all files with the given extension under `dataset_dir` (including subdirs). It
does not provide a dataset name, description, or card/Croissant; metadata is
minimal (path, name, id). To reuse the metadata written by `SaveToFile`, read
the `.meta.json` sidecar next to each file (e.g. in a `target_transform`).

### Using already-downloaded files (custom directory)

If you have instance files on disk already (e.g. from another source or a
previous run), point the dataset at that directory instead of downloading:

- **Same layout as the dataset expects:** use the usual class with `root` set
  to the parent of the dataset folder, and `download=False`:

  ```python
  # JSPLib expects root/jsplib/; your files are in /data/my_jsplib/
  ds = JSPLibDataset(root="/data", download=False)
  # Then set dataset_dir to your folder, or use a symlink /data/jsplib -> /data/my_jsplib
  ```

  Concrete dataset classes typically set `dataset_dir = root / self.name` (or
  `root / self.name / year / track`). So put your files under that path, or
  pass a custom `dataset_dir` when the constructor supports it.

- **Datasets that accept `dataset_dir`:** e.g. `MaxSATEvalDataset` and others
  take an optional `dataset_dir`; if provided, it overrides the default
  `root/name/...`:

  ```python
  from cpmpy.tools.datasets import MaxSATEvalDataset

  ds = MaxSATEvalDataset(
      root="./data",
      year=2022,
      track="exact-unweighted",
      dataset_dir="/path/to/my/wcnf/files",   # use this instead of downloading
      download=False,
  )
  for path, info in ds:
      ...
  ```

- **Arbitrary directory, any extension:** use `from_files(dataset_dir, extension)`
  as in the previous section (no download, no dedicated class).

### Generator-based datasets

For procedurally generated instances (e.g. random graphs, parameter sweeps),
use `IterableDataset.from_generator()`. You provide a generator function that
yields `(instance_ref, metadata)` pairs and optional keyword arguments;
optionally vary some arguments to get multiple generator runs:

```python
from cpmpy.tools.datasets.core import IterableDataset

def my_generator(n, seed):
    import random
    rng = random.Random(seed)
    for i in range(n):
        # instance_ref: e.g. dict of parameters or a file path
        ref = {"n": n, "seed": seed, "instance_id": i}
        meta = {"name": f"gen_{n}_{seed}_{i}"}
        yield ref, meta

# Single run
ds = IterableDataset.from_generator(my_generator, gen_kwargs={"n": 5, "seed": 42})
for ref, info in ds:
    print(info["name"])

# Multiple runs: vary "seed"
ds = IterableDataset.from_generator(
    my_generator,
    gen_kwargs={"n": 5, "seed": [10, 20, 30]},
    vary="seed",
)
# Iteration runs my_generator(n=5, seed=10), then (n=5, seed=20), then (n=5, seed=30)
```

Generator datasets do not support `len()` or indexing; they are iterable only.
See the `IterableDataset.from_generator` docstring for `vary` with multiple keys
and `vary_mode="product"` for Cartesian products.

### Load models and run analytics (solver-style preprocessing)

If you want to use CPMpy's internal transformation pipeline on loaded models
(like solvers do), see [](transforms_guide.md) for end-to-end examples.

## Sidecars and metadata collection

By default, file-based datasets collect instance metadata once and store it
in a `.meta.json` sidecar next to each instance file. Subsequent accesses use
the sidecar and avoid re-computing metadata.

Advanced constructor kwargs (documented in detail in [](dataset_authoring.md)):

- `metadata_workers`: parallelism for metadata collection during initial download
- `ignore_sidecar`: bypass sidecar read/write and collect metadata on demand

## Where to look next

- [](instance_metadata.md): `InstanceInfo`, structured partitions, `|`, and `without_format()`
- [](reading_and_writing.md): IO loaders/writers and dataset translation workflows
- [](dataset_authoring.md): implementing a new dataset class (best practices + checklists)
- [](transforms_guide.md): custom transforms, `enrich_metadata`, analytics pipelines
- [](benchmarking_workflows.md): dataset-driven experiments and transformation comparisons

## Runnable examples

The `libraries/cpmpy/examples/datasets/` directory contains runnable examples
that match the docs:

- `01_basic_usage.py`
- `02_dataset_card_and_croissant.py`
- `03_target_transforms.py`
- `04_custom_dataset.py`
- `05_features_merge.py`
- `06_benchmark_survey.py`
- `07_metadata_enrichment.py`
