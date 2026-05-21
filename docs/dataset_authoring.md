---
title: Dataset authoring
---

# Dataset authoring

This guide explains how to implement new datasets for `cpmpy.tools.datasets`
in a way that is consistent with the rest of the ecosystem:

- stable instance identifiers (`info.id`)
- structured instance metadata (`InstanceInfo`)
- dataset cards and Croissant export (`DatasetInfo`)
- sidecar metadata collection (`.meta.json`)
- PyTorch compatibility (`__len__`, `__getitem__`, `transform`, `target_transform`)

If you only want to *use* existing datasets, start with [](datasets.md).

## Design principles

### (1) Stable instance IDs

Every instance should have a stable identifier. For file-based datasets, the
default `FileDataset` behavior uses the instance file path string as the `id`.

If your dataset is not file-based (or uses nested structures), decide and
document what uniquely identifies an instance. The guiding rule is:

> The dataset class should define what the instance identifier means.

### (2) Metadata fields are a contract

Metadata is a flat dict. The important part is that it is **predictable**:

- problem-level fields: jobs, machines, horizon, …
- format-level fields: opb_*, wcnf_*, dimacs_*, …
- model-level fields: number of variables, constraints, objective info, …

Use `FeaturesInfo` / `FieldInfo` to document the fields your dataset provides.

## Minimal dataset: the required pieces

`FileDataset` is the base for file-backed datasets. A minimal dataset must:

- define class attributes: `name`, `description`, `homepage`
- implement:
  - `category() -> dict` (and/or `categories()` if you want to override)
  - `download()`
- optionally override `parse(instance)` for parse-first datasets

```python
import pathlib
from cpmpy.tools.datasets.core import FileDataset


class MyDataset(FileDataset):
    name = "mydataset"
    description = "A short description of the dataset."
    homepage = "https://example.com/mydataset"

    def __init__(self, root=".", transform=None, target_transform=None, download=False, **kwargs):
        super().__init__(
            dataset_dir=pathlib.Path(root) / self.name,
            extension=".txt",
            transform=transform,
            target_transform=target_transform,
            download=download,
            **kwargs,
        )

    def parse(self, instance):
        # Optional: parse file path to a domain structure
        # (for parse-first workflows with parse=True)
        return self.read(instance)

    def category(self) -> dict:
        # Empty dict if no categories apply
        return {}

    def download(self):
        # Download/extract instances into self.dataset_dir
        raise NotImplementedError
```

## Enriched dataset: optional dataset metadata and a field schema

To make your dataset “self-documenting”, add optional dataset-level attributes
and a `features` schema:

```python
from cpmpy.tools.datasets.metadata import FeaturesInfo, FieldInfo


class MyDataset(FileDataset):
    name = "mydataset"
    description = "A short description."
    homepage = "https://example.com/mydataset"
    citation = ["Author et al. My Dataset. 2026."]

    version = "1.0.0"
    license = "CC BY 4.0"
    domain = "constraint_programming"
    tags = ["combinatorial", "satisfaction"]
    language = "MyFormat"

    features = FeaturesInfo({
        "num_jobs": ("int", "Number of jobs in the instance"),
        "num_machines": ("int", "Number of machines"),
        "optimum": FieldInfo("int", "Known optimum (if available)", nullable=True),
    })
```

This schema is used for dataset cards and Croissant export; it does not change
how iteration works. If you do not provide `features`, cards and Croissant
still work but omit the domain-field schema section. For **per-field defaults**
and **what you lose** when omitting or simplifying the schema (e.g. empty
descriptions, default nullability), see [Instance Metadata — Level 7:
Declaring a metadata schema](instance_metadata.md#level-7--declaring-a-metadata-schema).

## Collecting instance metadata

Override `collect_instance_metadata(file)` to extract domain-specific metadata
once per instance (stored in sidecars by default):

```python
class MyDataset(FileDataset):
    # ... class attrs ...

    def collect_instance_metadata(self, file) -> dict:
        # `file` is the file path string by default
        return {"num_jobs": 10, "num_machines": 5}
```

## Sidecars and advanced kwargs

`FileDataset` supports two advanced constructor kwargs:

- `metadata_workers` (default: 1): number of workers used when collecting all
  instance metadata after download.
- `ignore_sidecar` (default: False): do not read/write sidecars; instead call
  `collect_instance_metadata()` on demand when iterating.

These kwargs are passed via `**kwargs` and unknown kwargs are ignored to keep
forward compatibility.

## Authoring loaders and writers (the `open=` convention)

All IO loaders accept an optional `open=` callable, so callers can control how
files are opened (e.g., decompression). Writers follow the same convention:
they accept an `open=` callable for writing.

Example: implementing a custom loader that supports compressed files:

```python
import os
from typing import Union
from io import StringIO

_std_open = open

def load_myformat(data: Union[str, os.PathLike], open=open):
    if isinstance(data, (str, os.PathLike)) and os.path.exists(data):
        f = open(data) if open is not None else _std_open(data, "rt")
    else:
        f = StringIO(data)
    # parse from `f`...
```

Example: writing with a custom `open` (compression decided by the caller):

```python
import lzma
from cpmpy.tools.io.opb import write_opb

xz_text = lambda path, mode="w": lzma.open(path, "wt")
write_opb(model, "out.opb.xz", open=xz_text)
```

## Extending existing datasets (schema inheritance)

If you subclass an existing dataset and add only a few new metadata fields,
declare only the new fields in `features`. The framework merges parent and child
schemas automatically.

See `libraries/cpmpy/examples/datasets/05_features_merge.py` for a runnable
example.
