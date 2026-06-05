---
title: Transforms guide
---

# Transforms guide

Datasets support PyTorch-style transforms:

- `transform`: applied to the instance reference (`x`) during iteration
- `target_transform`: applied to the metadata record (`y`)

Transforms are the intended way to build **pipelines**:
load → preprocess → analyze → translate → save.

This guide explains:

- the transform protocol (`__call__`, optional `enrich_metadata`)
- composition patterns
- metadata enrichment patterns that keep records portable
- using `cpmpy.transformations.*` for solver-style preprocessing and analytics

## The transform protocol

Any callable can be used as `dataset.transform`. If it is an object with an
`enrich_metadata(data, metadata)` method, the dataset will call that method
after `__call__` and use its return value as the updated metadata.

Conceptually:

```text
file_path -> transform(file_path) -> data
metadata  -> target_transform(metadata) -> metadata'
```

and optionally:

```text
metadata -> transform.enrich_metadata(data, metadata) -> metadata'
```

## Built-in transforms

The module `cpmpy.tools.datasets.transforms` provides common building blocks:

- `Open`: read raw file contents (decompression via an `open=` callable)
- `Load`: parse file content into a CPMpy model; enriches metadata with model features and decision variables
- `Serialize`: serialize a CPMpy model to a target format string
- `Translate`: Load + Serialize in one step
- `SaveToFile`: write content to disk (and optionally `.meta.json` sidecars)
- `Compose`: chain multiple transforms
- `Lambda`: wrap any callable as a named transform

## Common patterns

### Load models and solve

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.io import load_xcsp3

ds = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)
ds.transform = load_xcsp3

for model, info in ds:
    model.solve()
```

If you want metadata enrichment (model statistics + variables), use
`Load`:

```python
from cpmpy.tools.datasets.transforms import Load

ds.transform = Load(load_xcsp3, open=ds.open)
for model, info in ds:
    model.solve()
    print(info.model_features)  # populated by Load
```

### Translate and save

```python
from cpmpy.tools.datasets.transforms import Compose, Translate, SaveToFile

ds.transform = Compose([
    Translate(load_xcsp3, "opb", open=ds.open),
    SaveToFile("./out_opb/", extension=".opb", write_metadata=True),
])
```

## Metadata enrichment patterns

### Add computed fields (portable)

Use `|` to merge fields into the metadata record:

```python
ds = JSPLibDataset(
    root="./data",
    target_transform=lambda info: info | {
        "density": info["jobs"] / info["machines"],
    },
)
```

### Format-changing transforms: drop stale format fields

When a transform changes the file format, old format-prefixed fields become
misleading. Use `without_format()` to strip format fields and then attach the
new ones:

```python
from cpmpy.tools.datasets.transforms import extract_format_metadata

new_info = info.without_format() | extract_format_metadata(opb_string, "opb")
```

### Implement `enrich_metadata` in a transform

If the metadata update depends on the transform output, implement
`enrich_metadata(data, metadata)`:

```python
from cpmpy.tools.datasets.transforms import Translate, extract_format_metadata

class TranslateToOPB:
    def __init__(self, loader, open):
        self._translate = Translate(loader, "opb", open=open)

    def __call__(self, file_path):
        return self._translate(file_path)  # OPB string

    def enrich_metadata(self, data, metadata):
        return metadata.without_format() | extract_format_metadata(data, "opb")
```

## Solver-style preprocessing and analytics

CPMpy has an internal transformation toolbox under `cpmpy.transformations`.
Solvers use these transformations to rewrite high-level constraints into
supported low-level forms.

You can use the same transformations for analytics:

- how many constraints/variables are introduced by a decomposition?
- how long does a preprocessing pipeline take?
- does a decomposition improve solve time on a dataset subset?

### A minimal “preprocess then measure” pattern

```python
import time
from cpmpy.transformations.flatten_model import flatten_model
from cpmpy.transformations.get_variables import get_variables_model

t0 = time.perf_counter()
flat = flatten_model(model)
dt = time.perf_counter() - t0

num_vars = len(get_variables_model(flat))
num_cons = len(flat.constraints)
print("flatten:", dt, "vars:", num_vars, "cons:", num_cons)
```

### Decomposing a specific global constraint

The `decompose_global` transformation can decompose unsupported globals. You can
also provide custom decompositions to compare strategies in an experiment.

For a full runnable example that compares two strategies across a dataset
subset, see [](benchmarking_workflows.md) and the example script
`libraries/cpmpy/examples/datasets/08_transformation_benchmark.py`.
