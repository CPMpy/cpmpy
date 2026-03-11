# Reading, Writing and Datasets

CPMpy provides a suite of tools for working with different file formats and benchmark sets
from the various communities within Constraint Optimization (CO). They enable simple
programmatic access to these resources and facilitate cross-community access to benchmarks
and systematic comparisons of solvers across paradigms. 

More concretely, we provide a set of readers, loaders (loading problem files into CPMpy model),
datasets, metadata, transformations, writers, etc, all to lower the barrier to entry to experiment
with these benchmarks. 

The dataset class that we provide is PyTorch compatible, allowing for integration within larger
systems within the scientific field. Whilst this tooling builds on top of CPMpy to provide the 
transformation capabilities, its programmatic abstraction of CO benchmarks can be used in 
combination with any (constraint modelling) system. 

This guide walks you through everything, from a simple one-liner for downloading instance files 
to instructions on how to write your own dataset class.

---

## Supported Formats

| Format | Extension | Load | Write | Domain |
|--------|-----------|------|-------|--------|
| **OPB** | `.opb` | ✅ | ✅ | Pseudo-Boolean optimization |
| **WCNF** | `.wcnf` | ✅ | — | MaxSAT |
| **DIMACS** | `.cnf` | ✅ | ✅ | SAT |
| **MPS** | `.mps` | ✅ | ✅ | Mixed integer programming |
| **LP** | `.lp` | ✅ | ✅ | Linear/integer programming |
| **FZN** | `.fzn` | ✅ | ✅ | FlatZinc (MiniZinc) |
| **CIP** | `.cip` | ✅ | ✅ | Constraint integer programming |
| **GMS** | `.gms` | ✅ | ✅ | GAMS |
| **PIP** | `.pip` | ✅ | ✅ | Pseudo integer programming |
| **XCSP3** | `.xml` | ✅ | — | Constraint satisfaction/optimization |
| **JSPLib** | (none) | ✅ | — | Job Shop Scheduling |
| **PSPLib** | `.sm` | ✅ | — | Project Scheduling (RCPSP) |
| **NRP** | `.txt` | ✅ | — | Nurse Rostering |

---

## Loading and Writing Files

### Loading a file

The `load` function auto-detects the format from the file extension:

```python
from cpmpy.tools.io import load

model = load("instance.opb")   # format detected from extension
model = load("instance.cnf")
model = load("problem.mps")
```
If the exension does not reveal the intended format, one can also manually provide it:

```python
model = load("instance.txt", format="opb")
```

For format-specific control, use the dedicated loaders directly:

```python
from cpmpy.tools.io.opb import load_opb
from cpmpy.tools.io.wcnf import load_wcnf
from cpmpy.tools.io.dimacs import load_dimacs

model = load_opb("instance.opb")
model = load_wcnf("instance.wcnf")
model = load_dimacs("instance.cnf")

# Formats backed by SCIP (requires pyscipopt)
from cpmpy.tools.io.scip import load_scip
model = load_scip("instance.mps", format="mps")
model = load_scip("instance.lp",  format="lp")
```

<!-- Domain-specific loaders parse structured instance files into CPMpy models:

```python
from cpmpy.tools.io.jsplib import load_jsplib
from cpmpy.tools.io.rcpsp import load_rcpsp
from cpmpy.tools.io.nurserostering import load_nurserostering

model = load_jsplib("instance")          # Job Shop Scheduling
model = load_rcpsp("instance.sm")        # Resource-Constrained Project Scheduling
model = load_nurserostering("inst.txt")  # Nurse Rostering
``` -->

All loaders also accept raw content strings. Useful when the content was already read into memory (or when creating your own content strings through a generator, see next section):

```python
with open("instance.opb") as f:
    content = f.read()
model = load_opb(content)   # raw string works too
```

### Programmatic construction (problem generators)

Because loaders accept strings, you can *generate* instance content programmatically
and let CPMpy parse it into a model. This is useful for problem generators,
random instance sampling, or templating — no need to write files to disk.

**JSPLib** (Job Shop Scheduling): each line after the header is one job; pairs
are (machine, duration) per task:

```python
from cpmpy.tools.io.jsplib import load_jsplib

def make_jsplib(n_jobs, n_machines, durations):
    """Build a JSPLib string from job data. durations: list of (machine, dur) per job."""
    lines = [f"{n_jobs} {len(durations[0])}"]
    for job in durations:
        parts = [f"{m} {d}" for m, d in job]
        lines.append(" ".join(parts))
    return "\n".join(lines)

# Example: 2 jobs × 2 tasks
content = make_jsplib(2, 2, [[(0, 5), (1, 3)], [(1, 2), (0, 4)]])
# content is:
# 2 2
# 0 5 1 3
# 1 2 0 4
model = load_jsplib(content)
model.solve()
```

**Nurse Rostering** (NRP): the format uses tagged sections (`SECTION_HORIZON`,
`SECTION_SHIFTS`, `SECTION_STAFF`, `SECTION_COVER`, etc.). Build each section
as a string (e.g. from templates or loops over staff/shift data) and join them;
then pass the full string to `load_nurserostering`. See the [NRP format
guide](https://schedulingbenchmarks.org/nrp/instances1_24.html) for the required
structure.

**OPB** and other flat formats work the same way — construct the string, pass it
to the loader, and CPMpy returns a ready-to-solve model. This pattern is
especially handy for JSPLib, Nurse Rostering, and similar structured text formats
where you want to vary parameters or generate instances on the fly without
creating temporary files.

### Writing a model

Writing a model from CPMpy back to file is a very similar process to loading:

```python
import cpmpy as cp
from cpmpy.tools.io import write

x = cp.intvar(0, 10, name="x")
y = cp.intvar(0, 10, name="y")
model = cp.Model([x + y <= 5], minimize=x + y)

write(model, "output.opb")             # format auto-detected from extension
write(model, "out.txt", format="opb")  # explicit format

# Write to string instead of file (returns the string)
opb_string = write(model, format="opb")
```

Again, you can also directly use the format-specific writer functions:

```python
from cpmpy.tools.io.opb import write_opb
from cpmpy.tools.io.dimacs import write_dimacs
from cpmpy.tools.io.scip import write_scip

write_opb(model, "output.opb")
write_dimacs(model, "output.cnf")
write_scip(model, "output.mps", format="mps")
write_scip(model, "output.fzn", format="fzn")
```

### Handling compressed files

Many benchmark archives use `.xz` or `.lzma` compression. Pass a custom `open`
argument to any loader:

```python
import lzma
from cpmpy.tools.io.opb import load_opb

model = load_opb("instance.opb.xz", open=lzma.open)
```

Writers follow the same convention: pass an `open` callable to control how the
output file is opened (for example to write compressed output):

```python
import lzma
from cpmpy.tools.io.opb import write_opb

xz_text = lambda path, mode="w": lzma.open(path, "wt")
write_opb(model, "output.opb.xz", open=xz_text)
```

---

## Datasets

CPMpy datasets provide a PyTorch-style interface for collections of well-known 
CO benchmark instances: download with a single one-liner, iterate over the files
in `(file_path, metadata)` pairs and use built-in transforms for loading and translation.


### Available datasets

| Class | Domain | Format |
|-------|--------|--------|
| `XCSP3Dataset` | CP/COP | XCSP3 |
| `OPBDataset` | Pseudo-Boolean | OPB |
| `MaxSATEvalDataset` | MaxSAT | WCNF |
| `JSPLibDataset` | Job Shop Scheduling | JSPLib |
| `PSPLibDataset` | Project Scheduling | PSPLib |
| `NurseRosteringDataset` | Nurse Rostering | NRP |
| `MIPLibDataset` | Mixed Integer Programming | MPS |
| `SATDataset` | SAT | DIMACS (CNF) |

### Basic iteration

You can simply access the data within a dataset by iterating over its included
problem instances. If the data is not yet locally available, pass the `download=True`
optional argument and the dataset will be auto-downloaded from its original source.

```python
from cpmpy.tools.datasets import JSPLibDataset

dataset = JSPLibDataset(root="./data", download=True)
print(len(dataset), "instances")

for file_path, info in dataset:
    ...
```

Iterating over a dataset always returns 2-tuples. The first element is a problem instance identifier.
For now, all datasets are file-based, and thus the identifier will always be a filepath to the instance file.
In the furure this could hold other identifiers, like a database query.

<!-- TODO, also add link for next section -->
The second element `info` is an `InstanceInfo` — a dict subclass described in
detail in the next section. It contains the metadata, both of the instance that it gets paired with and of
the dataset as a whole. More info on metadata can be found in ...

### Loading instances into CPMpy models

Use an IO loader as the `transform` argument (PyTorch-style): 

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.io import load_xcsp3

dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True, 
                       transform=load_xcsp3)

for model, info in dataset:
    model.solve()
```

Alternatively, call an IO loader on demand inside the loop:

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.io import load_xcsp3

dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)

for file_path, info in dataset:
    model = load_xcsp3(file_path, open=dataset.open)
    model.solve()
```

For more advanced loading, e.g. when you need a custom `open` callable, see [Dataset transform helpers](#dataset-transform-helpers-pytorch-style) for details.

For adding model-level metadata (e.g. `model_features`, `model_objects`) via
transforms, see [Transform metadata enrichment](#transform-metadata-enrichment-advanced) below.

### Translating to another format

You can translate each instance to another format by looping over the dataset,
loading the instance into a CPMpy model, and calling a writer (or the unified
`write` function) to get a string or write to file.

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.io import write
from cpmpy.tools.io import load_xcsp3

dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)

for file_path, info in dataset:
    model = load_xcsp3(file_path, open=dataset.open)
    opb_string = write(model, format="opb")   # or write(model, "out.opb")
    print(info.id, len(opb_string), "bytes")
```

For a one-step transform that does load + serialize in the pipeline (with
optional custom `open` and metadata enrichment), use the `Translate` helper;
see [Dataset transform helpers](#dataset-transform-helpers-pytorch-style).

### Saving translated instances to disk

Loop over the dataset, load each instance, and write the model to a file in the
target format. Use the instance metadata (e.g. `info.id`) to build output paths
if you want one file per instance. You can optionally write a `.meta.json` sidecar
yourself, or use the `SaveToFile` helper in the pipeline; see
[Dataset transform helpers](#dataset-transform-helpers-pytorch-style).

```python
from pathlib import Path
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.io import write, load_xcsp3

dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)
out_dir = Path("./translated")
out_dir.mkdir(parents=True, exist_ok=True)

for file_path, info in dataset:
    model = load_xcsp3(file_path, open=dataset.open)
    out_path = out_dir / f"{info.id.replace('/', '_')}.opb"
    write(model, str(out_path))   # format inferred from extension
    print("Saved:", out_path)
```

### Dataset transform helpers (PyTorch-style)

The `cpmpy.tools.datasets.transforms` module provides composable transform classes
that you can assign to `dataset.transform` (or use inside `Compose`):

| Helper | Purpose |
|--------|---------|
| **`Load`** | Load a file path into a CPMpy model. Accepts a custom `open` callable (e.g. for compressed files) and implements `enrich_metadata` to add `model_features` and `model_objects` to the instance metadata. |
| **`Open`** | Open a file path and return its raw text contents (with optional custom `open` for decompression). No parsing. |
| **`Serialize`** | Turn a CPMpy model into a string in a given format (e.g. `"opb"`, `"dimacs"`, `"mps"` or a writer function). |
| **`Translate`** | Load from one format and serialize to another in one step (e.g. XCSP3 → OPB). Uses a custom `open` for reading and enriches metadata from the intermediate model. |
| **`SaveToFile`** | Write the transform output (e.g. a string) to a file under a given directory; optional `.meta.json` sidecar. |
| **`Compose`** | Chain several transforms; each step's output is passed to the next, and each step's `enrich_metadata` (if present) is called with its own output. |
| **`Lambda`** | Wrap a callable as a transform (e.g. `Lambda(lambda path: path.strip())`). |

Example — load with custom `open` and metadata enrichment:

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.datasets.transforms import Load
from cpmpy.tools.io import load_xcsp3

dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)
dataset.transform = Load(load_xcsp3, open=dataset.open)
for model, info in dataset:
    # info.model_features, info.model_objects are populated by Load
    model.solve()
```

Example — translate to another format on the fly:

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.datasets.transforms import Translate
from cpmpy.tools.io import load_xcsp3

dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)
dataset.transform = Translate(load_xcsp3, "opb", open=dataset.open)

for opb_string, info in dataset:
    print(len(opb_string), "bytes")
```

`Translate` accepts a format name string (`"opb"`, `"dimacs"`, `"mps"`, …) or a
writer function directly. Under the hood it loads the instance into a CPMpy model
and serializes it to the target format.

Example — translate and save to disk (with optional metadata sidecar):

```python
from cpmpy.tools.datasets import XCSP3Dataset
from cpmpy.tools.datasets.transforms import Compose, Translate, SaveToFile

dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)

dataset.transform = Compose([
    Translate(load_xcsp3, "opb", open=dataset.open),
    SaveToFile("./translated/", extension=".opb", write_metadata=True),
])

for output_path, info in dataset:
    print("Saved:", output_path)
```

`SaveToFile` with `write_metadata=True` writes a `.meta.json` sidecar alongside
each file, capturing the portable instance metadata.

Example — load to model, then serialize to string (Compose):

```python
from cpmpy.tools.datasets.transforms import Compose, Load, Serialize
from cpmpy.tools.io import load_xcsp3

dataset.transform = Compose([
    Load(load_xcsp3, open=dataset.open),
    Serialize("opb"),
])

for opb_string, info in dataset:
    ...
```

For more examples and custom transforms, see the [Transforms guide](transforms_guide.md).

### Transform metadata enrichment (advanced)

Transforms can be *classes* that implement an `enrich_metadata(self, data, metadata)` method. After each item is produced, the dataset calls this method so the transform can add or update metadata based on its output (e.g. the loaded model). That is how fields like `model_features` (variable/constraint counts, objective info) and `model_objects` (e.g. `variables`) appear in `info` when using `Load` — `Load` implements `enrich_metadata` and fills those in from the CPMpy model.

Any custom transform class can do the same: implement `__call__` for the transformation and `enrich_metadata` to update metadata from the result. The dataset calls `enrich_metadata` automatically after `__call__`. For full details and examples, see [Updating metadata from a transform](#updating-metadata-from-a-transform-enrich_metadata) in the Enriching Metadata section.

---

## Instance Metadata (`InstanceInfo`)

Every dataset iteration yields an `InstanceInfo` as the second element.
`InstanceInfo` is a plain dict subclass — dict access works
unchanged — with additional structured properties.

### Dict access

```python
file, info = dataset[0]

info["name"]          # "abz5"
info.get("jobs", 0)   # 10
"optimum" in info     # True
```

### Structured properties

```python
info.id               # "jsplib/abz5"  — stable slash-separated identifier
info.domain_metadata  # {"jobs": 10, "machines": 10, "optimum": 1234, …}
info.format_metadata  # {"opb_num_variables": …}  — only if in an OPB format
info.model_features   # {"num_variables": …, "objective": …}  — only after Load
info.model_objects    # {"variables": {name: var}}  — only after Load
```

The four metadata partitions:

| Property | What it contains | Serializable |
|----------|-----------------|:---:|
| `domain_metadata` | Problem-level, format-independent fields (`jobs`, `machines`, `horizon`, …) | ✅ |
| `format_metadata` | Format-specific fields (`opb_*`, `wcnf_*`, `mps_*`, `xcsp_*`, `dimacs_*`) | ✅ |
| `model_features` | CP model statistics: variable counts, constraint counts, objective info | ✅ |
| `model_objects` | Live CPMpy objects: `variables` map — **only in-memory when the transform returns a CPMpy model (e.g. `load_*`, `Load`, `Translate`)** | ❌ |

### Reading solution values from metadata

Any transform that returns a CPMpy model (including `cpmpy.tools.io.load_*`
functions used as dataset transforms) populates `info.model_objects["variables"]` with a
`{name: CPMpy_variable}` mapping. After solving, you can read values directly
from that map without needing a separate reference to the variables:

```python
from cpmpy.tools.datasets import JSPLibDataset
from cpmpy.tools.io import load_jsplib

dataset = JSPLibDataset(root="./data")
dataset.transform = load_jsplib

for model, info in dataset:
    if model.solve():
        vars = info.model_objects["variables"]
        print(f"{info['name']}: objective = {model.objective_value()}")
        for name, var in vars.items():
            print(f"  {name} = {var.value()}")
```

`model_objects` is intentionally excluded from [to_croissant()](#converting-to-standard-formats),
[to_gbd()](#converting-to-standard-formats), and [.meta.json sidecars](instance_metadata.md#why-model-objects-live-in-metadata) — the live variable objects exist
only for the duration of one iteration and cannot be serialised.

### Converting to standard formats

Instance metadata can be exported into standardized, interchange formats so that
benchmark records can be consumed by other tools, ML pipelines, or databases
without relying on CPMpy-specific types. Each format produces a plain Python
dict (JSON-serialisable) with a stable set of fields. Additional formats may
be added in future releases.

| Format      | Standard / use case                         | Method / adapter        |
|------------|----------------------------------------------|-------------------------|
| **Croissant** | MLCommons Croissant 1.0 (dataset metadata) | `info.to_croissant()`   |
| **GBD**       | Global Benchmark Database-style features   | `info.to_gbd()`         |

```python
info.to_croissant()  # flat dict record for Croissant-style export
info.to_gbd()        # flat dict record for GBD-style export
```

These adapters can also be passed directly as `target_transform`:

```python
from cpmpy.tools.datasets.metadata import to_croissant

dataset = JSPLibDataset(root="./data", target_transform=to_croissant)
for file_path, record in dataset:
    print(record["id"], record["jobs"])  # plain dict, Croissant-compatible
```

---

## Enriching Metadata

### Adding fields (most common)

Return a plain dict delta from `target_transform` or use `|` inside the loop.
Everything else in `info` is preserved automatically.

```python
# Via target_transform — applied automatically on every item
dataset = JSPLibDataset(
    root="./data",
    target_transform=lambda info: info | {
        "density":     info["jobs"] / info["machines"],
        "has_optimum": info.get("optimum") is not None,
    },
)

for file_path, info in dataset:
    print(info["density"], info["has_optimum"])
```

```python
# Or directly in the loop
for file_path, info in dataset:
    enriched = info | {"difficulty": compute_difficulty(file_path)}
```

The `|` operator always returns a new `InstanceInfo`, so structured properties
remain available on the result.

### Changing format

When a transform produces a different file format, the old format-specific fields
should be dropped and new ones added. `without_format()` handles the drop;
chain it with `|` to add the new fields:

```python
from cpmpy.tools.datasets.transforms import extract_format_metadata

for opb_string, info in dataset:
    new_info = info.without_format() | extract_format_metadata(opb_string, "opb")
    #   ↑ domain_metadata carried forward     ↑ new opb_* fields added
    print(new_info["jobs"])              # still there
    print(new_info["opb_num_variables"]) # new
```

`without_format()` with no arguments strips format fields and carries everything
else forward. Chaining with `|` is optional — omit it if you just want to strip:

```python
stripped = info.without_format()
assert not stripped.format_metadata
```

### Updating metadata from a transform (`enrich_metadata`)

When you write a custom transform class, implement `enrich_metadata(self, data,
metadata)` to update metadata based on the transform's output. It is called
automatically by the dataset after `__call__` returns.

```python
from cpmpy.tools.datasets.transforms import Translate, extract_format_metadata

class TranslateToOPB:
    """Translate a JSPLib instance to OPB format, updating metadata."""

    def __init__(self, loader, open):
        self._translate = Translate(loader, "opb", open=open)

    def __call__(self, file_path):
        self._last_output = self._translate(file_path)
        return self._last_output

    def enrich_metadata(self, data, metadata):
        # data  = OPB string from __call__
        # metadata = current InstanceInfo
        return metadata.without_format() | extract_format_metadata(data, "opb")


dataset = JSPLibDataset(root="./data")
dataset.transform = TranslateToOPB(load_jsplib, open=dataset.open)

for opb_string, info in dataset:
    print(info["jobs"])               # domain field: carried forward
    print(info["opb_num_variables"])  # populated from new format
```

---

## Dataset-Level Metadata

Every dataset class carries a `DatasetInfo` object with name, homepage, citation, etc, and a schema of the instance-level fields.

The instance field schema lives in `info.features`: it is a `FeaturesInfo` object whose `fields` attribute is a dict mapping each field name to a `FieldInfo` (with `dtype`, `description`, and optionally `nullable` and `example`). Iterating over it lets you inspect what metadata fields the dataset declares and their types and descriptions:

```python
info = JSPLibDataset.dataset_metadata()   # no instance needed

info.name        # "jsplib"
info.homepage    # "https://github.com/tamy0612/JSPLIB"
info.citation    # ["J. Adams et al. …"]

# Instance field schema: field_name → FieldInfo (dtype, description, nullable, example)
for field_name, fi in info.features.fields.items():
    print(field_name, fi.dtype, fi.description)
# Example output:
#   jobs int Number of jobs
#   machines int Number of machines
#   optimum int Known optimal makespan, if available
#   bounds dict Upper/lower bounds on the optimal makespan
```

For defining this schema when **creating your own dataset**, and for the full list of schema fields and shorthand forms, see [Instance Metadata — Declaring a metadata schema](instance_metadata.md#level-7--declaring-a-metadata-schema) and [Dataset authoring — Enriched dataset](dataset_authoring.md#enriched-dataset-optional-dataset-metadata-and-a-field-schema). There you will also find that **all schema fields are optional**: you can omit `features` entirely or use minimal declarations per field; the docs explain the defaults and what you lose by not defining fields fully.

`DatasetInfo` is also a dict subclass, so `info["name"]` works alongside `info.name`.

### Dataset card (HuggingFace convention)

*Dataset cards* are standard README-style documents for a dataset: a short description, homepage, citations, and a table of instance metadata fields. They follow the [HuggingFace Hub dataset card](https://huggingface.co/docs/hub/datasets-cards) convention so that both humans and tooling can understand what the dataset contains without loading any instances.

Use them as the README for a published dataset, as appendix material in papers, or to compare datasets. Generation requires no download — it uses only the class-level `DatasetInfo`.

`card()` returns a single string: a **YAML frontmatter** block (for machine parsing) followed by a **Markdown** body (description, homepage, citations, instance features table, and a short usage example).

```python
card = JSPLibDataset.card()   # classmethod — no download needed
print(card)
```

Example output (abbreviated):

```
---
name: jsplib
homepage: https://github.com/tamy0612/JSPLIB
citation:
  - "J. Adams, E. Balas, D. Zawack. The shifting bottleneck procedure for job shop scheduling. Management Science, 1988."
---

# jsplib Dataset

A collection of Job Shop Scheduling benchmark instances.

**Homepage:** https://github.com/tamy0612/JSPLIB

## License
MIT

## Instance Features (Domain Metadata)
| Field    | Type | Nullable | Description                    |
|----------|------|----------|--------------------------------|
| `jobs`   | int  | Yes      | Number of jobs                 |
| `machines` | int  | Yes      | Number of machines             |
...
```

### Croissant JSON-LD (MLCommons)

[Croissant](https://mlcommons.org/working-groups/data/croissant/) is the MLCommons metadata standard for machine-learning datasets. A Croissant descriptor is a **JSON-LD** document that describes a dataset (name, description, homepage) and the schema of each instance (field names, types, descriptions). It is machine-readable and uses standard vocabularies (schema.org, Croissant `cr:` terms) so that crawlers, search engines, and ML tooling can discover and interpret the dataset without loading it.

Use Croissant when you want to **publish a dataset** in a way that Google Dataset Search and other ML infrastructure can index, or when you need a **portable schema** (e.g. for validation or codegen). Like dataset cards, generation uses only `DatasetInfo` — no download of the actual dataset needed.

`to_croissant()` returns a dict that you can serialize to JSON and publish next to your data (e.g. as `metadata.json`):

```python
import json
croissant = JSPLibDataset.dataset_metadata().to_croissant()
print(json.dumps(croissant, indent=2))
# Or save to file: json.dump(croissant, open("metadata.json", "w"), indent=2)
```

Example output (abbreviated):

```json
{
  "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/1.0"},
  "@type": "sc:Dataset",
  "name": "jsplib",
  "description": "A collection of Job Shop Scheduling benchmark instances.",
  "url": "https://github.com/tamy0612/JSPLIB",
  "license": "MIT",
  "cr:recordSet": [{
    "@type": "cr:RecordSet",
    "name": "instances",
    "cr:field": [
      {"@type": "cr:Field", "name": "id", "dataType": "sc:Text"},
      {"@type": "cr:Field", "name": "jobs", "dataType": "sc:Integer", "description": "Number of jobs"},
      {"@type": "cr:Field", "name": "machines", "dataType": "sc:Integer", "description": "Number of machines"},
      {"@type": "cr:Field", "name": "optimum", "dataType": "sc:Integer", "description": "Known optimal makespan, if available"},
      …
    ]
  }]
}
```

The `cr:recordSet` describes the shape of each instance (e.g. one row per file); `cr:field` lists the instance-level metadata fields and their schema.org types. The descriptor also includes standard CP model feature fields (e.g. `num_variables`, `num_constraints`) so that downstream tools know what to expect after loading.

---

## Creating a Custom Dataset

### Minimal dataset

Subclass `FileDataset` and implement four things:

```python
from cpmpy.tools.datasets import FileDataset


class MyDataset(FileDataset):

    # Required class attributes
    name        = "mydataset"
    description = "A short description of the dataset."
    homepage    = "https://example.com/mydataset"

    def __init__(self, root=".", transform=None, target_transform=None,
                 download=False, metadata_workers=1):
        import pathlib
        super().__init__(
            dataset_dir=pathlib.Path(root) / self.name,
            transform=transform, target_transform=target_transform,
            download=download, extension=".txt",
            metadata_workers=metadata_workers,
        )

    def parse(self, instance):
        """Optional parse-first hook for non-model datasets."""
        return self.read(instance)

    def category(self) -> dict:
        """Return category labels (e.g. year/track). Empty dict if none."""
        return {}

    def download(self):
        """Download instances to self.dataset_dir."""
        raise NotImplementedError
```

### Adding rich metadata

Declare optional class attributes for a fully documented dataset:

```python
from cpmpy.tools.datasets.metadata import FeaturesInfo, FieldInfo


class MyDataset(FileDataset):

    name        = "mydataset"
    description = "A short description of the dataset."
    homepage    = "https://example.com/mydataset"
    citation    = ["Author et al. Title. Journal, 2024."]

    # Declares the per-instance metadata fields this dataset provides
    features = FeaturesInfo({
        "num_jobs":    ("int",  "Number of jobs"),
        "num_machines": ("int", "Number of machines"),
        "optimum":     FieldInfo("int", "Known optimal value", nullable=True),
    })

    def collect_instance_metadata(self, file) -> dict:
        """Extract metadata from a single instance file."""
        # Return a dict whose keys match the fields declared in `features`
        return {
            "num_jobs":     ...,
            "num_machines": ...,
        }

    # ... rest of the class as before ...
```

`card()` and `to_croissant()` use these attributes automatically — no extra work
needed.

### Subclassing an existing dataset

If you want to extend a dataset with additional fields, subclass it and declare
only the new fields. The framework merges parent and child schemas automatically:

```python
class DifficultyJSPDataset(JSPLibDataset):
    """JSPLib extended with a computed difficulty score."""

    # Only the NEW field — {jobs, machines, optimum, …} are merged in automatically
    features = FeaturesInfo({
        "difficulty": FieldInfo("float", "Makespan / num_jobs ratio", nullable=True),
    })

    def collect_instance_metadata(self, file) -> dict:
        meta = super().collect_instance_metadata(file)   # get parent fields
        jobs = meta.get("jobs", 1)
        makespan = meta.get("optimum") or (meta.get("bounds") or {}).get("upper")
        if makespan and jobs:
            meta["difficulty"] = round(makespan / jobs, 3)
        return meta
```

The merged schema appears in `card()`, `to_croissant()`, and `validate()` without
any extra code:

```python
info = DifficultyJSPDataset.dataset_metadata()
print(list(info.features.fields))
# ['jobs', 'machines', 'optimum', 'bounds', 'instance_description', 'difficulty']
```

You can also merge `FeaturesInfo` schemas directly using `|`:

```python
# Explicit merge — same result as auto-merge, more verbose
class MyJSP(JSPLibDataset):
    features = JSPLibDataset.features | FeaturesInfo({"difficulty": "float"})
```

---

## Writing a Custom Transform

Transforms can be **any callable**: a function or a lambda is enough when you only need to change the data. When you need to **update metadata from the transformed result** (e.g. add file size, or new format fields after translation), use a class that implements `enrich_metadata`. This section starts with simple callables, then describes their limitations, then introduces the class-based form.

### Simple transforms: functions and lambdas

The dataset calls your transform with the current item (file path, or the output of the previous transform in a pipeline) and uses the return value as the new item. A plain function or lambda is sufficient when you don't need to change metadata based on that result.

```python
# Pass-through (no change)
dataset.transform = lambda x: x

# Upper-case the path (silly but valid)
dataset.transform = lambda path: path.upper() if isinstance(path, str) else path

# Load and return the raw file content
def load_raw(path):
    with open(path) as f:
        return f.read()
dataset.transform = load_raw

for content, info in dataset:
    print(len(content), info["name"])
```

These work with `Compose` as well: any callable in the list is invoked in order, and the output of one becomes the input of the next.

### Limitations of callable-only transforms

A plain function or lambda **cannot** update metadata from the transformed data. The dataset only calls `enrich_metadata(data, metadata)` when the transform object has that method. So you cannot:

- Add fields derived from the transform output (e.g. file size from the path, or `opb_num_variables` from the translated string).
- Strip old format metadata and attach new format fields when the transform changes format (e.g. WCNF → OPB).

For metadata-only updates that don't depend on the transformed data, use **`target_transform`** instead (it receives the current `InstanceInfo` and returns an updated one). For updates that *do* depend on the transform output — or when you want to hold state (e.g. a loader, an `open` callable) in a clear way — use a **class-based transform** with `__call__` and optionally `enrich_metadata`.

### Class-based transforms

All transforms follow the same protocol: a callable `__call__(self, data)` that transforms the data, and an optional `enrich_metadata(self, data, metadata)` method that updates the instance metadata based on the transformed data.

```python
class MyTransform:

    def __call__(self, file_path: str) -> Any:
        """
        Transform the data. Receives the file path (or the output of the
        previous transform in a Compose chain) and returns anything.
        """
        ...

    def enrich_metadata(self, data, metadata: InstanceInfo) -> InstanceInfo:
        """
        Update metadata based on the output of __call__.

        - data     : the value returned by __call__
        - metadata : the current InstanceInfo for this instance
        - returns  : updated InstanceInfo

        Called automatically by the dataset after __call__ returns.
        Omit this method if your transform does not affect metadata.
        """
        return metadata | {"my_field": compute(data)}
```

### Example: annotating instances with file size

```python
class AnnotateFileSize:

    def __call__(self, file_path):
        return file_path   # pass through unchanged

    def enrich_metadata(self, data, metadata):
        import os
        return metadata | {"file_size_bytes": os.path.getsize(data)}


dataset = JSPLibDataset(root="./data")
dataset.transform = AnnotateFileSize()

for file_path, info in dataset:
    print(info["file_size_bytes"])
```

### Example: format-changing transform

When `__call__` produces output in a different format, use `without_format()` in
`enrich_metadata` to drop the old format fields and add the new ones:

```python
from cpmpy.tools.datasets.transforms import Translate, extract_format_metadata

class TranslateToDIMACS:

    def __init__(self, loader, open):
        self._translate = Translate(loader, "dimacs", open=open)

    def __call__(self, file_path):
        self._last_output = self._translate(file_path)
        return self._last_output

    def enrich_metadata(self, data, metadata):
        dimacs_fields = extract_format_metadata(data, "dimacs")
        return metadata.without_format() | dimacs_fields
```

### Composing transforms

Chain multiple transforms with `Compose`. Each step's `enrich_metadata` is called
with the output that step produced, so each transform sees its own output:

```python
from cpmpy.tools.datasets.transforms import Compose, Load, Serialize

dataset.transform = Compose([
    Load(load_xcsp3, open=dataset.open),       # file_path → CPMpy model
    Serialize("opb"),                          # CPMpy model → OPB string
])

# Load.enrich_metadata receives the model and adds model_features
# Serialize has no enrich_metadata — no metadata changes at that step
```

---

## Examples

Runnable examples are in `examples/datasets/`:

| File | Covers |
|------|--------|
| `01_basic_usage.py` | Iterating, dict access, `InstanceInfo` properties |
| `02_dataset_card_and_croissant.py` | `DatasetInfo`, `card()`, Croissant export |
| `03_target_transforms.py` | `target_transform`, `to_croissant`, `to_gbd` |
| `04_custom_dataset.py` | Minimal, enriched, and subclassed dataset classes |
| `05_features_merge.py` | `FeaturesInfo \|`, auto-merge, multi-level inheritance |
| `06_benchmark_survey.py` | Iterating all datasets, collecting metadata statistics |
| `07_metadata_enrichment.py` | `\|`, `without_format()`, `enrich_metadata` |

---

## Further Reading

- [Datasets](datasets.md) — dataset quickstart and pipelines
- [Instance Metadata](instance_metadata.md) — full guide to `InstanceInfo`, enrichment, and interoperability
- [Transforms guide](transforms_guide.md) — authoring transforms and analytics pipelines
- [Dataset authoring](dataset_authoring.md) — implementing datasets, loaders, metadata schemas
- [Benchmarking workflows](benchmarking_workflows.md) — dataset-driven experiment patterns
- [Datasets API](api/tools/datasets.rst)
- [Benchmark runner](api/tools/benchmark_runner.rst)
