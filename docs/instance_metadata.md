# Instance Metadata

When running experiments on benchmark sets, the raw instances are rarely enough.
You need to know *what* you are solving: how many variables, what domain sizes,
whether an optimal solution is known, what the structure looks like. You also
want to carry that information through transformations — if you translate an
instance from XCSP3 to OPB, the number of jobs in the original scheduling
problem should still be attached to the result. And when you finally save the
results of a batch run to a CSV or a database, you want a clean, predictable
record structure.

CPMpy's metadata system is built around a single class — `InstanceInfo` — that
addresses all of these concerns at once. This page explains the system from
first principles, starting with the simplest usage (it's just a dict) and
gradually introducing more powerful features.

---

## Level 1 — It's just a dict

`InstanceInfo` inherits directly from Python's `dict`.

```python
from cpmpy.tools.datasets import JSPLibDataset

dataset = JSPLibDataset(root="./data", download=True)
file_path, info = dataset[0]

# Standard dict access — always works, nothing to learn
info["name"]           # "abz5"
info["jobs"]           # 10
info.get("optimum")    # 1234  (or None if not recorded for this instance)
"machines" in info     # True
list(info.keys())      # ["dataset", "name", "category", "jobs", "machines", …]

for key, value in info.items():
    print(key, "=", value)
```

If you only need a handful of fields and don't care about structure, stop here.
The rest of this page describes what the metadata system adds on top.

---

## Level 2 — Structured properties

A benchmark instance metadata dict contains many different kinds of information
mixed together: system bookkeeping fields like `"name"` and `"path"`, problem
parameters like `"jobs"` and `"machines"`, format-specific header statistics
like `"opb_num_variables"`, and (after loading) CP model statistics like
`"num_constraints"`. All of these coexist in the flat dict.

`InstanceInfo` adds four read-only *properties* that partition that flat dict
into named groups. The data is not duplicated — the properties are computed
views over the same underlying dict:

```python
file_path, info = dataset[0]

info.domain_metadata  # {"jobs": 10, "machines": 10, "optimum": 1234, …}
info.format_metadata  # {"opb_num_variables": 42, "opb_num_constraints": 30}
info.model_features   # {"num_variables": 100, "num_constraints": 47, …}
info.model_objects    # {"variables": {"start_0_0": IntVar(…), …}}
```

The four partitions and what belongs in each:

| Property | Contents | Serializable |
|----------|----------|:---:|
| `domain_metadata` | Problem-level, format-independent fields: `jobs`, `machines`, `horizon`, `num_staff`, `optimum`, … | ✅ |
| `format_metadata` | Format-specific header fields, prefixed by format: `opb_*`, `wcnf_*`, `mps_*`, `xcsp_*`, `dimacs_*` | ✅ |
| `model_features` | CP model statistics computed from the parsed model: variable counts, constraint counts, domain sizes, objective info | ✅ |
| `model_objects` | Live Python objects added by `Load`: the `variables` name→variable map — **in-memory only** | ❌ |

The distinction matters in practice. Consider translating an OPB instance to
SAT (DIMACS format). The `domain_metadata` fields — problem-level parameters
such as the number of variables and constraints — describe the *problem* and
remain valid regardless of format. The `format_metadata` fields —
`opb_num_variables`, `opb_num_constraints` — describe the *file* and become
meaningless once the file is gone (replaced by `dimacs_*` for the new format).
The `model_features` describe the CPMpy model that was parsed from the file,
and may differ from the format statistics if, say, some variables were
simplified away during transformation.

### The stable instance ID

In addition to the four partitions, `InstanceInfo` provides a stable
slash-separated identifier built from the dataset name, any category labels
(year, track, variant), and the instance name:

```python
info.id   # "jsplib/abz5"
          # "xcsp3/2024/CSP/AverageAvoiding-20_c24"
          # "opb/miplib/aflow30b"
```

This `id` is designed to be unique and human-readable, making it
suitable as a primary key when storing results in a database, CSV, or
experiment log. It is included automatically in `to_croissant()` and `to_gbd()`
output. In the future, CPMpy may support **globally unique instance ID hashes**
as provided by the [Global Benchmark Database (GBD)](https://benchmark-database.de/); such hashes identify the same instance across collections and formats. For more on GBD’s instance identification and feature records, see the [GBD project](https://benchmark-database.de/) and the [SAT 2024 paper](https://doi.org/10.4230/LIPIcs.SAT.2024.18).

---

## Level 3 — Adding your own fields

The most common metadata operation is simply adding computed fields. You have
done your own analysis on an instance and want to attach the result alongside
the existing metadata so that everything travels together.

### The `|` operator

`InstanceInfo` overrides Python's dict merge operator `|` so that the result
is always a new `InstanceInfo`, not a plain dict. This means all structured
properties remain available after the merge:

```python
for file_path, info in dataset:
    enriched = info | {
        "density":      info["jobs"] / info["machines"],
        "has_optimum":  info.get("optimum") is not None,
    }

    # The new fields are just dict keys:
    print(enriched["density"])

    # But structured properties still work on the merged result:
    print(enriched.domain_metadata)   # includes "density" and "has_optimum"
    print(enriched.id)                # unchanged
```

The original `info` is not modified; `|` always creates a new object.

### Via `target_transform`

If you want the enrichment to happen automatically on every single item —
without writing a loop — pass a `target_transform` to the dataset constructor.
It is called with each `InstanceInfo` after the main `transform` has run, and
its return value replaces the info for that iteration:

```python
def add_difficulty(info):
    jobs     = info.get("jobs", 1)
    machines = info.get("machines", 1)
    return info | {
        "density":      jobs / machines,
        "has_optimum":  info.get("optimum") is not None,
    }

dataset = JSPLibDataset(root="./data", target_transform=add_difficulty)

# Now every item in the loop already has the extra fields:
for file_path, info in dataset:
    print(info["density"])
    print(info["has_optimum"])
```

A `lambda` works equally well for simple one-liners:

```python
dataset = JSPLibDataset(
    root="./data",
    target_transform=lambda info: info | {"density": info["jobs"] / info["machines"]},
)
```

`target_transform` is the right place for lightweight, stateless computations
that depend only on what is already in the metadata — computing derived ratios,
renaming fields, filtering, or converting types. For computations that depend
on the actual file content or a loaded model, use a full `transform` or
`enrich_metadata` (see Level 6).

---

## Level 4 — Handling format changes

Suppose you have a dataset of OPB instances and you want to translate all of
them to SAT (DIMACS format). After the translation, the `format_metadata` fields that describe
the OPB file (`opb_num_variables`, `opb_num_constraints`, …) no longer
describe the file you are working with. Leaving them in the metadata is
misleading. At the same time, the `domain_metadata` fields — the problem-level
parameters that were true of the original instance — are still valid and should
be kept.

`InstanceInfo.without_format()` solves this cleanly: it returns a copy of the
metadata with all format-specific fields stripped, while preserving everything
else:

```python
from cpmpy.tools.datasets.transforms import Translate, extract_format_metadata

dataset = OPBDataset(root="./data", download=True)
dataset.transform = Translate(dataset.load, "dimacs", open=dataset.open)

for dimacs_string, info in dataset:
    # At this point info still has the old opb_* fields from the original file.
    # Strip them and add the new dimacs_* fields extracted from the translated string:
    new_info = info.without_format() | extract_format_metadata(dimacs_string, "dimacs")

    # Domain fields are carried forward untouched:
    print(new_info["name"])   # ✅ still there

    # Old format fields are gone:
    assert "opb_num_variables" not in new_info

    # New format fields are present:
    print(new_info["dimacs_num_variables"])  # ✅ from the translated DIMACS string
```

`extract_format_metadata` parses the header of the output file to extract
format-specific statistics like variable and constraint counts. It currently
supports `"opb"`, `"dimacs"`, `"mps"`, and `"lp"`. Other formats (e.g. XCSP3,
WCNF) are not supported there because they do not have a simple line-based
header that can be parsed from a raw string — XCSP3 is XML, and WCNF shares
the DIMACS `p` line but is usually handled by the loader.

**Alternatives for formats without header-based extraction:**

- **`collect_instance_metadata(file)`** — In your dataset class, open the file,
  parse as much as you need (e.g. the first few lines or the XML root), and
  return a dict with format-prefixed keys (e.g. `xcsp_format`, `instance_type`).
  The framework stores these in `format_metadata`. See e.g. `XCSP3Dataset.collect_instance_metadata`
  in the codebase, which reads the XCSP3 XML header to set `xcsp_format` and
  `instance_type`.

- **`Load` + `model_features`** — After a `Load` transform, the framework
  fills `model_features` from the parsed CPMpy model (`num_variables`,
  `num_constraints`, etc.). That gives you portable instance statistics
  regardless of file format, without implementing format-specific header parsing.

If you only want to strip the old format fields without adding new ones — for
example, because you are translating to a format whose header has no useful
statistics — just call `without_format()` on its own:

```python
stripped = info.without_format()
assert not stripped.format_metadata     # empty
assert stripped["jobs"] == info["jobs"] # domain fields intact
```

The chain pattern — `without_format() | {...}` — is intentionally modelled
after Python's own dict merge so that it feels natural and composes well with
any additional fields you want to attach at the same time.

---

## Level 5 — Model objects and printing solution values

When you set the dataset's transform to the dataset's loader (e.g.
`dataset.transform = dataset.load`), each instance is loaded into a CPMpy
model and that model is the iteration value. But where do the *variable names*
go? The model holds CPMpy variable objects, but the connection between a
variable's string name and its object is not always easy to recover after the
fact.

The framework fills `model_objects["variables"]` with a `{name: CPMpy_variable}`
mapping whenever the transform returns a CPMpy model. After solving, you can
read every variable's value by name, without holding any separate reference to
the variable objects:

```python
from cpmpy.tools.datasets import JSPLibDataset

dataset = JSPLibDataset(root="./data", download=True)
dataset.transform = dataset.load

for model, info in dataset:
    print(f"Solving {info['name']} ({info['jobs']}×{info['machines']})…")
    if model.solve():
        print(f"  Optimal makespan: {model.objective_value()}")
        dvars = info.model_objects["variables"]
        # Print start times for all tasks
        for name, var in dvars.items():
            if name.startswith("start_"):
                print(f"  {name} = {var.value()}")
    else:
        print("  No solution found")
```

---

## Level 6 — Custom transforms with `enrich_metadata`

So far, metadata enrichment has been done either after the fact (with `|` in
the loop) or via `target_transform` (which sees the final metadata but not the
transformed data). Sometimes you need to enrich metadata based on the *output*
of a transform — for example, computing file size after compression, or
extracting format statistics from a translated string.

Any transform class can implement an optional `enrich_metadata(self, data,
metadata)` method. The dataset calls it automatically after `__call__` returns,
passing both the output of `__call__` and the current `InstanceInfo`. Whatever
`enrich_metadata` returns becomes the new metadata for that item.

### Example: annotating with file size

```python
import os

class AnnotateFileSize:
    """Passes the file path through unchanged, but records the file size."""

    def __call__(self, file_path):
        return file_path   # data is unchanged

    def enrich_metadata(self, data, metadata):
        # data is the file path (the return value of __call__)
        return metadata | {"file_size_bytes": os.path.getsize(data)}


dataset = JSPLibDataset(root="./data")
dataset.transform = AnnotateFileSize()

for file_path, info in dataset:
    print(f"{info['name']}: {info['file_size_bytes']} bytes")
```

This is useful in benchmark studies where instance size is a predictor of
solver difficulty, or when you want to log instance sizes alongside solve times.

### Example: format-changing transform

When `__call__` produces output in a different format, `enrich_metadata` is
the right place to call `without_format()` and attach new format statistics.
This keeps the format bookkeeping inside the transform, so the calling code
stays clean:

```python
from cpmpy.tools.datasets.transforms import Translate, extract_format_metadata

class TranslateToOPB:
    """Translates any instance to OPB, updating metadata to match."""

    def __init__(self, loader, open):
        self._translate = Translate(loader, "opb", open=open)

    def __call__(self, file_path):
        # Perform the actual translation; return the OPB string
        return self._translate(file_path)

    def enrich_metadata(self, data, metadata):
        # data     = OPB string (the return value of __call__)
        # metadata = InstanceInfo as it stood before this step
        # Strip old format fields; add new ones extracted from the OPB header
        return metadata.without_format() | extract_format_metadata(data, "opb")


dataset = JSPLibDataset(root="./data")
dataset.transform = TranslateToOPB(dataset.load, open=dataset.open)

for opb_string, info in dataset:
    print(info["jobs"])               # domain field — carried forward automatically
    print(info["opb_num_variables"])  # new format field — set by enrich_metadata
```

### Composing multiple transforms

`Compose` chains transforms into a pipeline. Each step receives the output of
the previous one, and each step's `enrich_metadata` is called with the output
*that specific step* produced — so the metadata is built up incrementally:

```python
from cpmpy.tools.datasets.transforms import Compose, Load, Serialize, SaveToFile

dataset.transform = Compose([
    Load(dataset.load, open=dataset.open),
    # ↑ file path → CPMpy model; enrich_metadata adds model_features and
    #   variables to the metadata

    Serialize("opb"),
    # ↑ CPMpy model → OPB string; no enrich_metadata, so metadata unchanged

    SaveToFile("./out/", extension=".opb", write_metadata=True),
    # ↑ OPB string → saved file path; writes .meta.json sidecar;
    #   enrich_metadata adds "output_path" to the metadata
])

for output_path, info in dataset:
    # At this point info contains everything: domain fields, model features,
    # and the output path — all accumulated across the three steps.
    print(f"Saved {info['name']} to {output_path}")
    print(f"  Variables: {info['num_variables']}, Constraints: {info['num_constraints']}")
```

The `write_metadata=True` flag in `SaveToFile` causes a `.meta.json` sidecar
file to be written alongside each output file. The sidecar contains everything
from `domain_metadata`, `format_metadata`, and `model_features` — enough to
reconstruct the full context of the instance without re-running the pipeline.

---

## Level 7 — Declaring a metadata schema

When you publish a dataset, you want to document what metadata fields it
provides, what their types are, and which instances might not have a value for
a given field. CPMpy provides two classes for this: `FieldInfo` describes a
single field; `FeaturesInfo` collects all fields for a dataset.

### Declaring fields

```python
from cpmpy.tools.datasets.metadata import FeaturesInfo, FieldInfo

class MyDataset(FileDataset):
    ...
    features = FeaturesInfo({
        # Minimal shorthand: just (dtype, description)
        "num_jobs":     ("int", "Number of jobs in the instance"),
        "num_machines": ("int", "Number of machines"),

        # Full FieldInfo when you need nullable or example:
        "optimum": FieldInfo(
            dtype       = "int",
            description = "Known optimal makespan, if available",
            nullable    = True,    # some instances may not have a known optimum
            example     = 1234,
        ),
        "bounds": FieldInfo(
            dtype       = "dict",
            description = "Lower and upper bounds on the makespan",
            nullable    = True,
            example     = {"lower": 800, "upper": 1500},
        ),
    })

    def collect_instance_metadata(self, file) -> dict:
        """Called once per instance after download."""
        return {
            "num_jobs":     ...,
            "num_machines": ...,
            # "optimum" and "bounds" omitted here if not known — that's fine
        }
```

The supported dtypes are `"int"`, `"float"`, `"str"`, `"bool"`, `"dict"`,
and `"list"`. Setting `nullable=True` (the default) means the field may be
absent or `None` for some instances — useful for fields like `"optimum"` that
are only known for a subset.

`FeaturesInfo` also accepts several shorthand forms so that the common cases
are not verbose:

```python
FeaturesInfo({"jobs": "int"})                 # dtype only
FeaturesInfo({"jobs": ("int", "Job count")})  # dtype + description
FeaturesInfo({"jobs": FieldInfo("int", "Job count", nullable=False)})  # full
```

### Schema fields are optional

Providing a full schema is **optional**. You can:

- **Omit `features` entirely** — the dataset still works. Dataset cards and Croissant export still run but **omit the domain-field schema section** (no table of instance fields, no `cr:field` entries for domain fields in Croissant).
- **Use minimal declarations per field** — the framework coerces shorthand to `FieldInfo` with defaults for anything you leave out:

  | You provide | Defaults applied |
  |-------------|------------------|
  | `"jobs": "int"` (dtype only) | `description=""`, `nullable=True`, `example=None` |
  | `"jobs": ("int", "Number of jobs")` | `nullable=True`, `example=None` |
  | Full `FieldInfo(dtype, description, nullable=..., example=...)` | No defaults; you control each attribute |

If you omit or simplify the schema you lose:

- **No or partial field list** — cards and Croissant won't document your instance fields (or will show them with empty descriptions and "nullable: Yes" for everything).
- **No nullability signal** — consumers and tooling can't tell which fields are guaranteed to be present vs optional.
- **No example values** — documentation and generated cards won't show example values for fields.
- **Weaker typing for tooling** — anything that relies on `info.features` (e.g. validation, codegen, or exports) will have less precise type and description information.

For a minimal dataset that only defines `name`, `description`, and `homepage`, skipping `features` is fine. For anything you intend to publish or integrate with cards/Croissant, defining at least `(dtype, description)` per field is recommended.

### Schema inheritance

When you subclass an existing dataset to add new fields, declare only the
*new* fields in `features`. The framework merges parent and child schemas
automatically via `__init_subclass__`, so card generation and Croissant export
always see the full combined schema:

```python
class DifficultyJSPDataset(JSPLibDataset):
    """JSPLib extended with a difficulty score computed from the known optimum."""

    features = FeaturesInfo({
        "difficulty": FieldInfo(
            "float",
            "Estimated difficulty: known optimal makespan divided by the number of jobs",
            nullable=True,
        ),
    })

    def collect_instance_metadata(self, file) -> dict:
        meta  = super().collect_instance_metadata(file)   # get all parent fields
        jobs  = meta.get("jobs", 1)
        bound = meta.get("optimum") or (meta.get("bounds") or {}).get("upper")
        if bound and jobs:
            meta["difficulty"] = round(bound / jobs, 3)
        return meta
```

After subclassing, the merged schema includes every parent field plus the new
one — without any extra code:

```python
info = DifficultyJSPDataset.dataset_metadata()
list(info.features.fields)
# ['jobs', 'machines', 'optimum', 'bounds', 'instance_description', 'difficulty']
```

You can also merge `FeaturesInfo` objects explicitly with `|`, which gives
identical results and is useful when you need to compose schemas from multiple
sources:

```python
extra = FeaturesInfo({
    "difficulty":   FieldInfo("float", "Hardness proxy"),
    "cluster_id":   FieldInfo("int",   "Cluster assignment from k-means study"),
})
class MyJSP(JSPLibDataset):
    features = JSPLibDataset.features | extra
```

---

## Level 8 — Dataset-level metadata and interoperability

### DatasetInfo

Every dataset class exposes a `dataset_metadata()` classmethod that returns a
`DatasetInfo` object. Like `InstanceInfo`, `DatasetInfo` is a dict subclass
with structured properties on top. It is available without downloading anything:

```python
info = JSPLibDataset.dataset_metadata()

# Structured properties:
info.name        # "jsplib"
info.homepage    # "https://github.com/tamy0612/JSPLIB"
info.citation    # ["J. Adams, E. Balas, D. Zawack. …"]
info.features    # FeaturesInfo with the per-instance field schema

# And as a dict:
info["name"]     # "jsplib"  (backward-compatible)

# Inspect the field schema:
for name, fi in info.features.fields.items():
    nullable = " (optional)" if fi.nullable else ""
    print(f"  {name}: {fi.dtype}{nullable} — {fi.description}")
```

This is useful for tooling: you can enumerate all registered datasets, print
their metadata, and compare schemas without loading a single instance.

### Dataset cards

`card()` generates a human-readable summary in the HuggingFace Hub convention:
a YAML frontmatter block (for machine parsing) followed by a Markdown body
(for human reading). It includes the citations, the
full `features` schema, the standard CP model feature fields, etc:

```python
print(JSPLibDataset.card())
```

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
| …        | …    | …        | …                              |
…
```

Cards are useful for quickly documenting datasets in publications or READMEs.

### Croissant JSON-LD (MLCommons)

[Croissant](https://mlcommons.org/working-groups/data/croissant/) is the
MLCommons metadata standard for ML datasets, expressed as JSON-LD. CPMpy can
generate a compliant Croissant descriptor for any dataset, including the
per-instance field schema as a `cr:RecordSet`:

```python
import json

croissant = JSPLibDataset.dataset_metadata().to_croissant()
print(json.dumps(croissant, indent=2))
```

```json
{
  "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/1.0"},
  "@type": "sc:Dataset",
  "name": "jsplib",
  "description": "…",
  "url": "https://github.com/tamy0612/JSPLIB",
  "cr:recordSet": [{
    "@type": "cr:RecordSet",
    "name": "instances",
    "cr:field": [
      {"@type": "cr:Field", "name": "id",   "dataType": "sc:Text"},
      {"@type": "cr:Field", "name": "jobs", "dataType": "sc:Integer", "description": "Number of jobs"},
      …
    ]
  }]
}
```

Croissant descriptors are recognized by Google Dataset Search and other ML
infrastructure tooling.

### Producing ML-ready records from instances

Two adapter methods on `InstanceInfo` convert a single instance's metadata to
a flat, standard-format record:

```python
file_path, info = dataset[0]

# Croissant example record — id + domain fields + model features, flat dict
record = info.to_croissant()
# {"id": "jsplib/abz5", "jobs": 10, "machines": 10, "optimum": 1234,
#  "num_variables": 100, "num_constraints": 47, …}

# GBD (Global Benchmark Database) feature record
record = info.to_gbd()
# {"id": "jsplib/abz5", "filename": "abz5", "dataset": "jsplib",
#  "jobs": 10, "machines": 10, …}
```

Both adapters exclude `format_metadata` (format-specific, not portable) and
`model_objects` (not serializable). They include `domain_metadata` and, if
available, `model_features`.

You can use them as `target_transform` to have all instances automatically
converted on every iteration — useful when you are feeding the output directly
into a DataFrame or a database insert:

```python
from cpmpy.tools.datasets.metadata import to_croissant

dataset = JSPLibDataset(
    root="./data",
    transform=dataset.load,  # populate model_features
    target_transform=to_croissant,
)

import pandas as pd
records = [record for _, record in dataset]
df = pd.DataFrame(records)
# Columns: id, jobs, machines, optimum, num_variables, num_constraints, …
print(df.describe())
```

---

## Quick reference

| What you want | How |
|---------------|-----|
| Read a field | `info["jobs"]` or `info.get("jobs", default)` |
| Iterate all fields | `for k, v in info.items()` |
| Stable instance ID | `info.id` |
| Problem-level fields only | `info.domain_metadata` |
| Format-specific fields | `info.format_metadata` |
| CP model statistics | `info.model_features` — populated after `Load` |
| Variable name → CPMpy var | `info.model_objects["variables"]` — after `Load` |
| Add a field in the loop | `enriched = info \| {"my_field": value}` |
| Add fields on every item automatically | `target_transform=lambda info: info \| {...}` |
| Strip stale format fields | `info.without_format()` |
| Strip old + add new format fields | `info.without_format() \| extract_format_metadata(data, "opb")` |
| Enrich from transform output | implement `enrich_metadata(self, data, metadata)` |
| Declare field schema | `FeaturesInfo({"field": ("dtype", "description")})` |
| Extend an existing schema | subclass and declare only new fields in `features` |
| Merge schemas explicitly | `FeaturesInfo_a \| FeaturesInfo_b` |
| Dataset-level metadata | `MyDataset.dataset_metadata()` |
| Dataset card (Markdown) | `MyDataset.card()` |
| Croissant descriptor (JSON-LD) | `MyDataset.dataset_metadata().to_croissant()` |
| ML-ready records per instance | `target_transform=to_croissant` or `to_gbd` |
