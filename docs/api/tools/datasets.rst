Datasets (:mod:`cpmpy.tools.datasets`)
=======================================

CPMpy provides a PyTorch-style dataset interface for loading and iterating over
benchmark instance collections. Each dataset handles downloading, file discovery,
metadata collection, and decompression automatically.

For worked, narrative guides (Markdown), see:

- :doc:`/datasets` (quickstart + common pipelines)
- :doc:`/instance_metadata` (the metadata system)
- :doc:`/transforms_guide` (transform pipelines, enrichment, analytics)
- :doc:`/dataset_authoring` (implementing new datasets/loaders/writers)
- :doc:`/benchmarking_workflows` (dataset-driven experiments)
- :doc:`/reading_and_writing` (IO loaders/writers + translation workflows)

Basic Usage
-----------

Create a dataset and iterate over ``(file_path, info)`` pairs:

.. code-block:: python

    from cpmpy.tools.datasets import JSPLibDataset

    dataset = JSPLibDataset(root="./data", download=True)

    for file_path, info in dataset:
        print(info["name"], info["jobs"], "×", info["machines"])

The second element ``info`` is an :class:`InstanceInfo
<cpmpy.tools.datasets.metadata.InstanceInfo>` — a dict subclass with additional
structured properties.

Available Datasets
------------------

.. list-table::
   :header-rows: 1

   * - **Class**
     - **Domain**
     - **Format**
   * - :class:`XCSP3Dataset <cpmpy.tools.datasets.xcsp3.XCSP3Dataset>`
     - CP / COP
     - XCSP3
   * - :class:`OPBDataset <cpmpy.tools.datasets.opb.OPBDataset>`
     - Pseudo-Boolean
     - OPB
   * - :class:`MaxSATEvalDataset <cpmpy.tools.datasets.mse.MaxSATEvalDataset>`
     - MaxSAT
     - WCNF
   * - :class:`JSPLibDataset <cpmpy.tools.datasets.jsplib.JSPLibDataset>`
     - Job Shop Scheduling
     - JSPLib
   * - :class:`PSPLibDataset <cpmpy.tools.datasets.psplib.PSPLibDataset>`
     - Project Scheduling
     - PSPLib
   * - :class:`NurseRosteringDataset <cpmpy.tools.datasets.nurserostering.NurseRosteringDataset>`
     - Nurse Rostering
     - NRP
   * - :class:`MIPLibDataset <cpmpy.tools.datasets.miplib.MIPLibDataset>`
     - Mixed Integer Programming
     - MPS
   * - :class:`SATDataset <cpmpy.tools.datasets.sat.SATDataset>`
     - SAT
     - DIMACS (CNF)

Loading into CPMpy Models
--------------------------

Use an IO loader as the dataset ``transform`` (or wrap it with ``Load``):

.. code-block:: python

    from cpmpy.tools.datasets import XCSP3Dataset
    from cpmpy.tools.datasets.transforms import Load
    from cpmpy.tools.io import load_xcsp3

    dataset = XCSP3Dataset(root="./data", year=2024, track="CSP", download=True)
    dataset.transform = Load(load_xcsp3, open=dataset.open)

    for model, info in dataset:
        model.solve()
        vars = info.model_objects["variables"]
        print({name: v.value() for name, v in vars.items()})

Reading and writing with compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Datasets use :meth:`~cpmpy.tools.datasets.core.FileDataset.open` for reading
(and thus decompression): e.g. ``dataset.open(path)`` returns a text stream
for ``.xml.lzma`` or ``.opb.xz``. Writers mirror this with an ``open``
parameter: pass a callable that accepts ``(path, mode="w")`` and returns a
text stream. Use any Python code you like (for example ``lzma.open`` or
``gzip.open``) to write compressed output:

.. code-block:: python

    import lzma
    from cpmpy.tools.io.dimacs import write_dimacs
    from cpmpy.tools.io import write_opb

    xz_text = lambda path, mode="w": lzma.open(path, "wt")
    write_opb(model, "out.opb.xz", open=xz_text)
    write_dimacs(model, "out.cnf.xz", open=xz_text)

Loaders (e.g. :func:`cpmpy.tools.io.dimacs.load_dimacs`, :func:`cpmpy.tools.io.opb.load_opb`)
also accept an ``open`` parameter for decompression on read.

Transforms
----------

Transforms are applied to the file path on each iteration. Set them on the
``transform`` attribute or pass them to the constructor.

.. list-table::
   :header-rows: 1

   * - **Class**
     - **Description**
   * - :class:`Open <cpmpy.tools.datasets.transforms.Open>`
     - Read raw file contents (handles decompression)
   * - :class:`Load <cpmpy.tools.datasets.transforms.Load>`
     - Load file into a CPMpy model; enriches metadata with model statistics and ``variables``
   * - :class:`Serialize <cpmpy.tools.datasets.transforms.Serialize>`
     - Serialize a CPMpy model to a format string
   * - :class:`Translate <cpmpy.tools.datasets.transforms.Translate>`
     - Load then serialize in one step (format translation)
   * - :class:`SaveToFile <cpmpy.tools.datasets.transforms.SaveToFile>`
     - Write output to disk; optionally writes ``.meta.json`` sidecars
   * - :class:`Lambda <cpmpy.tools.datasets.transforms.Lambda>`
     - Wrap any callable as a named transform
   * - :class:`Compose <cpmpy.tools.datasets.transforms.Compose>`
     - Chain multiple transforms sequentially

.. code-block:: python

    from cpmpy.tools.datasets.transforms import Compose, Translate, SaveToFile
    from cpmpy.tools.io import load_xcsp3

    dataset.transform = Compose([
        Translate(load_xcsp3, "opb", open=dataset.open),
        SaveToFile("./translated/", extension=".opb", write_metadata=True),
    ])

    for output_path, info in dataset:
        print("Saved:", output_path)

Instance Metadata (``InstanceInfo``)
-------------------------------------

``InstanceInfo`` is a ``dict`` subclass — all existing dict access is unchanged.
Structured properties partition the flat dict into four named groups:

.. list-table::
   :header-rows: 1

   * - **Property**
     - **Contents**
     - **Serializable**
   * - ``domain_metadata``
     - Problem-level fields: ``jobs``, ``machines``, ``horizon``, …
     - ✅
   * - ``format_metadata``
     - Format-specific fields: ``opb_*``, ``wcnf_*``, ``mps_*``, …
     - ✅
   * - ``model_features``
     - CP model statistics: variable counts, constraint counts, objective info
     - ✅
   * - ``model_objects``
     - Live CPMpy objects: ``variables`` map — in-memory only after ``Load``
     - ❌

.. code-block:: python

    file, info = dataset[0]

    # Dict access (unchanged)
    info["name"]
    info.get("jobs", 0)

    # Structured properties
    info.id                # "jsplib/abz5"
    info.domain_metadata   # {"jobs": 10, "machines": 10, …}
    info.model_features    # {"num_variables": …}  — after Load
    info.model_objects     # {"variables": {…}}  — after Load

Metadata Enrichment
--------------------

``InstanceInfo`` supports two common enrichment patterns. The most frequent
case is simply adding computed fields. Use the ``|`` operator to merge any
dict into the metadata — everything already in ``info`` is preserved, and
the result is still an ``InstanceInfo`` so all structured properties keep
working:

.. code-block:: python

    for file_path, info in dataset:
        enriched = info | {"density": info["jobs"] / info["machines"]}
        print(enriched.domain_metadata)   # includes the new "density" field

To have enrichment happen automatically on every iteration without touching
the loop, pass a ``target_transform``. It receives each ``InstanceInfo``
and its return value replaces the metadata for that item:

.. code-block:: python

    dataset = JSPLibDataset(
        root="./data",
        target_transform=lambda info: info | {
            "density":     info["jobs"] / info["machines"],
            "has_optimum": info.get("optimum") is not None,
        },
    )

    for file_path, info in dataset:
        print(info["density"])   # already computed, no extra code needed

The second pattern arises when a transform changes the file format — for
example, translating a WCNF instance to OPB. The old format-specific fields
(``wcnf_*``) are now stale and should be dropped, while new ones (``opb_*``)
should be added. ``without_format()`` strips all format-prefixed fields and
carries everything else forward; chain it with ``|`` to attach the new ones:

.. code-block:: python

    from cpmpy.tools.datasets.transforms import extract_format_metadata

    # "jobs" and other domain fields survive; wcnf_* are removed; opb_* are added
    new_info = info.without_format() | extract_format_metadata(opb_string, "opb")

For a full explanation with more examples and use cases, see
:doc:`/instance_metadata`.

.. _datasets_advanced_metadata:

Advanced Metadata System (Placeholder)
--------------------------------------

This section is intentionally reserved for an in-depth guide on the metadata
system used by ``InstanceInfo``, ``DatasetInfo``, ``FeaturesInfo``, and
``FieldInfo``.

Planned content includes:

- detailed metadata lifecycle (collection, sidecar storage, loading, enrichment)
- domain vs format-specific vs model-feature metadata boundaries
- schema design guidelines with ``FeaturesInfo`` and ``FieldInfo``
- dtype normalisation (canonical strings, Python types, schema.org types)
- JSON serialisation contracts (``to_dict``, ``to_jsonable``, ``to_json``)
- export mappings and constraints for Croissant and dataset cards
- recommendations for robust metadata validation and versioning

Until this section is expanded, use the ``Instance Metadata`` and
``Dataset-Level Metadata`` sections in this page, and the
``cpmpy.tools.datasets.metadata`` API reference below.

Dataset-Level Metadata
-----------------------

Every dataset class exposes a :class:`DatasetInfo
<cpmpy.tools.datasets.metadata.DatasetInfo>` with name, version, license, tags,
citation, and the instance field schema:

.. code-block:: python

    info = JSPLibDataset.dataset_metadata()   # classmethod — no download needed

    info.name        # "jsplib"
    info.version     # "1.0.0"
    info.license     # "MIT"
    info.features    # FeaturesInfo with field schema

    # HuggingFace-style dataset card
    print(JSPLibDataset.card())

    # MLCommons Croissant JSON-LD
    import json
    print(json.dumps(JSPLibDataset.dataset_metadata().to_croissant(), indent=2))

Creating a Custom Dataset
--------------------------

For a complete authoring guide (design patterns, metadata conventions, and
implementation checklist), see :doc:`/dataset_authoring`.

Subclass :class:`FileDataset <cpmpy.tools.datasets.core.FileDataset>` and
implement the required abstract methods. A **minimal** dataset needs only the
class-level name/description/homepage attributes and three methods:

.. code-block:: python

    from cpmpy.tools.datasets import FileDataset


    class MinimalDataset(FileDataset):

        name        = "minimal"
        description = "Minimal example dataset."
        homepage    = "https://example.com/minimal"

        def parse(self, instance):
            """Optional parse-first hook."""
            return self.read(instance)

        def category(self) -> dict:
            return {}   # or {"year": ..., "track": ...}

        def download(self):
            ...  # download files to self.dataset_dir

An **enriched** dataset adds optional metadata fields and a custom ``__init__``
to control the dataset directory and extension:

.. code-block:: python

    import pathlib
    from cpmpy.tools.datasets import FileDataset
    from cpmpy.tools.datasets.metadata import FeaturesInfo, FieldInfo


    class MyDataset(FileDataset):

        name        = "mydataset"
        description = "A short description."
        homepage    = "https://example.com/mydataset"
        citation    = ["Author et al. My Dataset. Journal, 2024."]

        version      = "1.0.0"
        license      = "CC BY 4.0"
        domain       = "constraint_programming"
        tags         = ["combinatorial", "satisfaction"]
        language     = "MyFormat"
        release_notes = {"1.0.0": "Initial release."}

        features = FeaturesInfo({
            "num_vars":    ("int", "Number of decision variables"),
            "optimum":     FieldInfo("int", "Known optimal value", nullable=True),
        })

        def __init__(self, root=".", transform=None, target_transform=None,
                     download=False):
            super().__init__(
                dataset_dir=pathlib.Path(root) / self.name,
                transform=transform, target_transform=target_transform,
                download=download, extension=".txt",
            )

        def parse(self, instance):
            """Optional parse-first hook."""
            return self.read(instance)

        def category(self) -> dict:
            return {}

        def download(self):
            ...  # download files to self.dataset_dir

        def collect_instance_metadata(self, file) -> dict:
            return {"num_vars": ...}

Field Type Normalisation (``FieldInfo.dtype``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FieldInfo`` accepts canonical dtype strings, schema.org dtype strings, or
Python types.
Internally, values are normalised to canonical strings so metadata schemas are
stable and JSON-serialisable. These canonical dtypes are also mapped to
schema.org ``dataType`` values for Croissant export.

Unknown dtype strings or unsupported Python types raise an exception.

.. list-table::
   :header-rows: 1

   * - Canonical dtype
     - Accepted schema.org dtype
     - Accepted Python type
     - Croissant/schema.org ``dataType``
   * - ``"int"``
     - ``"sc:Integer"``
     - ``int``
     - ``sc:Integer``
   * - ``"float"``
     - ``"sc:Float"``
     - ``float``
     - ``sc:Float``
   * - ``"str"``
     - ``"sc:Text"``
     - ``str``
     - ``sc:Text``
   * - ``"bool"``
     - ``"sc:Boolean"``
     - ``bool``
     - ``sc:Boolean``
   * - ``"dict"``
     - ``"sc:StructuredValue"``
     - ``dict``
     - ``sc:StructuredValue``
   * - ``"list"``
     - ``"sc:ItemList"``
     - ``list``
     - ``sc:ItemList``

.. code-block:: python

    features = FeaturesInfo({
        "jobs": int,                               # normalised to "int"
        "deadline": "sc:Integer",                 # schema.org string also accepted
        "machines": ("int", "Number of machines"), # canonical string form
        "optimum": FieldInfo(float, "Best value", nullable=True),
    })

To extend an existing dataset, subclass it and declare only the new fields —
the framework merges parent and child schemas automatically:

.. code-block:: python

    class DifficultyJSP(JSPLibDataset):
        features = FeaturesInfo({
            "difficulty": FieldInfo("float", "Makespan / num_jobs ratio", nullable=True),
        })

        def collect_instance_metadata(self, file) -> dict:
            meta = super().collect_instance_metadata(file)
            jobs = meta.get("jobs", 1)
            bound = meta.get("optimum") or meta.get("bounds", {}).get("upper")
            if bound and jobs:
                meta["difficulty"] = round(bound / jobs, 3)
            return meta

Writing a Custom Transform
---------------------------

Implement ``__call__`` for the data transformation and optionally
``enrich_metadata`` to update instance metadata based on the output:

.. code-block:: python

    class MyTransform:

        def __call__(self, file_path):
            """Transform the data. Return the new data value."""
            ...

        def enrich_metadata(self, data, metadata):
            """
            Update metadata based on __call__'s output.
            Called automatically by the dataset after __call__.
            Returns an updated InstanceInfo.
            """
            return metadata | {"my_field": compute(data)}

For format-changing transforms use ``without_format()`` to drop old format fields:

.. code-block:: python

        def enrich_metadata(self, data, metadata):
            return metadata.without_format() | extract_format_metadata(data, "opb")

.. _datasets_advanced_authoring:

Advanced Dataset Authoring (Placeholder)
----------------------------------------

This placeholder section has been superseded by the Markdown guides:

- :doc:`/dataset_authoring`
- :doc:`/transforms_guide`
- :doc:`/benchmarking_workflows`

API Reference
-------------

.. automodule:: cpmpy.tools.datasets
    :members:
    :undoc-members:

.. automodule:: cpmpy.tools.datasets.metadata
    :members:
    :undoc-members:

.. automodule:: cpmpy.tools.datasets.transforms
    :members:
    :undoc-members:

.. automodule:: cpmpy.tools.datasets.core
    :members:
    :undoc-members:
