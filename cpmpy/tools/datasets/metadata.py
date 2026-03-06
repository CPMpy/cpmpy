"""
Structured Metadata Classes for CPMpy Datasets

When iterating over a dataset, 2-tuples (instance, metadata) are returned.
The metadata is a subclass of the standard python dictionary. It has additional
methods that aid in managing the metadata and help convert it to different formats, 
like Croissant, GBD, Dataset Cards, etc.

Provides:
- :class:`FieldInfo`     — schema for one domain metadata field
- :class:`FeaturesInfo`  — schema for all domain metadata fields of a dataset
- :class:`InstanceInfo`  — dict-compatible per-instance metadata with structured access
- :class:`DatasetInfo`   — dict-compatible dataset-level metadata with card/Croissant export
- :func:`to_croissant_example` — adapter for use as ``target_transform``
- :func:`to_gbd_features`      — adapter for use as ``target_transform``

Design notes
------------
``InstanceInfo`` and ``DatasetInfo`` both inherit from ``dict``, so all existing
``meta['year']``, ``meta.get('jobs')``, and ``dataset_metadata()['name']`` calls
continue to work unchanged.  Structured access (``info.domain_metadata``,
``info.model_features``, ``DatasetInfo.card()``, etc.) is purely additive.

Inspired by:
- HuggingFace ``datasets.DatasetInfo`` and ``Features``/``Value``
- TensorFlow Datasets ``DatasetInfo``, ``FeatureConnector``, and ``BuilderConfig``
- MLCommons Croissant 1.0 (JSON-LD metadata standard)
- Global Benchmark Database (GBD) feature records
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Constants — keys that partition the flat instance metadata dict
# ---------------------------------------------------------------------------

# System-level keys added by instance_metadata() — not domain metadata
_SYSTEM_KEYS: frozenset = frozenset({"id", "dataset", "categories", "name", "path"})

# Fields produced by extract_model_features() (requires full CPMpy model parse)
_MODEL_FEATURE_FIELDS: frozenset = frozenset({
    "num_variables", "num_bool_variables", "num_int_variables",
    "num_constraints", "constraint_types", "has_objective",
    "objective_type", "objective", "objective_is_min",
    "domain_size_min", "domain_size_max", "domain_size_mean",
})

# Live Python objects added by Load — not JSON-serialisable, excluded from exports
_MODEL_OBJECT_KEYS: frozenset = frozenset({
    "variables",
})

# Prefixes for format-specific metadata (not portable across format translations)
_FORMAT_SPECIFIC_PREFIXES: tuple = ("opb_", "wcnf_", "mps_", "xcsp_", "dimacs_")


# ---------------------------------------------------------------------------
# FieldInfo
# ---------------------------------------------------------------------------

@dataclass
class FieldInfo:
    """
    Schema declaration for a single domain metadata field.

    Inspired by HuggingFace ``Value`` and TFDS ``FeatureConnector``, but
    intentionally simpler — no serialisation semantics needed for CO benchmarks.

    Arguments:

        dtype (str or type): Canonical dtype string, schema.org dtype string,
        or Python type.
        Accepted canonical strings are ``"int"``, ``"float"``, ``"str"``,
        ``"bool"``, ``"dict"``, and ``"list"``. Accepted schema.org strings
        are ``"sc:Integer"``, ``"sc:Float"``, ``"sc:Text"``,
        ``"sc:Boolean"``, ``"sc:StructuredValue"``, and ``"sc:ItemList"``.
        Accepted Python types are ``int``, ``float``, ``str``, ``bool``,
        ``dict``, and ``list``.
        Values are normalised to the canonical string representation at
        construction time.
        description (str): Human-readable description of the field.
        nullable (bool): Whether the field may be absent / ``None`` for some instances.
        example (Any): Optional example value (used in documentation / cards).
    """

    dtype: Any
    description: str = ""
    nullable: bool = True
    example: Any = None

    # Maps internal dtype strings → schema.org types (for Croissant export)
    _DTYPE_TO_SCHEMA_ORG: Dict[str, str] = None  # populated below as class var

    def __post_init__(self):
        self.dtype = self.normalize_dtype(self.dtype)

    def schema_org_type(self) -> str:
        """Return the schema.org dataType string for use in Croissant fields."""
        return _DTYPE_TO_SCHEMA_ORG.get(self.dtype, "sc:Text")

    @classmethod
    def normalize_dtype(cls, dtype: Any) -> str:
        """
        Normalise a dtype specification to a canonical dtype string.

        Accepts canonical string dtypes, schema.org dtype strings, and selected
        builtin Python types.
        Raises when a dtype cannot be normalised.
        """
        if isinstance(dtype, str):
            if dtype in _DTYPE_TO_SCHEMA_ORG:
                return dtype
            mapped_schema_dtype = _SCHEMA_ORG_TO_DTYPE.get(dtype)
            if mapped_schema_dtype is not None:
                return mapped_schema_dtype
            known = ", ".join(sorted(_DTYPE_TO_SCHEMA_ORG.keys()))
            known_schema = ", ".join(sorted(_SCHEMA_ORG_TO_DTYPE.keys()))
            raise ValueError(
                f"Unknown dtype string {dtype!r}. "
                f"Use a canonical dtype ({known}) or schema.org dtype ({known_schema})."
            )

        if isinstance(dtype, type):
            mapped = _PY_TYPE_TO_DTYPE.get(dtype)
            if mapped is not None:
                return mapped
            known_types = ", ".join(t.__name__ for t in _PY_TYPE_TO_DTYPE)
            raise TypeError(
                f"Cannot normalise Python type {dtype!r} to a dataset dtype. "
                f"Known Python types: {known_types}."
            )

        raise TypeError(
            "dtype must be a canonical dtype string, schema.org dtype string, "
            f"or Python type, got {type(dtype).__name__}."
        )

    @classmethod
    def coerce(cls, value: Any) -> "FieldInfo":
        """
        Normalise shorthand input into a :class:`FieldInfo`.

        Accepted forms:

        - ``FieldInfo(...)``                        — returned as-is
        - ``"int"``, ``"sc:Integer"``, or ``int``   — treated as ``FieldInfo(dtype=...)``
        - ``("int", "desc")``                       — ``FieldInfo(dtype="int", description="desc")``
        - ``("sc:Text", "desc")``                   — ``FieldInfo(dtype="sc:Text", description="desc")``
        - ``(int, "desc")``                         — ``FieldInfo(dtype=int, description="desc")``
        - ``("int", "desc", False)``                — adds ``nullable=False``
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, type)):
            return cls(dtype=value)
        if isinstance(value, tuple):
            return cls(*value)
        raise TypeError(
            f"Cannot coerce {value!r} to FieldInfo. "
            "Use a FieldInfo, a dtype string or Python type, "
            "or a (dtype, description[, nullable]) tuple."
        )

    def to_dict(self) -> dict:
        """Serialisable plain dict (for JSON sidecar storage)."""
        d = {"dtype": self.dtype, "description": self.description, "nullable": self.nullable}
        if self.example is not None:
            d["example"] = self.example
        return d


# Class-level constant (defined after the class to avoid dataclass conflicts)
_DTYPE_TO_SCHEMA_ORG: Dict[str, str] = {
    "int":   "sc:Integer",
    "float": "sc:Float",
    "str":   "sc:Text",
    "bool":  "sc:Boolean",
    "dict":  "sc:StructuredValue",
    "list":  "sc:ItemList",
}

_PY_TYPE_TO_DTYPE: Dict[type, str] = {
    int: "int",
    float: "float",
    str: "str",
    bool: "bool",
    dict: "dict",
    list: "list",
}

_SCHEMA_ORG_TO_DTYPE: Dict[str, str] = {
    schema_type: dtype for dtype, schema_type in _DTYPE_TO_SCHEMA_ORG.items()
}


# ---------------------------------------------------------------------------
# FeaturesInfo
# ---------------------------------------------------------------------------

class FeaturesInfo:
    """
    Schema for all domain metadata fields of a dataset.

    Analogous to HuggingFace ``Features`` or TFDS ``FeatureConnector`` trees,
    but without serialisation encoding — purely declarative.

    The constructor accepts a plain ``dict`` whose values are anything accepted
    by :meth:`FieldInfo.coerce`:

    .. code-block:: python

        # Minimal — just type strings
        FeaturesInfo({"jobs": "int", "machines": "int"})

        # With descriptions
        FeaturesInfo({"jobs": ("int", "Number of jobs")})

        # Full control where needed
        FeaturesInfo({
            "jobs":    ("int", "Number of jobs"),
            "optimum": FieldInfo("int", "Known optimal makespan", nullable=True),
        })
    """

    def __init__(self, fields: Dict[str, Any]):
        self.fields: Dict[str, FieldInfo] = {
            k: FieldInfo.coerce(v) for k, v in fields.items()
        }

    def __repr__(self) -> str:
        return f"FeaturesInfo({self.fields!r})"

    def __or__(self, other: "FeaturesInfo") -> "FeaturesInfo":
        """
        Merge two :class:`FeaturesInfo` schemas, with ``other`` taking
        precedence for any duplicate field names.

        Follows the same convention as Python's ``dict | dict`` (Python 3.9+).

        .. code-block:: python

            # Explicit merge — useful when you want full control:
            class MyJSPDataset(JSPLibDataset):
                features = JSPLibDataset.features | FeaturesInfo({"difficulty": "float"})
        """
        merged = FeaturesInfo.__new__(FeaturesInfo)
        merged.fields = {**self.fields, **other.fields}
        return merged

    def validate(self, domain_metadata: dict) -> List[str]:
        """
        Validate a domain_metadata dict against this schema.

        Returns a list of error strings (empty list = valid).
        """
        errors = []
        for name, fi in self.fields.items():
            if not fi.nullable and name not in domain_metadata:
                errors.append(f"Required field '{name}' missing from domain_metadata")
        return errors

    def to_croissant_fields(self) -> List[dict]:
        """
        Generate a list of Croissant ``cr:Field`` dicts for use in a
        ``cr:RecordSet``.
        """
        result = []
        for name, fi in self.fields.items():
            cr_field: Dict[str, Any] = {
                "@type": "cr:Field",
                "name": name,
                "dataType": _DTYPE_TO_SCHEMA_ORG.get(fi.dtype, "sc:Text"),
            }
            if fi.description:
                cr_field["description"] = fi.description
            result.append(cr_field)
        return result

    def to_dict(self) -> dict:
        """Serialisable plain dict (for JSON sidecar storage)."""
        return {name: fi.to_dict() for name, fi in self.fields.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeaturesInfo":
        """Reconstruct from the serialised plain dict produced by :meth:`to_dict`."""
        return cls({
            name: FieldInfo(
                dtype=v.get("dtype", "str"),
                description=v.get("description", ""),
                nullable=v.get("nullable", True),
                example=v.get("example"),
            )
            for name, v in d.items()
        })


# ---------------------------------------------------------------------------
# InstanceInfo
# ---------------------------------------------------------------------------

class InstanceInfo(dict):
    """
    Per-instance metadata dict with structured access.

    Inherits from ``dict`` and supports normal dictionary access patterns
    such as ``meta['year']``, ``meta.get('jobs')``, and
    ``for k, v in meta.items()``.

    Structured access is additive:

    .. code-block:: python

        file, info = dataset[0]

        # Dict access:
        info['name']
        info.get('jobs', 0)
        info['categories']['year']

        # New structured properties:
        info.id                  # "jsplib/abz5"
        info.category            # {"year": 2024, "track": "CSP", ...}
        info.domain_metadata     # {"jobs": 10, "machines": 5, ...}
        info.model_features      # {"num_variables": 100, ...}
        info.format_metadata     # {"opb_num_variables": 12, ...}

        # Standards converters:
        info.to_croissant()
        info.to_gbd()
    """

    @property
    def id(self) -> str:
        """
        Stable instance identifier.

        Uses explicit ``id`` when present (recommended for dataset-defined
        identifiers). Otherwise falls back to:
        ``"dataset/cat_val1/cat_val2/.../instance_name"``.

        For file-based datasets, ``id`` is typically set to the instance
        reference returned as the first element of the dataset ``(x, y)``
        tuple.

        Example: ``"xcsp3/2024/CSP/AverageAvoiding-20_c24"``
        """
        explicit = self.get("id")
        if explicit:
            return str(explicit)

        parts = [str(self.get("dataset", ""))]
        cat = self.get("categories", {})
        if isinstance(cat, dict):
            parts += [str(v) for v in cat.values()]
        parts.append(str(self.get("name", "")))
        return "/".join(p for p in parts if p)

    @property
    def name(self) -> str:
        """Human-readable instance name."""
        return self.get("name", "")

    @property
    def dataset(self) -> str:
        """Parent dataset name."""
        return self.get("dataset", "")

    @property
    def categories(self) -> dict:
        """Category dict (year, track, variant, family, …)."""
        return self.get("categories", {})

    @property
    def domain_metadata(self) -> dict:
        """
        Domain-specific metadata fields.

        These are format-independent, problem-level fields such as
        ``jobs``, ``machines``, ``optimum``, ``horizon``, ``num_staff``, etc.

        Excludes system keys, CP model statistics, live model objects, and
        format-specific fields.
        """
        return {
            k: v for k, v in self.items()
            if k not in _SYSTEM_KEYS
            and k not in _MODEL_FEATURE_FIELDS
            and k not in _MODEL_OBJECT_KEYS
            and not any(k.startswith(p) for p in _FORMAT_SPECIFIC_PREFIXES)
        }

    @property
    def model_features(self) -> dict:
        """
        CP model statistics extracted via ``collect_features()``.

        Fields: ``num_variables``, ``num_bool_variables``, ``num_int_variables``,
        ``num_constraints``, ``constraint_types``, ``has_objective``,
        ``objective_type``, ``domain_size_min``, ``domain_size_max``,
        ``domain_size_mean``.
        """
        return {k: v for k, v in self.items() if k in _MODEL_FEATURE_FIELDS}

    @property
    def model_objects(self) -> dict:
        """
        Live Python objects added by the ``Load`` transform.

        Currently contains:

        - ``variables``: ``{name: CPMpy_variable}`` mapping for every
          decision variable in the loaded model.

        These objects are **not JSON-serialisable** and are excluded from
        ``domain_metadata``, ``to_croissant_example()``, and ``to_gbd_features()``.
        They are available only in-memory after a ``Load`` transform has run.

        .. code-block:: python

            dataset.transform = Load(dataset.loader, open=dataset.open)
            model, info = dataset[0]

            vars = info.model_objects["variables"]
            model.solve()
            print({name: v.value() for name, v in vars.items()})
        """
        return {k: v for k, v in self.items() if k in _MODEL_OBJECT_KEYS}

    @property
    def format_metadata(self) -> dict:
        """
        Format-specific metadata fields (``opb_*``, ``wcnf_*``, ``mps_*``, …).

        These are not portable across format translations.
        """
        return {
            k: v for k, v in self.items()
            if any(k.startswith(p) for p in _FORMAT_SPECIFIC_PREFIXES)
        }

    def without_format(self) -> "InstanceInfo":
        """
        Return a copy with all format-specific metadata removed.

        Use when changing representation format. Chain with ``|`` to add new
        format fields, or use as-is to just strip:

        .. code-block:: python

            # Strip and add new format fields
            return opb_bytes, info.without_format() | extract_opb_features(opb_bytes)
            return opb_bytes, info.without_format() | {"opb_num_variables": 47}

            # Just strip
            return opb_bytes, info.without_format()

            # Simple addition without format change (most common)
            return data, info | {"difficulty": 0.8}
        """
        return InstanceInfo({k: v for k, v in self.items()
                             if not any(k.startswith(p) for p in _FORMAT_SPECIFIC_PREFIXES)})

    def __or__(self, other: dict) -> "InstanceInfo":
        return InstanceInfo(super().__or__(other))

    def __ror__(self, other: dict) -> "InstanceInfo":
        return InstanceInfo(super().__ror__(other))

    def to_croissant(self) -> dict:
        """
        Convert to a Croissant-compatible record.

        Returns a flat dict with ``id``, domain metadata, and model features.
        """
        record: dict = {"id": self.id}
        record.update(self.domain_metadata)
        record.update(self.model_features)
        return record

    def to_gbd(self) -> dict:
        """
        Convert to a GBD-style (Global Benchmark Database) feature record.

        GBD uses hash-based instance IDs; here we use the path-based ``.id``
        property as a stable identifier instead.

        .. note::

            In the future, hash-based instance IDs coming from GBD might be added.
            For now, this has to bed added manually.
        """
        record: dict = {
            "id": self.id,
            "filename": self.get("name", ""),
            "dataset": self.get("dataset", ""),
        }
        record.update(self.categories)
        record.update(self.domain_metadata)
        record.update(self.model_features)
        return record


# ---------------------------------------------------------------------------
# DatasetInfo
# ---------------------------------------------------------------------------

# CP model feature fields documented in dataset cards
_MODEL_FEATURE_DOCS = [
    ("num_variables",      "int",   "Total number of decision variables"),
    ("num_bool_variables", "int",   "Number of Boolean variables"),
    ("num_int_variables",  "int",   "Number of integer variables"),
    ("num_constraints",    "int",   "Total number of constraints"),
    ("constraint_types",   "dict",  'Map: constraint type name → count (e.g. ``{"==": 50, "no_overlap": 3}``)'),
    ("has_objective",      "bool",  "Whether the instance has an objective function"),
    ("objective_type",     "str",   '``"min"``, ``"max"``, or ``"none"``'),
    ("domain_size_min",    "int",   "Minimum variable domain size"),
    ("domain_size_max",    "int",   "Maximum variable domain size"),
    ("domain_size_mean",   "float", "Mean variable domain size"),
]

# schema.org types for model feature fields (for Croissant export)
_MODEL_FEATURE_SCHEMA_ORG = {
    "num_variables": "sc:Integer", "num_bool_variables": "sc:Integer",
    "num_int_variables": "sc:Integer", "num_constraints": "sc:Integer",
    "constraint_types": "sc:StructuredValue", "has_objective": "sc:Boolean",
    "objective_type": "sc:Text", "domain_size_min": "sc:Integer",
    "domain_size_max": "sc:Integer", "domain_size_mean": "sc:Float",
}


class DatasetInfo(dict):
    """
    Dataset-level metadata dict with structured access and export methods.

    Inherits from ``dict`` for full backward compatibility — existing
    ``dataset_metadata()['name']`` access continues unchanged.

    Structured properties (``version``, ``license``, ``tags``, ``domain``,
    ``language``, ``features``) and methods (:meth:`card`, :meth:`to_croissant`)
    are additive.

    Analogous to HuggingFace ``DatasetInfo`` and TFDS ``DatasetInfo``.
    """

    # -- Structured properties ------------------------------------------------

    @property
    def name(self) -> str:
        return self.get("name", "")

    @property
    def description(self) -> str:
        return self.get("description", "")

    @property
    def homepage(self) -> str:
        """Homepage URL (HuggingFace / TFDS naming convention)."""
        return self.get("homepage", "") or self.get("url", "")

    @property
    def features(self) -> Optional[FeaturesInfo]:
        """
        Schema for domain metadata fields.

        Reconstructed from the serialised dict stored in the ``"features"`` key,
        so this property works whether the DatasetInfo was created programmatically
        or loaded from a JSON sidecar.
        """
        raw = self.get("features")
        if raw is None:
            return None
        if isinstance(raw, FeaturesInfo):
            return raw
        if isinstance(raw, dict):
            return FeaturesInfo.from_dict(raw)
        return None

    # -- JSON serialisation ---------------------------------------------------

    def to_jsonable(self) -> dict:
        """
        Return a JSON-serialisable plain dict representation.

        In particular, this serialises ``features`` (when present) to a plain
        dict via :meth:`FeaturesInfo.to_dict`.
        """
        data = dict(self)
        feats = data.get("features")
        if isinstance(feats, FeaturesInfo):
            data["features"] = feats.to_dict()
        return data

    def to_json(self, **kwargs) -> str:
        """
        Return this metadata as a JSON string.

        Arguments:
            **kwargs: forwarded to :func:`json.dumps`.
        """
        import json
        return json.dumps(self.to_jsonable(), **kwargs)

    # -- Card generation ------------------------------------------------------

    def card(self, format: str = "markdown") -> str:
        """
        Generate a Dataset Card.

        Follows the HuggingFace Hub convention: a YAML frontmatter block
        (machine-readable) followed by a markdown body (human-readable).
        Sections are omitted gracefully when data is absent.

        Parameters
        ----------
        format:
            Only ``"markdown"`` is supported currently.

        Returns
        -------
        str
            The dataset card as a string.
        """
        lines: List[str] = []

        # --- YAML frontmatter (HuggingFace convention) ---
        lines.append("---")
        lines.append(f"name: {self.name}")
        if self.version:
            lines.append(f"version: {self.version}")
        lic = self.license
        if lic:
            if isinstance(lic, list):
                lines.append("license:")
                for entry in lic:
                    lines.append(f"  - {entry}")
            else:
                lines.append(f"license: {lic}")
        if self.tags:
            lines.append("tags:")
            for tag in self.tags:
                lines.append(f"  - {tag}")
        lines.append(f"domain: {self.domain}")
        if self.language:
            lines.append(f"language: {self.language}")
        lines.append("---")
        lines.append("")

        # --- Markdown body ---
        lines.append(f"# {self.name} Dataset")
        lines.append("")
        if self.description:
            lines.append(self.description)
            lines.append("")
        if self.homepage:
            lines.append(f"**Homepage:** {self.homepage}")
            lines.append("")

        # License
        if lic:
            lines.append("## License")
            lines.append("")
            if isinstance(lic, list):
                for entry in lic:
                    lines.append(f"- {entry}")
            else:
                lines.append(str(lic))
            lines.append("")

        # Citation
        citations = self.get("citation", [])
        if citations:
            lines.append("## Citation")
            lines.append("")
            for c in citations:
                lines.append(f"- {c}")
            lines.append("")

        # Changelog
        rn = self.release_notes
        if rn:
            lines.append("## Changelog")
            lines.append("")
            for ver, note in rn.items():
                lines.append(f"- **{ver}**: {note}")
            lines.append("")

        # Instance features (domain metadata schema)
        features = self.features
        if features and features.fields:
            lines.append("## Instance Features (Domain Metadata)")
            lines.append("")
            lines.append("| Field | Type | Nullable | Description |")
            lines.append("|-------|------|----------|-------------|")
            for fname, fi in features.fields.items():
                nullable_str = "Yes" if fi.nullable else "No"
                lines.append(f"| `{fname}` | {fi.dtype} | {nullable_str} | {fi.description} |")
            lines.append("")

        # CP model features (always documented)
        lines.append("## CP Model Features (from `collect_features()`)")
        lines.append("")
        lines.append("| Field | Type | Description |")
        lines.append("|-------|------|-------------|")
        for fname, ftype, fdesc in _MODEL_FEATURE_DOCS:
            lines.append(f"| `{fname}` | {ftype} | {fdesc} |")
        lines.append("")

        # Usage example
        lines.append("## Usage")
        lines.append("")
        lines.append("```python")
        # Best-effort class name guess from dataset name
        class_guess = self.name.replace("-", "_").title().replace("_", "") + "Dataset"
        lines.append(f"from cpmpy.tools.datasets import {class_guess}")
        lines.append(f'dataset = {class_guess}(root="./data", download=True)')
        lines.append("for instance, info in dataset:")
        lines.append("    print(info.name, info.domain_metadata)")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    # -- Croissant export -----------------------------------------------------

    def to_croissant(self) -> dict:
        """
        Generate a Croissant-compatible JSON-LD dataset metadata document.

        Follows the `MLCommons Croissant 1.0
        <http://mlcommons.org/croissant/1.0>`_ specification.

        Returns
        -------
        dict
            A JSON-serialisable dict representing the Croissant document.
            Pass to ``json.dumps()`` to get the JSON string.
        """
        doc: Dict[str, Any] = {
            "@context": {
                "@vocab": "https://schema.org/",
                "cr": "http://mlcommons.org/croissant/1.0",
                "sc": "https://schema.org/",
            },
            "@type": "sc:Dataset",
            "name": self.name,
            "description": self.description,
            "url": self.homepage,
        }

        if self.version:
            doc["version"] = self.version
        lic = self.license
        if lic:
            doc["license"] = lic if isinstance(lic, str) else ", ".join(lic)
        if self.tags:
            doc["keywords"] = self.tags

        citations = self.get("citation", [])
        if citations:
            doc["citation"] = "\n".join(citations)

        # Build RecordSet: id + name + path + domain fields + model features
        cr_fields: List[dict] = [
            {"@type": "cr:Field", "name": "id",   "dataType": "sc:Text",
             "description": "Stable instance identifier (dataset/category/name)"},
            {"@type": "cr:Field", "name": "name", "dataType": "sc:Text",
             "description": "Instance name"},
            {"@type": "cr:Field", "name": "path", "dataType": "sc:Text",
             "description": "File path"},
        ]

        features = self.features
        if features:
            cr_fields.extend(features.to_croissant_fields())

        # Standard CP model feature fields
        for fname, fdesc in [(k, d) for k, _, d in _MODEL_FEATURE_DOCS]:
            cr_fields.append({
                "@type": "cr:Field",
                "name": fname,
                "dataType": _MODEL_FEATURE_SCHEMA_ORG.get(fname, "sc:Text"),
                "description": fdesc,
            })

        doc["cr:recordSet"] = [{
            "@type": "cr:RecordSet",
            "name": "instances",
            "cr:field": cr_fields,
        }]

        return doc


# ---------------------------------------------------------------------------
# Standalone adapter functions (for use as target_transform)
# ---------------------------------------------------------------------------

def to_croissant(metadata: dict) -> dict:
    """
    Convert instance metadata to a Croissant record.

    Usable as a ``target_transform``::

        from cpmpy.tools.datasets.metadata import to_croissant
        dataset = JSPLibDataset(root="data", target_transform=to_croissant)
        for instance, record in dataset:
            print(record["id"], record["jobs"])
    """
    return InstanceInfo(metadata).to_croissant()


def to_gbd(metadata: dict) -> dict:
    """
    Convert instance metadata to a GBD-style feature record.

    Usable as a ``target_transform``::

        from cpmpy.tools.datasets.metadata import to_gbd
        dataset = JSPLibDataset(root="data", target_transform=to_gbd)
        for instance, record in dataset:
            print(record["id"], record["num_constraints"])
    """
    return InstanceInfo(metadata).to_gbd()
