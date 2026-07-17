import os
import json
import shutil
from itertools import product

import pytest
import cpmpy as cp

from cpmpy.tools.datasets.xcsp3 import XCSP3Dataset
from cpmpy.tools.datasets.jsplib import JSPLibDataset
from cpmpy.tools.datasets.psplib import PSPLibDataset
from cpmpy.tools.datasets.miplib import MIPLibDataset
from cpmpy.tools.datasets.mse import MaxSATEvalDataset
from cpmpy.tools.datasets.opb import OPBDataset
from cpmpy.tools.datasets.sat import SATDataset
from cpmpy.tools.datasets.nurserostering import NurseRosteringDataset

# Matching model loaders for each dataset (turn a raw instance into a cp.Model).
from cpmpy.tools.io import (
    load_xcsp3,
    load_jsplib,
    load_rcpsp,
    load_scip_format,
    load_wcnf,
    load_opb,
    load_dimacs,
    load_nurserostering,
)


# ---------------------------------------------------------------------------- #
#                            Pytest parametrisation                            #
# ---------------------------------------------------------------------------- #

RAW_DATASET_SPECS = [
    {
        "id": "xcsp3",
        "dataset_cls": XCSP3Dataset,
        "loader": load_xcsp3,
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        # Single category field:
        # - dict: one cartesian group (scalars are treated as fixed dimensions)
        # - list[dict]: multiple groups; each dict can be:
        #     - explicit combo (all scalar values), or
        #     - cartesian group (one/more list-valued dimensions)
        #
        # Explicit combinations example:
        # "categories": [
        #     {"year": 2024, "track": "MiniCOP"},
        #     {"year": 2025, "track": "COP"},
        # ],
        #
        # Cartesian product example:
        # "categories": {"year": [2024, 2025], "track": ["COP", "CSP"]},
        "categories": [
            {"year": 2024, "track": ["COP", "CSP"]},
        ],
        "expected_instance_suffix": ".xml.lzma",
        "expected_categories": {},
    },
    {
        "id": "jsplib",
        "dataset_cls": JSPLibDataset,
        "loader": load_jsplib,
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        "categories": None,  # no category dimensions
        "expected_instance_suffix": "",  # JSPLib instances have no file extension
        "expected_categories": {},
    },
    {
        "id": "psplib",
        "dataset_cls": PSPLibDataset,
        "loader": load_rcpsp,
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        "categories": [
            {"variant": "rcpsp", "family": "j30"},
        ],
        "expected_instance_suffix": ".sm",
        "expected_categories": {},
    },
    {
        "id": "miplib",
        "dataset_cls": MIPLibDataset,
        "loader": load_scip_format,
        "download_timeout": 3600,  # collection.zip is several GB
        # Not every MIPLib instance is representable in CPMpy (e.g. continuous
        # variables); point the loader tests at the pure-integer 'gen-ip' family.
        "loader_instance": "gen-ip",
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        "categories": [
            {"year": 2024, "track": "exact-unweighted"},
        ],
        "expected_instance_suffix": ".mps.gz",
        "expected_categories": {},
    },
    {
        "id": "mse",
        "dataset_cls": MaxSATEvalDataset,
        "loader": load_wcnf,
        "download_timeout": 3600,  # full track archive can be large
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        "categories": [
            {"year": 2024, "track": "exact-unweighted"},
        ],
        "expected_instance_suffix": ".wcnf.xz",
        "expected_categories": {},
    },
    {
        "id": "opb",
        "dataset_cls": OPBDataset,
        "loader": load_opb,
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        "categories": [
            {"year": 2024, "track": "OPT-LIN"},
        ],
        "expected_instance_suffix": ".opb.xz",
        "expected_categories": {},
    },
    {
        "id": "sat",
        "dataset_cls": SATDataset,
        "loader": load_dimacs,
        "download_timeout": 3600,  # downloads many/large CNF instances
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        "categories": [
            {"track": "main_2025", "context": "cnf"},
        ],
        "expected_instance_suffix": ".cnf.xz",
        "expected_categories": {},
    },
    {
        "id": "nurserostering",
        "dataset_cls": NurseRosteringDataset,
        "loader": load_nurserostering,
        "init_kwargs": {
            "ignore_sidecar": True,
        },
        "categories": None,  # no category dimensions
        "expected_instance_suffix": ".txt",
        "expected_categories": {},
    },
]


def _expand_category_group(group: dict) -> list[dict]:
    if not isinstance(group, dict) or not group:
        raise ValueError("Each category group must be a non-empty dict.")

    keys = list(group.keys())
    values_per_key = []
    for key in keys:
        value = group[key]
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                raise ValueError(f"Category dimension '{key}' has an empty value list.")
            values_per_key.append(list(value))
        else:
            values_per_key.append([value])

    return [dict(zip(keys, combo)) for combo in product(*values_per_key)]


def _expand_dataset_specs(raw_specs):
    expanded = []
    for spec in raw_specs:
        categories = spec.get("categories")

        if isinstance(categories, dict):
            combinations = _expand_category_group(categories)
        elif isinstance(categories, list):
            combinations = []
            for group in categories:
                combinations.extend(_expand_category_group(group))
        elif categories is None:
            expanded.append(spec.copy())
            continue
        else:
            raise ValueError(
                f"Spec '{spec['id']}' has invalid 'categories' value. "
                "Use dict or list[dict]."
            )

        for category_combo in combinations:
            expanded_spec = spec.copy()
            combo_id = "_".join(f"{k}{str(v).lower()}" for k, v in category_combo.items())
            expanded_spec["id"] = f"{spec['id']}_{combo_id}"

            init_kwargs = dict(spec.get("init_kwargs", {}))
            init_kwargs.update(category_combo)
            expanded_spec["init_kwargs"] = init_kwargs

            expected_categories = dict(spec.get("expected_categories", {}))
            expected_categories.update(category_combo)
            expanded_spec["expected_categories"] = expected_categories

            expanded.append(expanded_spec)
    return expanded


DATASET_SPECS = _expand_dataset_specs(RAW_DATASET_SPECS)
SIDE_CAR_METADATA_SAMPLE_SIZE = 3


def _categories_from_metadata(metadata):
    # Keep compatibility with both keys used across refactors.
    return metadata.get("categories", metadata.get("category", {}))


# ---------------------------------------------------------------------------- #
#                                     Tests                                    #
# ---------------------------------------------------------------------------- #


DEFAULT_DOWNLOAD_TIMEOUT = 180

# Attach a per-dataset timeout marker to each fixture param. The actual download
# happens during this fixture's (module-scoped) setup, whose time is attributed
# to the first test that uses it; the marker therefore governs the download.
# Large datasets (e.g. MIPLib's multi-GB collection.zip) override the default.
_DATASET_PARAMS = [
    pytest.param(
        spec,
        marks=pytest.mark.timeout(spec.get("download_timeout", DEFAULT_DOWNLOAD_TIMEOUT)),
        id=spec["id"],
    )
    for spec in DATASET_SPECS
]


@pytest.fixture(scope="module", params=_DATASET_PARAMS)
def downloaded_dataset(request, tmp_path_factory):
    spec = request.param
    root = tmp_path_factory.mktemp(spec["id"])
    try:
        dataset = spec["dataset_cls"](
            root=str(root),
            download=True,
            **spec["init_kwargs"],
        )
        assert len(dataset) > 0
        yield spec, dataset
    finally:
        # Free this dataset's files before the next one is fetched. pytest's
        # tmp_path_factory otherwise retains them for the whole session, so
        # downloads accumulate and can exhaust the disk quota (some datasets are
        # multiple GB). The download runs inside the try so that even a *failed*
        # download (e.g. quota exceeded mid-transfer) still gets its partial
        # files removed here, preventing it from breaking every later dataset.
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.dataset_download
def test_dataset_download_stage(downloaded_dataset):
    _spec, dataset = downloaded_dataset
    assert dataset._check_exists()
    assert dataset.dataset_dir.exists()
    assert len(dataset) > 0


@pytest.mark.dataset_download
def test_dataset_categories_interface(downloaded_dataset):
    spec, dataset = downloaded_dataset
    categories = dataset.categories()
    assert isinstance(categories, dict)
    for key, value in spec["expected_categories"].items():
        assert categories[key] == value


@pytest.mark.dataset_download
def test_dataset_list_instances(downloaded_dataset):
    spec, dataset = downloaded_dataset
    instances = dataset._list_instances()
    assert len(instances) > 0
    assert all(str(p).endswith(spec["expected_instance_suffix"]) for p in instances)


@pytest.mark.dataset_download
def test_dataset_getitem_metadata(downloaded_dataset):
    spec, dataset = downloaded_dataset
    instance_path, metadata = dataset[0]
    assert instance_path.endswith(spec["expected_instance_suffix"])
    assert os.path.exists(instance_path)
    assert metadata["dataset"] == dataset.name
    categories = _categories_from_metadata(metadata)
    for key, value in spec["expected_categories"].items():
        assert categories[key] == value


@pytest.mark.dataset_download
def test_dataset_open_and_read(downloaded_dataset):
    spec, dataset = downloaded_dataset
    instance_path, _metadata = dataset[0]
    assert instance_path.endswith(spec["expected_instance_suffix"])

    with dataset.open(instance_path) as f:
        header = f.readline()
    assert isinstance(header, str)
    assert len(header) > 0
    contents = dataset.read(instance_path)
    assert isinstance(contents, str)
    assert len(contents) >= len(header)


@pytest.mark.dataset_download
def test_dataset_collect_instance_metadata(downloaded_dataset):
    _spec, dataset = downloaded_dataset
    instance_path, _metadata = dataset[0]
    collected = dataset.collect_instance_metadata(instance_path)
    assert isinstance(collected, dict)


def _loadable_instance_index(spec, dataset):
    """Return the index of an instance the dataset's loader can build a model from.

    Most datasets load their first instance fine, but some collections (e.g.
    MIPLib) contain instances CPMpy cannot represent (such as continuous
    variables). An optional per-spec ``loader_instance`` substring points the
    loader tests at a supported instance/family; from there we pick the first
    one that actually loads. Skips if none can be loaded.
    """
    loader = spec["loader"]
    instances = dataset._list_instances()

    hint = spec.get("loader_instance")
    if hint is not None:
        candidates = [(i, p) for i, p in enumerate(instances) if hint in p.name]
        if not candidates:
            pytest.skip(f"No instance matching {hint!r} found in dataset")
    else:
        candidates = list(enumerate(instances))

    for i, path in candidates[:50]:
        try:
            loader(str(path), open=dataset.open)
            return i
        except Exception:
            continue
    pytest.skip("No loadable instance found for this dataset's loader")


@pytest.mark.dataset_download
def test_dataset_loader_function(downloaded_dataset):
    """The dataset's matching io loader turns a raw instance file into a cp.Model.

    Exercises the loader as a standalone function, opening the (possibly
    compressed) instance through the dataset's ``open`` callable.
    """
    spec, dataset = downloaded_dataset
    if spec.get("loader") is None:
        pytest.skip("No loader configured for this dataset")

    index = _loadable_instance_index(spec, dataset)
    instance_path, _metadata = dataset[index]
    model = spec["loader"](instance_path, open=dataset.open)
    assert isinstance(model, cp.Model)


@pytest.mark.dataset_download
def test_dataset_loader_as_transform(downloaded_dataset):
    """The matching io loader works as a ``transform``: ``dataset[i]`` yields a cp.Model.

    Reuses the already-downloaded files (``download=False``) and lets the
    dataset pass its ``open`` to the loader automatically.
    """
    spec, dataset = downloaded_dataset
    if spec.get("loader") is None:
        pytest.skip("No loader configured for this dataset")

    index = _loadable_instance_index(spec, dataset)
    transformed = spec["dataset_cls"](
        root=str(dataset.root),
        download=False,
        transform=spec["loader"],
        **spec["init_kwargs"],
    )
    model, metadata = transformed[index]
    assert isinstance(model, cp.Model)
    # metadata is still returned alongside the transformed instance
    assert metadata["dataset"] == dataset.name


@pytest.fixture(scope="module")
def sidecar_payload(downloaded_dataset):
    spec, dataset = downloaded_dataset

    # Use a fresh dataset instance rooted at the already-downloaded files.
    sidecar_dataset = spec["dataset_cls"](
        root=str(dataset.root),
        download=False,
        **spec["init_kwargs"],
    )

    # Metadata collection expects a reader that returns a CPMpy model.
    sidecar_dataset.reader = lambda *_args, **_kwargs: cp.Model()

    instances = sidecar_dataset._list_instances()
    assert len(instances) > 0

    # Sidecar tests only need a representative sample. Some public datasets are
    # large enough that collecting metadata for every instance can exceed CI
    # disk/time limits, even though the dataset downloaded successfully.
    sample_instances = instances[:SIDE_CAR_METADATA_SAMPLE_SIZE]
    sidecar_dataset._list_instances = lambda: sample_instances
    sidecar_dataset._collect_all_metadata(force=True)

    instance_path = sample_instances[0]
    meta_path = sidecar_dataset._metadata_path(instance_path)
    assert meta_path.exists()

    with open(meta_path, "r") as f:
        sidecar = json.load(f)
    return spec, sidecar_dataset, instance_path, meta_path, sidecar


@pytest.mark.dataset_download
def test_sidecar_file_creation(sidecar_payload):
    _spec, _dataset, _instance_path, meta_path, _sidecar = sidecar_payload
    assert meta_path.exists()


@pytest.mark.dataset_download
def test_sidecar_dataset_and_categories(sidecar_payload):
    spec, dataset, _instance_path, _meta_path, sidecar = sidecar_payload
    assert sidecar["dataset"]["name"] == dataset.name
    for key, value in spec["expected_categories"].items():
        assert sidecar["categories"][key] == value


@pytest.mark.dataset_download
def test_sidecar_source_file_and_model_features(sidecar_payload):
    _spec, _dataset, instance_path, _meta_path, sidecar = sidecar_payload
    assert sidecar["source_file"].endswith(instance_path.name)
