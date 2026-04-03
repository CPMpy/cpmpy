import os
import json
from itertools import product

import pytest
import cpmpy as cp

from cpmpy.tools.datasets.xcsp3 import XCSP3Dataset


# ---------------------------------------------------------------------------- #
#                            Pytest parametrisation                            #
# ---------------------------------------------------------------------------- #

RAW_DATASET_SPECS = [
    {
        "id": "xcsp3",
        "dataset_cls": XCSP3Dataset,
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


def _categories_from_metadata(metadata):
    # Keep compatibility with both keys used across refactors.
    return metadata.get("categories", metadata.get("category", {}))


# ---------------------------------------------------------------------------- #
#                                     Tests                                    #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope="module", params=DATASET_SPECS, ids=lambda spec: spec["id"])
def downloaded_dataset(request, tmp_path_factory):
    spec = request.param
    root = tmp_path_factory.mktemp(spec["id"])
    dataset = spec["dataset_cls"](
        root=str(root),
        download=True,
        **spec["init_kwargs"],
    )
    assert len(dataset) > 0
    return spec, dataset


@pytest.mark.dataset_download
@pytest.mark.timeout(180)
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
    sidecar_dataset._collect_all_metadata(force=True, workers=1)

    instances = sidecar_dataset._list_instances()
    assert len(instances) > 0

    instance_path = instances[0]
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
    assert "model_features" in sidecar
