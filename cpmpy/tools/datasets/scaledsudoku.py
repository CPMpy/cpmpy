"""
Scaled Sudoku is a corpus of Sudoku puzzles of varying sizes with propagation hardness labels.
Origin: https://github.com/zayenz/scaled-sudoku-instances
"""

from __future__ import annotations

import json
import os
import pathlib
import zipfile
from typing import Any, Optional, Callable

from cpmpy.tools.datasets.core import FileDataset
from cpmpy.tools.io.sudoku import parse_sudoku


class ScaledSudokuDataset(FileDataset):  # torch.utils.data.Dataset compatible
    """
    Scaled Sudoku Dataset in a PyTorch compatible format.

    - Origin: https://github.com/zayenz/scaled-sudoku-instances
    - Reference: Mikael Z. Lagerkvist. 'Scaling Sudoku as a Constraint Problem.', ModRef 2026.
      https://2026.modref.org/papers/ModRef2026-07-Scaling-Sudoku.pdf

    Instances are available as unique-solution ``base`` puzzles and as easier
    ``walks`` variants obtained by adding givens. Grid sizes range from ``6x6``
    to ``36x36``. Each puzzle file (``.sdk.txt``) has a matching ``.sdk.json``
    sidecar with solution and hardness metadata.

    To load an instance into a CPMpy model, use :func:`~cpmpy.tools.io.sudoku.load_sudoku`.
    For examples of using a loader as a dataset ``transform``, see the
    :ref:`modeling guide <modeling-datasets>`.

    Arguments:
        root (str): Root directory containing the instances (if 'download', instances will be downloaded to this location)
        kind (str): Instance collection, ``base`` or ``walks`` (default="base")
        size (str): Grid size, one of ``6x6``, ``9x9``, ``16x16``, ``25x25``, ``36x36`` (default="9x9")
        transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
        target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """

    name = "scaledsudoku"
    description = "Scaled Sudoku benchmark instances of varying sizes with propagation hardness labels."
    homepage = "https://github.com/zayenz/scaled-sudoku-instances"
    citation = [
        "Mikael Z. Lagerkvist. 'Scaling Sudoku as a Constraint Problem.', ModRef 2026. https://2026.modref.org/papers/ModRef2026-07-Scaling-Sudoku.pdf",
    ]

    KINDS = ("base", "walks")
    SIZES = ("6x6", "9x9", "16x16", "25x25", "36x36")

    def __init__(self, root: str = ".", kind: str = "base", size: str = "9x9",
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 download: bool = False, **kwargs: Any):
        """
        Constructor for a dataset object for scaled Sudoku instances.

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested kind/size combination is not available.
        """
        self.root = pathlib.Path(root)
        self.kind = kind
        self.size = size

        if kind not in self.KINDS:
            raise ValueError(f"Unknown kind {kind!r}. Must be any of {', '.join(self.KINDS)}")
        if size not in self.SIZES:
            raise ValueError(f"Unknown size {size!r}. Must be any of {', '.join(self.SIZES)}")

        dataset_dir = self.root / self.name / self.kind / self.size

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform,
            download=download, extension=".sdk.txt",
            **kwargs
        )

        # The corpus is extracted as a whole, so a later kind/size may already be
        # on disk without CPMpy metadata sidecars from the original download.
        if not self._ignore_sidecar:
            files = self._list_instances()
            if files and not self._metadata_path(files[0]).exists():
                self._collect_all_metadata()

    @classmethod
    def parse(cls, instance: os.PathLike) -> dict[str, Any]:
        """
        Parse a Sudoku instance into a grid and box shape.
        """
        return parse_sudoku(instance, open=cls.open)

    def categories(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "size": self.size,
        }

    def collect_instance_metadata(self, file: pathlib.Path) -> dict[str, Any]:
        """
        Extract metadata from the matching ``.sdk.json`` sidecar when available.
        """
        result: dict[str, Any] = {}
        sidecar_path = file.with_suffix(".json")  # puzzle-....sdk.txt -> puzzle-....sdk.json
        try:
            if sidecar_path.exists():
                with open(sidecar_path, "r") as f:
                    sidecar = json.load(f)

                result["source_id"] = sidecar.get("id")
                if "clues_count" in sidecar:
                    result["clues_count"] = sidecar["clues_count"]

                size_info = sidecar.get("size") or {}
                if "order" in size_info:
                    result["order"] = size_info["order"]
                if "box_width" in size_info:
                    result["box_width"] = size_info["box_width"]
                if "box_height" in size_info:
                    result["box_height"] = size_info["box_height"]

                hardness = sidecar.get("hardness") or {}
                if "level" in hardness:
                    result["hardness"] = hardness["level"]

                minimality = sidecar.get("minimality") or {}
                if "is_minimal" in minimality:
                    result["is_minimal"] = minimality["is_minimal"]

                identity = sidecar.get("identity") or {}
                if "solution_hash" in identity:
                    result["solution_hash"] = identity["solution_hash"]

                provenance = sidecar.get("provenance") or {}
                if "kind" in provenance:
                    result["provenance_kind"] = provenance["kind"]
            else:
                # Fall back to parsing the puzzle header only
                data = self.parse(file)
                result["order"] = data["size"]
                result["box_width"] = data["box_width"]
                result["box_height"] = data["box_height"]
                result["clues_count"] = int((data["grid"] != 0).sum())
        except Exception:
            pass
        return result

    def download(self):
        # GitHub only ships a full-repo archive (no per-size bundles), so download
        # once and extract every kind/size under root/scaledsudoku/.
        url = "https://github.com/zayenz/scaled-sudoku-instances/archive/refs/heads/"
        target = "main.zip"
        target_download_path = self.root / target
        dataset_root = self.root / self.name

        print("Downloading scaled Sudoku corpus from github.com/zayenz/scaled-sudoku-instances")

        try:
            target_download_path = self._download_file(
                url, target, destination=str(target_download_path),
                desc="scaledsudoku.zip",
            )
        except ValueError as e:
            raise ValueError(f"Failed to download scaled Sudoku corpus. Error: {str(e)}")

        repo_prefix = "scaled-sudoku-instances-main/"
        with zipfile.ZipFile(target_download_path, "r") as zip_ref:
            dataset_root.mkdir(parents=True, exist_ok=True)

            extracted = 0
            for file_info in zip_ref.infolist():
                if file_info.is_dir() or not file_info.filename.startswith(repo_prefix):
                    continue
                if not (file_info.filename.endswith(".sdk.txt") or file_info.filename.endswith(".sdk.json")):
                    continue

                rel = file_info.filename[len(repo_prefix):]
                if rel.startswith("base/"):
                    dest_rel = rel
                elif rel.startswith("walks/variants/"):
                    dest_rel = "walks/" + rel[len("walks/variants/"):]
                else:
                    continue

                dest = dataset_root / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(file_info) as source, open(dest, "wb") as out:
                    out.write(source.read())
                extracted += 1

        target_download_path.unlink()

        if extracted == 0:
            raise ValueError("Archive contained no scaled Sudoku instances")

        print(f"Extracted {extracted} puzzle and sidecar files under {dataset_root}")

        if not self.dataset_dir.exists():
            raise ValueError(
                f"Archive did not contain kind {self.kind!r} and size {self.size!r}"
            )


if __name__ == "__main__":
    dataset = ScaledSudokuDataset(kind="base", size="6x6", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
