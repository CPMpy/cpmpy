"""
SAT Competition Dataset

Instances from the benchmark database (benchmark-database.de) for the SAT competition.
"""

import io
import lzma
import os
import pathlib
import re
import tempfile
from urllib.request import Request, urlopen

from cpmpy.tools.dataset._base import URLDataset
from cpmpy.tools.dataset.utils import download as download_manager


# Base URL for the instance list (getinstances returns one file URL per line)
INSTANCE_LIST_URL = "https://benchmark-database.de/getinstances"
DEFAULT_QUERY = "track=main_2025"
DEFAULT_CONTEXT = "cnf"


class SATDataset(URLDataset):
    """
    SAT competition benchmark dataset (DIMACS CNF).

    Instances are listed at benchmark-database.de via getinstances; each line
    is a URL to a CNF file (served XZ-compressed). Files are stored as .cnf.xz.

    More information: https://benchmark-database.de/
    """

    name = "sat"
    description = "SAT competition benchmark instances (DIMACS CNF) from benchmark-database.de."
    url = "https://benchmark-database.de/"
    license = ""
    citation = []

    def __init__(
        self,
        root: str = ".",
        track: str = "main_2025",
        context: str = "cnf",
        transform=None,
        target_transform=None,
        download: bool = False,
        **kwargs
    ):
        """
        Constructor for the SAT competition dataset.

        Arguments:
            root (str): Root directory where the dataset is stored or will be downloaded (default=".").
            track (str): Track query parameter for getinstances (default="main_2025").
            context (str): Context query parameter for getinstances (default="cnf").
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dict.
            download (bool): If True, download the instance list and all instances if not present (default=False).
            **kwargs: Passed through to download() (e.g. workers for parallel downloads).
        """
        self.root = pathlib.Path(root)
        self.track = track
        self.context = context

        dataset_dir = self.root / self.name / track / context

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            target_transform=target_transform,
            download=download,
            extension=".cnf.xz",
            **kwargs
        )

    @staticmethod
    def reader(file_path, open=open):
        """
        Reader for SAT dataset.
        Parses a DIMACS CNF file path into a CPMpy model (uses open for .cnf.xz).
        """
        with open(file_path) as f:
            content = f.read()
        return SATDataset.loader(content)

    @staticmethod
    def loader(content: str):
        """
        Loader for SAT dataset.
        Loads a CPMpy model from raw DIMACS CNF content string.
        """
        from cpmpy.tools.dimacs import load_dimacs
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".cnf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return load_dimacs(tmp_path)
        finally:
            os.unlink(tmp_path)

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        """Open instance file; use lzma for .cnf.xz (XZ-compressed) files."""
        path = str(instance)
        return lzma.open(instance, "rt") if path.endswith(".xz") else open(instance, "r")

    def instance_metadata(self, file: pathlib.Path) -> dict:
        """Add instance metadata; ensure name strips .cnf from stem (e.g. hash.cnf.xz -> hash)."""
        metadata = super().instance_metadata(file)
        stem = pathlib.Path(file).stem
        if stem.endswith(".cnf"):
            metadata["name"] = stem[:-4]
        return metadata

    def category(self) -> dict:
        return {
            "track": self.track,
            "context": self.context,
        }

    def collect_instance_metadata(self, file) -> dict:
        """Extract num variables and num clauses from DIMACS p-line."""
        result = {}
        try:
            with self.open(file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("p"):
                        match = re.search(r"p\s+cnf\s+(\d+)\s+(\d+)", line)
                        if match:
                            result["dimacs_num_variables"] = int(match.group(1))
                            result["dimacs_num_clauses"] = int(match.group(2))
                        break
        except Exception:
            pass
        return result

    def download(self, **kwargs):
        """Fetch the instance list from getinstances, then download each CNF file via the download manager."""
        params = f"query=track%3D{self.track}&context={self.context}"
        list_url = f"{INSTANCE_LIST_URL}?{params}"

        print(f"Fetching SAT instance list from {list_url}")
        req = Request(list_url)
        with urlopen(req) as response:
            body = response.read().decode("utf-8")

        # One file URL per line (e.g. http://benchmark-database.de/file/00d5a43a...)
        file_urls = [line.strip() for line in body.splitlines() if line.strip()]

        if not file_urls:
            raise ValueError(
                f"No instances returned from {list_url}. "
                "Check track and context parameters."
            )

        # Use last path segment (hash) as filename; store as .cnf.xz (server sends XZ-compressed)
        def path_to_name(url: str) -> str:
            name = url.rstrip("/").split("/")[-1]
            if name.lower().endswith(".cnf.xz"):
                return name
            if name.lower().endswith(".cnf"):
                return f"{name}.xz"
            return f"{name}.cnf.xz"

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate by destination (instance list may contain duplicate URLs)
        seen_dest = set()
        items = []
        for url in file_urls:
            dest = self.dataset_dir / path_to_name(url)
            if dest not in seen_dest:
                seen_dest.add(dest)
                items.append((url, dest))

        workers = kwargs.get("workers", 1)
        print(f"Downloading {len(items)} SAT instances to {self.dataset_dir} (workers={workers})")
        download_manager(
            items,
            desc_prefix="Instance",
            skip_existing=True,
            **kwargs,
        )

        files = self._list_instances()
        if not files:
            raise ValueError(
                f"Download completed but no .cnf.xz files found in {self.dataset_dir}"
            )
        self._collect_all_metadata()
        print(f"Finished downloading {len(files)} instances")


if __name__ == "__main__":
    dataset = SATDataset(track="main_2025", context="cnf", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
