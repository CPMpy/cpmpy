"""
MaxSAT Evaluation (MSE) Dataset

https://maxsat-evaluations.github.io/
"""


import os
import lzma
from typing import Optional
import zipfile
import pathlib
import io

from cpmpy.tools.datasets.core import FileDataset
from cpmpy.tools.datasets.metadata import FeaturesInfo


class MaxSATEvalDataset(FileDataset):  # torch.utils.data.Dataset compatible

    """
    MaxSAT Evaluation benchmark dataset.

    Provides access to benchmark instances from the MaxSAT Evaluation
    competitions. Instances are grouped by `year` and `track` (e.g.,
    `"exact-unweighted"`, `"exact-weighted"`) and stored as `.wcnf.xz` files.
    If the dataset is not available locally, it can be automatically
    downloaded and extracted.

    More information on the competition can be found here: https://maxsat-evaluations.github.io/
    """

    # -------------------------- Dataset-level metadata -------------------------- #

    name = "maxsateval"
    description = "MaxSAT Evaluation competition benchmark instances."
    homepage = "https://maxsat-evaluations.github.io/"
    citation = []

    features = FeaturesInfo({
        "wcnf_num_variables":        ("int", "Number of propositional variables"),
        "wcnf_num_clauses":          ("int", "Total number of clauses (hard + soft)"),
        "wcnf_num_hard_clauses":     ("int", "Number of hard clauses"),
        "wcnf_num_soft_clauses":     ("int", "Number of soft clauses"),
        "wcnf_total_literals":       ("int", "Total number of literals across all clauses"),
        "wcnf_num_distinct_weights": ("int", "Number of distinct soft clause weights"),
    })

    # ---------------------------------------------------------------------------- #

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "exact-unweighted", 
            transform=None, target_transform=None, 
            download: bool = False,
            dataset_dir: Optional[os.PathLike] = None,
            **kwargs
        ):
        """
        Constructor for a dataset object of the MaxSAT Evaluation competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). If `dataset_dir` is provided, this argument is ignored.
            year (int): Competition year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the competition instances to load (default="exact-unweighted").
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).
            dataset_dir (Optional[os.PathLike]): Path to the dataset directory. If not provided, it will be inferred from the root and year/track.

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
        """

        # Dataset-specific attributes
        self.root = pathlib.Path(root)
        self.year = year
        self.track = track

        # Check requested dataset is valid
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. OPT-LIN, DEC-LIN, ...")

        dataset_dir = pathlib.Path(dataset_dir) / str(year) / track if dataset_dir else self.root / self.name / str(year) / track

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".wcnf.xz",
            **kwargs
        )


    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }

    def categories(self) -> dict:
        return self.category()

    def collect_instance_metadata(self, file) -> dict:
        """
        Extract statistics from WCNF header comments.

        WCNF files from MSE contain JSON-like statistics in comment lines:
        nvars, ncls, nhards, nsofts, total_lits, nsoft_wts, and length stats.
        """
        import re
        result = {}
        try:
            with self.open(file) as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("c"):
                        break
                    # Extract all numeric fields from JSON-style comments
                    for key, meta_key in [
                        ("nvars", "wcnf_num_variables"),
                        ("ncls", "wcnf_num_clauses"),
                        ("nhards", "wcnf_num_hard_clauses"),
                        ("nsofts", "wcnf_num_soft_clauses"),
                        ("total_lits", "wcnf_total_literals"),
                        ("nsoft_wts", "wcnf_num_distinct_weights"),
                    ]:
                        match = re.search(rf'"{key}"\s*:\s*(\d+)', line)
                        if match:
                            result[meta_key] = int(match.group(1))
        except Exception:
            pass
        return result

    def download(self):
        url = f"https://www.cs.helsinki.fi/group/coreo/MSE{self.year}-instances/"
        target = f"mse{str(self.year)[2:]}-{self.track}.zip"
        target_download_path = self.root / target

        print(f"Downloading MaxSAT Eval {self.year} {self.track} instances from cs.helsinki.fi")
                
        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
        except ValueError as e:
            raise ValueError(f"No dataset available for year {self.year} and track {self.track}. Error: {str(e)}")
        
        # Extract only the specific track folder from the tar
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:                    
            # Create track folder in root directory, parents=True ensures recursive creation
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                # Extract file to family_dir, removing main_folder/track prefix
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())

        # Clean up the zip file
        target_download_path.unlink()

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        return lzma.open(instance, "rt") if str(instance).endswith(".xz") else open(instance)


if __name__ == "__main__":
    dataset = MaxSATEvalDataset(year=2024, track="exact-weighted", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
