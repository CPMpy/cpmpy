"""
Pseudo Boolean Competition (PB) Dataset

https://www.cril.univ-artois.fr/PB25/
"""

import fnmatch
import lzma
import os
import pathlib
import tarfile
import io

from cpmpy.tools.datasets.core import FileDataset
from cpmpy.tools.datasets.metadata import FeaturesInfo, FieldInfo


class OPBDataset(FileDataset):
    """
    Pseudo Boolean Competition (PB) benchmark dataset.

    Provides access to benchmark instances from the Pseudo Boolean
    competitions. Instances are grouped by `year` and `track` (e.g.,
    `"OPT-LIN"`, `"DEC-LIN"`) and stored as `.opb.xz` files.
    If the dataset is not available locally, it can be automatically
    downloaded and extracted.

    More information on the competition can be found here: https://www.cril.univ-artois.fr/PB25/
    """

    name = "opb"
    description = "Pseudo-Boolean Competition benchmark instances."
    homepage = "https://www.cril.univ-artois.fr/PB25/"
    citation = [
        "Berre, D. L., Parrain, A. The Pseudo-Boolean Evaluation 2011. JSAT, 7(1), 2012.",
    ]

    features = FeaturesInfo({
        "author":              ("str", "Author extracted from filename convention"),
        "opb_num_variables":   ("int", "Number of Boolean variables (from OPB header)"),
        "opb_num_constraints": ("int", "Number of constraints (from OPB header)"),
        "opb_num_products":    FieldInfo("int", "Number of non-linear product terms (from OPB header)", nullable=True),
    })

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "OPT-LIN", 
            competition: bool = True,
            transform=None, target_transform=None, 
            download: bool = False,
            **kwargs
        ):
        """
        Constructor for a dataset object of the PB competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Competition year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the competition instances to load (default="OPT-LIN").
            competition (bool): If True, the dataset will filtered on competition-used instances.
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).


        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
        """

        self.root = pathlib.Path(root)
        self.year = year
        self.track = track
        self.competition = competition

        # Check requested dataset
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. exact-weighted, exact-unweighted, ...")

        dataset_dir = self.root / self.name / str(year) / track / ('selected' if self.competition else 'normalized')
        
        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".opb.xz",
            **kwargs
        )

    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }

    def categories(self) -> dict:
        return self.category()

    def collect_instance_metadata(self, file: os.PathLike) -> dict:
        """Extract metadata from OPB filename and file header.

        Parses the `* #variable= ... #constraint= ...` header line and
        extracts the author from the filename convention (first part before `_`).
        """
        import re
        result = {}
        # Author from filename
        filename = pathlib.Path(file).name
        parts = filename.split("_")
        if len(parts) > 1:
            result["author"] = parts[0]
        # Parse header for variable/constraint counts
        try:
            with self.open(file) as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("*"):
                        break
                    var_match = re.search(r'#variable=\s*(\d+)', line)
                    con_match = re.search(r'#constraint=\s*(\d+)', line)
                    if var_match:
                        result["opb_num_variables"] = int(var_match.group(1))
                    if con_match:
                        result["opb_num_constraints"] = int(con_match.group(1))
                    prod_match = re.search(r'#product=\s*(\d+)', line)
                    if prod_match:
                        result["opb_num_products"] = int(prod_match.group(1))
        except Exception:
            pass
        return result
                
    def download(self):
                
        url = "https://www.cril.univ-artois.fr/PB24/benchs/"
        target = f"{'normalized' if not self.competition else 'selected'}-PB{str(self.year)[2:]}.tar"
        target_download_path = self.root / target

        print(f"Downloading OPB {self.year} {self.track} {'competition' if self.competition else 'non-competition'} instances from www.cril.univ-artois.fr")
        
        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
        except ValueError as e:
            raise ValueError(f"No dataset available for year {self.year}. Error: {str(e)}")
        
        # Extract only the specific track folder from the tar
        with tarfile.open(target_download_path, "r:*") as tar_ref:  # r:* handles .tar, .tar.gz, .tar.bz2, etc.
            # Get the main folder name
            main_folder = None
            for name in tar_ref.getnames():
                if "/" in name:
                    main_folder = name.split("/")[0]
                    break

            if main_folder is None:
                raise ValueError("Could not find main folder in tar file")

            # Extract only files from the specified track
            # Get all unique track names from tar
            if not self.competition:
                tracks = set()
                for member in tar_ref.getmembers():
                    parts = member.name.split("/")
                    if len(parts) > 2 and parts[0] == main_folder:
                        tracks.add(parts[1])
            else:
                tracks = set()
                for member in tar_ref.getmembers():
                    parts = member.name.split("/")
                    if len(parts) > 2 and parts[0] == main_folder:
                        tracks.add(parts[2])

            # Check if requested track exists
            if self.track not in tracks:
                raise ValueError(f"Track '{self.track}' not found in dataset. Available tracks: {sorted(tracks)}")

            # Create track folder in root directory
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

            # Extract files for the specified track
            if not self.competition:
                prefix = f"{main_folder}/{self.track}/"
            else:
                prefix = f"{main_folder}/*/{self.track}/"
            for member in tar_ref.getmembers():
                if fnmatch.fnmatch(member.name, prefix + "*") and member.isfile():
                    # Path relative to main_folder/track
                    # Find where the track folder ends and get everything after
                    track_marker = f"/{self.track}/"
                    marker_pos = member.name.find(track_marker)
                    relative_path = member.name[marker_pos + len(track_marker):]

                    # Flatten: replace "/" with "_" to encode subfolders (some instances have clashing names)
                    flat_name = relative_path#.replace("/", "_")
                    target_path = self.dataset_dir / flat_name

                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    with tar_ref.extractfile(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        # Clean up the tar file
        target_download_path.unlink()

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        return lzma.open(instance, 'rt') if str(instance).endswith(".xz") else open(instance)


if __name__ == "__main__":
    dataset = OPBDataset(year=2024, track="DEC-LIN", competition=True, download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
