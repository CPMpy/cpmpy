"""
MIPLib Dataset

https://maxsat-evaluations.github.io/
"""


import os
import gzip
import zipfile
import pathlib
import io

from cpmpy.tools.datasets._base import FileDataset


class MIPLibDataset(FileDataset):  # torch.utils.data.Dataset compatible

    """
    MIPLib Dataset in a PyTorch compatible format.

    More information on MIPLib can be found here: https://miplib.zib.de/
    """

    name = "miplib"
    description = "Mixed Integer Programming Library benchmark instances."
    homepage = "https://miplib.zib.de/"
    citation = [
        "Gleixner, A., Hendel, G., Gamrath, G., Achterberg, T., Bastubbe, M., Berthold, T., Christophel, P. M., Jarck, K., Koch, T., Linderoth, J., Lubbecke, M., Mittelmann, H. D., Ozyurt, D., Ralphs, T. K., Salvagnin, D., and Shinano, Y. MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library. Mathematical Programming Computation, 2021. https://doi.org/10.1007/s12532-020-00194-3.",
    ]

    version = "2017"
    license = "CC BY 4.0"
    domain = "mip"
    tags = ["optimization", "mixed-integer-programming", "mip", "combinatorial"]
    language = "MPS"
   

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "exact-unweighted", 
            transform=None, target_transform=None, 
            download: bool = False,
            metadata_workers: int = 1
        ):
        """
        Constructor for a dataset object of the MIPLib competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the dataset instances to load (default="exact-unweighted").
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

        dataset_dir = self.root / self.name / str(year) / track

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".mps.gz",
            metadata_workers=metadata_workers
        )

    @staticmethod
    def reader(file_path, open=open):
        """
        Reader for MIPLib dataset.
        Parses a file path directly into a CPMpy model.
        For backward compatibility. Consider using read() + load() instead.
        """
        from cpmpy.tools.io.scip import load_scip
        return load_scip(file_path, open=open)

    @staticmethod
    def loader(content: str):
        """
        Loader for MIPLib dataset.
        Loads a CPMpy model from raw MPS/LP content string.
        Note: SCIP requires a file, so content is written to a temporary file.
        """
        import tempfile
        import os
        from cpmpy.tools.io.scip import load_scip
        
        # SCIP requires a file path, so write content to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.mps') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            return load_scip(tmp_path)
        finally:
            os.unlink(tmp_path)

    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }
    
    def download(self):
        
        url = "https://miplib.zib.de/downloads/"
        target = "collection.zip"
        target_download_path = self.root / target

        print("Downloading MIPLib instances from miplib.zib.de")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path), origins=self.origins)
        except ValueError as e:
            raise ValueError(f"No dataset available on {url}. Error: {str(e)}")
        
        # Extract files
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:                    
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())

        # Clean up the zip file
        target_download_path.unlink()

    def collect_instance_metadata(self, file) -> dict:
        """Extract row/column counts from MPS file sections."""
        result = {}
        try:
            with self.open(file) as f:
                section = None
                num_rows = 0
                columns = set()
                has_objective = False
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("NAME"):
                        section = "NAME"
                    elif stripped == "ROWS":
                        section = "ROWS"
                    elif stripped == "COLUMNS":
                        section = "COLUMNS"
                    elif stripped in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                        section = stripped
                    elif section == "ROWS" and stripped:
                        parts = stripped.split()
                        if parts[0] == "N":
                            has_objective = True
                        else:
                            num_rows += 1
                    elif section == "COLUMNS" and stripped:
                        parts = stripped.split()
                        if parts:
                            columns.add(parts[0])
                    elif section in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                        pass  # skip to avoid parsing entire file
                        if section == "ENDATA":
                            break
                result["mps_num_rows"] = num_rows
                result["mps_num_columns"] = len(columns)
                result["mps_has_objective"] = has_objective
        except Exception:
            pass
        return result

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        return gzip.open(instance, "rt") if str(instance).endswith(".gz") else open(instance)


if __name__ == "__main__":
    dataset = MIPLibDataset(download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
