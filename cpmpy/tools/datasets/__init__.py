from .core import (
    expand_varying_kwargs,
    FileDataset,
)
from .utils import (
    extract_model_features,
    portable_instance_metadata,
)
from .metadata import (
    InstanceInfo,
    DatasetInfo,
    FeaturesInfo,
    FieldInfo,
    to_croissant,
    to_gbd,
)

__all__ = [
    # Base
    "FileDataset",
    "extract_model_features",
    "expand_varying_kwargs",
    "portable_instance_metadata",
    # Metadata
    "InstanceInfo",
    "DatasetInfo",
    "FeaturesInfo",
    "FieldInfo",
    "to_croissant",
    "to_gbd",
    # Datasets
    "MIPLibDataset",
    "JSPLibDataset",
    "PSPLibDataset",
    "NurseRosteringDataset",
    "XCSP3Dataset",
    "OPBDataset",
    "MaxSATEvalDataset",
    "SATDataset",
    # Transforms
    "Compose",
    "Open",
    "Load",
    "Parse",
    "Serialize",
    "Translate",
    "SaveToFile",
    "Lambda",
    "extract_format_metadata",
]
from .miplib import MIPLibDataset
from .jsplib import JSPLibDataset
from .psplib import PSPLibDataset
from .nurserostering import NurseRosteringDataset
from .xcsp3 import XCSP3Dataset
from .opb import OPBDataset
from .mse import MaxSATEvalDataset
from .sat import SATDataset
from .transforms import Compose, Open, Load, Serialize, Translate, SaveToFile, Lambda, extract_format_metadata
# Backward compatibility alias
Parse = Load
