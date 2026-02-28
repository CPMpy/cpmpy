"""
Configuration for CPMpy dataset download origins.

This module provides configuration for custom download origins that can be used
as alternatives to the original dataset sources. Origins are tried in order,
falling back to the original source if all custom origins fail.

Configuration can be set via:
1. Environment variables (CPMPY_DATASET_ORIGINS_{DATASET_NAME})
2. This config file
3. Class attributes in dataset classes
"""

import os
from typing import Dict, List, Optional

# Default origins configuration
# Format: {dataset_name: [list of URL bases]}
_DEFAULT_ORIGINS: Dict[str, List[str]] = {
    # Example:
    # "xcsp3": ["https://cpmpy-datasets.example.com/xcsp3"],
    # "mse": ["https://cpmpy-datasets.example.com/mse"],
}

def get_origins(dataset_name: str) -> List[str]:
    """
    Get custom origins for a dataset.
    
    Checks in order:
    1. Environment variable CPMPY_DATASET_ORIGINS_{DATASET_NAME}
    2. _DEFAULT_ORIGINS dictionary
    3. Returns empty list (no custom origins)
    
    Arguments:
        dataset_name (str): Name of the dataset (e.g., "xcsp3", "mse")
        
    Returns:
        List[str]: List of origin URL bases to try
    """
    # Check environment variable first
    env_var = f"CPMPY_DATASET_ORIGINS_{dataset_name.upper()}"
    env_value = os.getenv(env_var)
    if env_value:
        # Split by comma and strip whitespace
        return [url.strip() for url in env_value.split(",") if url.strip()]
    
    # Check default origins
    return _DEFAULT_ORIGINS.get(dataset_name, [])

def set_default_origin(dataset_name: str, origin_url: str):
    """
    Set a default origin URL for a dataset (for programmatic configuration).
    
    Arguments:
        dataset_name (str): Name of the dataset
        origin_url (str): Base URL for the origin
    """
    if dataset_name not in _DEFAULT_ORIGINS:
        _DEFAULT_ORIGINS[dataset_name] = []
    if origin_url not in _DEFAULT_ORIGINS[dataset_name]:
        _DEFAULT_ORIGINS[dataset_name].append(origin_url)

def set_default_origins(dataset_name: str, origin_urls: List[str]):
    """
    Set multiple default origin URLs for a dataset (for programmatic configuration).
    
    Arguments:
        dataset_name (str): Name of the dataset
        origin_urls (List[str]): List of base URLs for origins
    """
    _DEFAULT_ORIGINS[dataset_name] = origin_urls.copy()



