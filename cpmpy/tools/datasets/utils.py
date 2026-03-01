"""
Dataset utilities: generic download manager.

Downloads one or multiple files from URLs. Supports optional parallel downloads
via a configurable worker count. How files are fetched (HTTP, progress bars,
chunking) is encapsulated here; datasets only pass (url, destination) and options.
"""

import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union
from urllib.request import Request, urlopen


def _get_content_length(url: str) -> int:
    """Return Content-Length for url, or 0 if unknown."""
    try:
        req = Request(url)
        req.get_method = lambda: "HEAD"
        with urlopen(req) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        return 0


def _download_url(
    url: str,
    destination: Union[str, pathlib.Path],
    desc: str = None,
    chunk_size: int = 1024 * 1024,
    _sequential_impl=None,
) -> pathlib.Path:
    """
    Download a single file from url to destination.
    Uses _sequential_impl(url, path, total_size, desc, chunk_size) if provided,
    otherwise delegates to the dataset base implementation.
    """
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if desc is None:
        desc = destination.name
    total_size = _get_content_length(url)
    if _sequential_impl is None:
        from cpmpy.tools.dataset._base import _Dataset
        _sequential_impl = _Dataset._download_sequential
    _sequential_impl(url, destination, total_size, desc, chunk_size)
    return destination


def download_manager(
    url: Union[str, List[str]],
    destination: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]] = None,
    *,
    workers: int = 1,
    desc_prefix: str = "Downloading",
    chunk_size: int = 1024 * 1024,
    skip_existing: bool = True,
    **kwargs,
) -> Union[pathlib.Path, List[pathlib.Path]]:
    """
    Generic download manager: one URL or many, sequential or parallel.

    Single file:
        path = download("https://example.com/file.zip", "/tmp/file.zip")
        path = download("https://example.com/file.zip", destination="/tmp/out.zip", workers=1)

    Multiple files (list of (url, destination)):
        paths = download([("https://a.com/1.cnf", "/data/1.cnf"), ...], workers=4)

    Arguments:
        url: Either a single URL string, or a list of URL strings.
        destination: For single-URL mode, path to save the file. For multiple-URL mode, list of matching destination paths.
        workers: Number of parallel download workers. 1 = sequential. >1 = parallel (only for multiple files).
        desc_prefix: Prefix for progress description (e.g. "Instance 1/100").
        chunk_size: Chunk size in bytes for streaming.
        skip_existing: If True, skip pairs where destination already exists (multi-file only).
        **kwargs: Ignored; allows callers to pass through options (e.g. from dataset download(**kwargs)).

    Returns:
        For single URL: path to the downloaded file.
        For multiple: list of paths that were downloaded (skipped files are not in the list).
    """
    if isinstance(url, str):
        if destination is None:
            raise ValueError("destination is required when passing a single URL")
        return _download_url(url, destination, desc=desc_prefix or url, chunk_size=chunk_size)

    items: List[Tuple[str, pathlib.Path]] = [
        (url, pathlib.Path(dest)) for url, dest in zip(url, destination)
    ]

    if not items:
        return []

    if skip_existing:
        items = [(u, d) for u, d in items if not d.exists()]

    if not items:
        return []

    if workers is None or workers <= 1:
        # Sequential
        results = []
        for i, (url, dest) in enumerate(items):
            desc = f"{desc_prefix} {i + 1}/{len(items)} {dest.name}"
            try:
                results.append(_download_url(url, dest, desc=desc, chunk_size=chunk_size))
            except Exception as e:
                warnings.warn(f"Failed to download {url}: {e}")
        return results

    # Parallel
    max_workers = min(workers, len(items))
    results = []
    errors = []

    def do_one(url: str, dest: pathlib.Path, idx: int) -> Tuple:
        desc = f"{desc_prefix} {idx + 1}/{len(items)} {dest.name}"
        try:
            return _download_url(url, dest, desc=desc, chunk_size=chunk_size), None
        except Exception as e:
            return None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(do_one, url, dest, i): (url, dest)
            for i, (url, dest) in enumerate(items)
        }
        for future in as_completed(futures):
            result, err = future.result()
            if result is not None:
                results.append(result)
            else:
                url, dest = futures[future]
                errors.append((dest.name, err))

    if errors:
        warnings.warn(
            f"Failed to download {len(errors)}/{len(items)} files. "
            f"First error: {errors[0][0]} - {errors[0][1]}"
        )

    return results


# Convenience alias for multi-file callers
download_many = download_manager
