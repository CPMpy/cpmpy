"""
Command-line interface for CPMpy.

Usage:
    cpmpy <COMMAND>

Commands:
    version                Show CPMpy version and solver backends
    dataset list           List available datasets
    dataset info <name>    Show dataset details
    dataset download <name> [options]  Download a dataset
"""

import argparse
from cpmpy import __version__
import cpmpy as cp


# ── Dataset class registry ───────────────────────────────────────
# Maps CLI name -> (class, {param_name: default_value})

DATASET_CLASSES = {
    "xcsp3":          ("XCSP3Dataset",          {"year": 2024, "track": "CSP"}),
    "mse":            ("MSEDataset",            {"year": 2024, "track": "exact-unweighted"}),
    "opb":            ("OPBDataset",            {"year": 2024, "track": "OPT-LIN"}),
    "miplib":         ("MIPLibDataset",         {"year": 2024, "track": "exact-unweighted"}),
    "psplib":         ("PSPLibDataset",         {"variant": "rcpsp", "family": "j30"}),
    "nurserostering": ("NurseRosteringDataset", {}),
    "jsplib":         ("JSPLibDataset",         {}),
}


def _import_dataset_class(class_name):
    """Lazily import a dataset class from cpmpy.tools.dataset."""
    import cpmpy.tools.dataset as ds
    return getattr(ds, class_name)


# ── Commands ─────────────────────────────────────────────────────

def command_version(args):
    print(f"CPMpy version: {__version__}")
    cp.SolverLookup().print_version()


def command_dataset_list(args):
    print("Available datasets:\n")
    for name, (cls_name, params) in DATASET_CLASSES.items():
        try:
            cls = _import_dataset_class(cls_name)
            desc = getattr(cls, "description", "")
        except Exception:
            desc = ""
        line = f"  {name:<20s}"
        if desc:
            # Truncate long descriptions
            short = desc if len(desc) <= 60 else desc[:57] + "..."
            line += f" {short}"
        print(line)
    print(f"\nUse 'cpmpy dataset info <name>' for details.")


def command_dataset_info(args):
    name = args.name.lower()
    if name not in DATASET_CLASSES:
        print(f"Unknown dataset: {args.name}")
        print(f"Available: {', '.join(DATASET_CLASSES)}")
        return

    cls_name, params = DATASET_CLASSES[name]
    try:
        cls = _import_dataset_class(cls_name)
        meta = cls.dataset_metadata()
    except Exception as e:
        print(f"Error loading dataset class: {e}")
        return

    print(f"\n  {meta.get('name', name).upper()}")
    print(f"  {'─' * 40}")
    if meta.get("description"):
        print(f"  {meta['description']}")
    print()
    for key in ("domain", "format", "url", "license"):
        val = meta.get(key)
        if val:
            print(f"  {key:<12s} {val}")

    if params:
        print(f"\n  Parameters:")
        for p, default in params.items():
            print(f"    --{p:<14s} (default: {default})")

    # Show example usage
    print(f"\n  Example:")
    arg_parts = []
    for p, default in params.items():
        arg_parts.append(f"--{p} {default}")
    extra = (" " + " ".join(arg_parts)) if arg_parts else ""
    print(f"    cpmpy dataset download {name}{extra}")
    print()


def command_dataset_download(args):
    name = args.name.lower()
    if name not in DATASET_CLASSES:
        print(f"Unknown dataset: {args.name}")
        print(f"Available: {', '.join(DATASET_CLASSES)}")
        return

    cls_name, param_defaults = DATASET_CLASSES[name]

    # Build constructor kwargs from CLI args
    kwargs = {"root": args.root, "download": True}

    for param, default in param_defaults.items():
        cli_val = getattr(args, param, None)
        if cli_val is not None:
            # Cast to int if the default is int
            if isinstance(default, int):
                try:
                    cli_val = int(cli_val)
                except ValueError:
                    pass
            kwargs[param] = cli_val
        else:
            kwargs[param] = default

    cls = _import_dataset_class(cls_name)
    print(f"Downloading {name} dataset...")
    for param, default in param_defaults.items():
        print(f"  {param}: {kwargs.get(param, default)}")
    print(f"  root: {args.root}")
    print()

    try:
        dataset = cls(**kwargs)
        print(f"\nDone! {len(dataset)} instances downloaded to {args.root}/")
    except Exception as e:
        print(f"\nError: {e}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPMpy command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # cpmpy version
    version_parser = subparsers.add_parser("version", help="Show version information on CPMpy and its solver backends")
    version_parser.set_defaults(func=command_version)

    # cpmpy dataset ...
    dataset_parser = subparsers.add_parser("dataset", help="Browse and download benchmark datasets")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    # cpmpy dataset list
    list_parser = dataset_sub.add_parser("list", help="List available datasets")
    list_parser.set_defaults(func=command_dataset_list)

    # cpmpy dataset info <name>
    info_parser = dataset_sub.add_parser("info", help="Show dataset details")
    info_parser.add_argument("name", help="Dataset name")
    info_parser.set_defaults(func=command_dataset_info)

    # cpmpy dataset download <name> [options]
    dl_parser = dataset_sub.add_parser("download", help="Download a dataset")
    dl_parser.add_argument("name", help="Dataset name")
    dl_parser.add_argument("--root", default="./data", help="Download directory (default: ./data)")
    dl_parser.add_argument("--year", default=None, help="Year/edition")
    dl_parser.add_argument("--track", default=None, help="Track/category")
    dl_parser.add_argument("--variant", default=None, help="Variant (e.g. for psplib)")
    dl_parser.add_argument("--family", default=None, help="Family (e.g. for psplib)")
    dl_parser.set_defaults(func=command_dataset_download)

    args = parser.parse_args()
    args.func(args)
