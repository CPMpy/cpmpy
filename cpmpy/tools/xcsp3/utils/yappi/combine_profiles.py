import pstats
from pathlib import Path
import argparse
import os

def combine_pstats(profile_dir):
    combined_stats = None
    profile_dir = Path(profile_dir)

    for profile_file in sorted(profile_dir.glob("*.pstat")):
        stats = pstats.Stats(str(profile_file))
        if combined_stats is None:
            combined_stats = stats
        else:
            combined_stats.add(stats)

    return combined_stats


def relativize_path(path, base):
    """Return path relative to base, or original if not under base."""
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


def print_pretty_stats(stats, limit=50, fullpath=False):
    cwd = Path.cwd()
    if not fullpath:
        stats.strip_dirs()  # remove directory info if not full path

    entries = []
    for func, data in stats.stats.items():
        filename, lineno, funcname = func
        cc, nc, tt, ct, callers = data

        if fullpath:
            display_file = relativize_path(filename, cwd)
        else:
            display_file = os.path.basename(filename)

        entries.append({
            "ncalls": nc,
            "tottime": tt,
            "percall": tt / nc if nc else 0,
            "cumtime": ct,
            "percall_cum": ct / nc if nc else 0,
            "func": f"{display_file}:{lineno}({funcname})"
        })

    entries.sort(key=lambda e: e["tottime"], reverse=True)

    header_fmt = "{:>10} {:>8} {:>8} {:>8} {:>8}  {}"
    print(header_fmt.format("ncalls", "tottime", "percall", "cumtime", "percall", "function"))
    
    line_fmt = "{ncalls:10} {tottime:8.3f} {percall:8.3f} {cumtime:8.3f} {percall_cum:8.3f}  {func}"
    for e in entries[:limit]:
        print(line_fmt.format(**e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and print pstats files with aligned columns.")
    parser.add_argument("profile_dir", nargs="?", default="profiles",
                        help="Directory containing .pstat files (default: profiles)")
    parser.add_argument("--limit", type=int, default=50, help="Max number of entries to print")
    parser.add_argument("--fullpath", action="store_true", help="Show full file paths (relative to cwd)")

    args = parser.parse_args()

    combined = combine_pstats(args.profile_dir)
    if combined:
        print_pretty_stats(combined, limit=args.limit, fullpath=args.fullpath)
    else:
        print("No profiles found.")
