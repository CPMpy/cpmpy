import argparse
import pstats
from pathlib import Path

def print_pretty_profile(pstat_file, sort_by="tottime", output_file=None, limit=50):
    stats = pstats.Stats(str(pstat_file))
    stats.sort_stats(sort_by)

    entries = []
    for func, stat in stats.stats.items():
        filename, lineno, name = func
        cc, nc, tt, ct, callers = stat
        entries.append((
            nc,                        # number of calls
            f"{tt:.3f}",               # total time
            f"{tt/nc if nc else 0:.3f}",  # per-call total time
            f"{ct:.3f}",               # cumulative time
            f"{ct/nc if nc else 0:.3f}",  # per-call cumulative time
            f"{filename}:{lineno}({name})"
        ))

    entries.sort(key=lambda x: float(x[1]), reverse=True)

    header = f"{'ncalls':>10} {'tottime':>8} {'percall':>8} {'cumtime':>8} {'percall':>8}  function"
    output_lines = [header]
    output_lines += [f"{nc:>10} {tt:>8} {tt_pc:>8} {ct:>8} {ct_pc:>8}  {func}" for nc, tt, tt_pc, ct, ct_pc, func in entries[:limit]]

    output_text = "\n".join(output_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output_text)
        print(f"Wrote formatted output to {output_file}")
    else:
        print(f"\n=== {pstat_file.name} ===")
        print(output_text)


def main():
    parser = argparse.ArgumentParser(description="Print .pstat files in pretty columns.")
    parser.add_argument("profile", type=str, nargs="?", default=None,
                        help="Path to a .pstat file or directory containing them.")
    parser.add_argument("--sort", type=str, default="tottime",
                        help="Sort key (default: tottime).")
    parser.add_argument("--out", type=str, default=None,
                        help="Write output to file.")
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum number of entries to print.")
    args = parser.parse_args()

    path = Path(args.profile) if args.profile else Path("profiles")
    if path.is_file():
        print_pretty_profile(path, sort_by=args.sort, output_file=args.out, limit=args.limit)
    elif path.is_dir():
        pstat_files = sorted(path.glob("*.pstat"))
        if not pstat_files:
            print("No .pstat files found.")
            return
        for pstat_file in pstat_files:
            print_pretty_profile(pstat_file, sort_by=args.sort, limit=args.limit)
    else:
        print(f"Invalid path: {path}")

if __name__ == "__main__":
    main()
