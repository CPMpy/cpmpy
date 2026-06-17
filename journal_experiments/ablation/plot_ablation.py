"""
Plot an ECDF (cactus-style) of the ablation runtimes.

For each solver we draw, per ablation variant, the empirical cumulative
distribution of solve time over all benchmark instances. Runs that errored or
hit the time limit without a solution are treated as *unsolved* and dropped, so
each curve plateaus at the number of instances that variant actually solved
within the time limit (higher/left = better).

Reads the flat JSON records written by ``run_model.py`` (searched recursively),
one file per (model, solver, ablation) run.

Usage:
    python journal_experiments/plot_ablation.py <results_dir> [<figures_dir>]

Run with the repo conda env, e.g.::

    /Users/ignaceb/miniforge3/envs/cpmpy/bin/python journal_experiments/plot_ablation.py run-ablation figures
"""
import os
import sys
import json
import glob
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save to file, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))

# Default locations (overridable via the command line).
DEFAULT_RESULTS_DIR = os.path.join(_HERE, "ablation_runs")
DEFAULT_FIGURES_DIR = os.path.join(_HERE, "figures")

# Human-friendly label per --ablate value (None/missing == the un-ablated baseline).
VARIANT_LABEL = {
    "baseline": "baseline",
    "no-ilpfriendly": "no linear-friendly",
    "no-detect-categorical": "no categorical",
}


def load_results(results_dir):
    """Load every run_model.py JSON under ``results_dir`` into a DataFrame."""
    records = []
    for path in glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True):
        with open(path, "r") as f:
            records.append(json.load(f))
    if not records:
        raise SystemExit("No result JSON files found under {}".format(results_dir))
    return pd.DataFrame(records)


def get_finished_instances(df):
    """Keep only runs that found a solution: a feasible satisfaction model or a
    proven optimum for an optimization model."""
    is_finished = ((df["status"] == "FEASIBLE") & df["objective_value"].isna()) | \
                  (df["status"] == "OPTIMAL")
    return df[is_finished]


def save_figure(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=150, bbox_inches="tight")


def plot_runtime_ecdf(df, ax, solver, runtime_col="runtime", time_limit=None):
    """Draw the per-variant ECDF of solve time for one ``solver`` on ``ax``.

    Each row of ``df`` is one (model, solver, ablation) run as written by
    run_model.py. Unsolved runs are dropped, so each curve plateaus at the
    number of instances that variant solved within the time limit.

    Arguments:
        df: results DataFrame (one row per run), already filtered to one solver.
        ax: matplotlib Axes to draw on.
        solver: solver name (used for the title).
        runtime_col: which runtime column to put on the x-axis.
        time_limit: x-axis upper bound (s); inferred from the data if None.

    Returns:
        The Axes ``ax``.
    """
    if time_limit is None:
        time_limit = float(df["time_limit"].max())

    sns.ecdfplot(data=df, x=runtime_col, hue="variant", stat="count", ax=ax)
    ax.set_xscale("log")
    ax.set_xlim(left=0.01, right=time_limit)
    ax.set_xlabel("solve time (s)")
    ax.set_ylabel("number of instances solved")
    ax.set_title(solver)
    return ax


def plot_all_solvers(df, figures_dir, runtime_col="runtime"):
    """One ECDF figure per solver, saved into ``figures_dir``."""
    plot_df = df.copy()
    plot_df["ablate"] = plot_df["ablate"].fillna("baseline")
    plot_df["variant"] = plot_df["ablate"].map(VARIANT_LABEL).fillna(plot_df["ablate"])
    plot_df = get_finished_instances(plot_df)

    os.makedirs(figures_dir, exist_ok=True)
    for solver in sorted(plot_df["solver"].unique()):
        print("Plotting solver:", solver)
        solver_df = plot_df[plot_df["solver"] == solver]
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_runtime_ecdf(solver_df, ax=ax, solver=solver, runtime_col=runtime_col)
        save_figure(fig, os.path.join(figures_dir, f"ablation_{solver}"))
        plt.close(fig)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    figures_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FIGURES_DIR

    df = load_results(results_dir)
    print(df)

    plot_all_solvers(df, figures_dir=figures_dir, runtime_col="runtime")
