#!/usr/bin/env python
"""
Plot ablation study results written by ``run_model.py``.

1. **Runtime ECDFs** — per solver, cactus-style curves of solve time per variant.
2. **Transform-size ECDFs** — per solver, cumulative distribution of
   ``n_constraints``, ``n_integer``, and ``n_boolean`` across instances for each
   ablation variant.

Reads flat JSON records (searched recursively), one file per (model, solver,
ablation) run.

Usage:
    python journal_experiments/ablation/plot_ablation.py <results_dir> [<figures_dir>]

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
    "baseline": "pipeline",
    "no-ilpfriendly": "no linear-friendly",
    "no-detect-categorical": "no categorical",
    "ilpfriendly": "linear-friendly",
}

STAT_METRICS = {
    "n_constraints": "# constraints",
    "n_integer": "# integer vars",
    "n_boolean": "# boolean vars",
}

def variant_hue_order(variants):
    hue_order = sorted(variants)
    baseline = VARIANT_LABEL["baseline"]
    if baseline in hue_order:
        hue_order.remove(baseline)
        hue_order.insert(0, baseline)
    return hue_order


def load_results(results_dir):
    """Load every run_model.py JSON under ``results_dir`` into a DataFrame."""
    records = []
    for path in glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True):
        with open(path, "r") as f:
            content = f.read()
            if len(content) > 0: # some files are empty due to crashes
                records.append(json.loads(content))
    if not records:
        raise SystemExit("No result JSON files found under {}".format(results_dir))

    df = pd.DataFrame(records)
    df['error'].fillna("OK", inplace=True)
    df['status'].fillna("OK", inplace=True)
    # fill baseline
    df["ablate"] = df["ablate"].fillna("baseline")
    df["variant"] = df["ablate"].map(VARIANT_LABEL).fillna(df["ablate"])
    df.loc[df["solver"] == "rc2", "solver"] = "pysat"
    return df


def get_finished_instances(df):
    """Keep only runs that found a solution: a feasible satisfaction model or a
    proven optimum for an optimization model."""
    is_finished = ((df["status"] == "FEASIBLE") & df["objective_value"].isna()) | \
                  (df["status"] == "OPTIMAL")
    return df[is_finished]


def save_figure(fig, name):
    # fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=150, bbox_inches="tight")


def plot_runtime_ecdf(df, ax, solver, runtime_col="runtime", time_limit=None, hue_order=None):
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

    if hue_order is None:
        hue_order = variant_hue_order(df["variant"].unique())

    sns.ecdfplot(data=df, x=runtime_col, hue="variant", stat="count", ax=ax, hue_order=hue_order)
    ax.set_xscale("log")
    ax.set_xlim(left=0.01, right=time_limit)
    ax.set_xlabel("solve time (s)")
    ax.set_ylabel("number of instances solved")
    ax.set_title(solver)
    return ax

def plot_stats(df, ax, metric, hue_order=None):
    """ECDF of one transform-size metric across ablation variants."""

    df = df.copy()
    if hue_order is None:
        hue_order = variant_hue_order(df["variant"].unique())
    
    # num_solved = df.groupby("variant").size()
    # df['variant'] = df["variant"].map(lambda x: f"{x} ({num_solved[x]})")

    sns.ecdfplot(data=df, x=metric, hue="variant", ax=ax, hue_order=hue_order, stat="count")
    if metric == "n_constraints":
        ax.set_xscale("log")
    ax.set_xlabel(STAT_METRICS[metric])
    ax.set_ylabel("cumulative proportion")
    return ax


def plot_all_stats(df, figures_dir):
    """One figure per solver: ECDFs of constraints / integer / boolean vars."""
    plot_df = df.copy()
    for col in STAT_METRICS:
        if col not in plot_df.columns:
            print(f"No {col} column in results — skipping transform-size plots")
            return

    plot_df = plot_df.dropna(subset=list(STAT_METRICS.keys()))
    if plot_df.empty:
        print("No transform-size records found — skipping transform-size plots")
        return

    print(plot_df.groupby(["solver", "variant"]).size())

    os.makedirs(figures_dir, exist_ok=True)
    for solver in sorted(plot_df["solver"].unique()):
        print("Plotting transform stats for solver:", solver)
        solver_df = plot_df[plot_df["solver"] == solver]
        hue_order = variant_hue_order(solver_df["variant"].unique())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, STAT_METRICS):
            plot_stats(solver_df, ax=ax, metric=metric, hue_order=hue_order)
        
        fig.suptitle(f"{solver} transform size", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, os.path.join(figures_dir, f"ablation_stats_{solver}"))
        plt.close(fig)


def plot_all_solvers(df, figures_dir, runtime_col="runtime"):
    """One ECDF figure per solver, saved into ``figures_dir``."""
    plot_df = df.copy()

    time_limits = set(df["time_limit"].dropna())
    if len(time_limits) > 1:
        raise ValueError(f"Time limits are not consistent: {time_limits}")

    print(plot_df.groupby(["solver", "variant"]).size())

    plot_df = get_finished_instances(plot_df)

    hue_order = variant_hue_order(plot_df["variant"].unique())

    print(plot_df["solver"].unique())

    os.makedirs(figures_dir, exist_ok=True)
    for solver in sorted(plot_df["solver"].unique()):
        print("Plotting solver:", solver)
        solver_df = plot_df[plot_df["solver"] == solver]
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_runtime_ecdf(solver_df, ax=ax, solver=solver, runtime_col=runtime_col,
                          hue_order=hue_order)
        save_figure(fig, os.path.join(figures_dir, f"ablation_{solver}"))
        plt.close(fig)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    figures_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FIGURES_DIR

    df = load_results(results_dir)

    print(df.columns)

   

    print("Raw data:")
    print(df.groupby(["solver", "variant", "status"]).size())

    # plot_all_solvers(df, figures_dir=figures_dir, runtime_col="runtime")
    plot_all_stats(df, figures_dir=figures_dir)
