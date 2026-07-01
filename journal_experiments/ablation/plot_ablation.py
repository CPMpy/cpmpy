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
from re import U
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
    "no-cp-friendly": "linear-friendly",
    "no-positive-decompositions": "no positive decomp",
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
    df['error'] = df['error'].fillna("OK")

    memlimit = 8
    print(f"Filering to runs with {memlimit}GB of memory")
    df = df[df['memory_limit'] == memlimit]

    # fill baseline
    df["ablate"] = df["ablate"].fillna("baseline")
    df["variant"] = df["ablate"].map(VARIANT_LABEL).fillna(df["ablate"])
    df.loc[df["solver"] == "rc2", "solver"] = "pysat"

    df['total_time'] = df['runtime'] + df['transformation_time']

    return df


def get_finished_instances(df):
    """Keep only runs that found a solution: a feasible satisfaction model or a
    proven optimum for an optimization model."""
    is_finished = ((df["status"] == "FEASIBLE") & df["objective_value"].isna()) | \
                  (df["status"] == "OPTIMAL")
    return df[is_finished]


def save_figure(fig, name):
    print(f"Saving figure to {name}.png/pdf")
    # fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=150, bbox_inches="tight")



def plot_stats(df, figures_dir, subtitle=None, fname=None):
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

    os.makedirs(figures_dir, exist_ok=True)
    for solver in sorted(plot_df["solver"].unique()):
        print("Plotting transform stats for solver:", solver)
        solver_df = plot_df[plot_df["solver"] == solver]
        hue_order = variant_hue_order(solver_df["variant"].unique())

        solver_df = solver_df[solver_df.groupby("model")["model"].transform("size") == len(hue_order)]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, STAT_METRICS):
            
            sns.ecdfplot(data=solver_df, 
                         x=metric, hue="variant", ax=ax, hue_order=hue_order, stat="count")
            #ax.set_xscale("log")
            ax.set_xlabel(STAT_METRICS[metric])
            ax.set_ylabel("Number of instances")
            
        title = f"Model size after transformations for {solver}"
        if subtitle is not None:
            title += f"\n{subtitle}"
        
        fig.suptitle(title)
        fig.tight_layout()
        if fname is None:
            fname = "ablation_stats_{}"
        save_figure(fig, os.path.join(figures_dir, fname.format(solver)))
        plt.close(fig)


def plot_runtime(df, figures_dir, runtime_col="runtime", subtitle=None, fname=None):
    """One ECDF figure per solver, saved into ``figures_dir``."""


    hue_order = variant_hue_order(df["variant"].unique())
    solvers = sorted(df["solver"].unique())
    os.makedirs(figures_dir, exist_ok=True)

    time_limits = set(df["time_limit"].unique())
    if len(time_limits) > 1:
        raise ValueError(f"Time limits are not consistent: {time_limits}")
    time_limit = time_limits.pop()

    ylim = int(df.groupby(["solver", "variant"]).size().max() * 1.1) # add 10% margin
    
    for solver in solvers:
        print("Plotting solver:", solver)
        fig, ax = plt.subplots(figsize=(4, 3))
        
        sns.ecdfplot(data=df[df["solver"] == solver], 
                     x=runtime_col, hue="variant", stat="count", ax=ax, hue_order=hue_order)
        
        #ax.set_xscale("log")
        ax.set_xlabel("runtime (seconds)")
        ax.set_ylabel("Number of instances")
        ax.set_xlim(ax.get_xlim()[0], time_limit)
        ax.set_ylim(0, ylim)
        title = f"Runtime for {solver}"
        if subtitle is not None:
            title += f"\n{subtitle}"
        ax.set_title(title)
        if fname is None:
            fname = "ablation_{}"
        save_figure(fig, os.path.join(figures_dir,fname.format(solver)))
        plt.close(fig)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    figures_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FIGURES_DIR

    df = load_results(results_dir)

    print("Raw data:")
    print(df.groupby(["solver", "variant", "status"]).size())
    
    df = df.sort_values(['solver','model','variant'])
    df.to_csv("results_"+results_dir.split("/")[-1]+".csv")

    CSP_prefix = "/cw/dtailocal/ignaceb/datasets/xcsp3/2024/CSP/"
    COP_prefix = "/cw/dtailocal/ignaceb/datasets/xcsp3/2024/COP/"

    # remove the errors
    df = df[df['error'] == "OK"]

    df_csp = df[df['model_path'].str.startswith(CSP_prefix)]
    df_cop = df[df['model_path'].str.startswith(COP_prefix)]


    #finished = get_finished_instances(df_csp)
    #subtitle="XCSP3 2024 CSP"

    #plot_runtime(finished, figures_dir=figures_dir, runtime_col="runtime", subtitle=subtitle, fname="ablation_csp24_{}")
    #plot_stats(df_csp, figures_dir=figures_dir, subtitle=subtitle, fname="ablation_stats_csp24_{}")

    finished = get_finished_instances(df_cop)
    subtitle= "XCSP3 2024 COP"
    
    plot_runtime(finished, figures_dir=figures_dir, runtime_col="runtime", subtitle=subtitle, fname="ablation_cop24_{}")
    plot_stats(df_cop, figures_dir=figures_dir, subtitle=subtitle, fname="ablation_stats_cop24_{}")
