#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Experiment: track how the PySAT transformation stack reshapes a model.

This replays the exact sequence of transformations used by
:meth:`cpmpy.solvers.pysat.CPM_pysat.transform`, but stops after every stage to
record three metrics on the intermediate constraint list:

1. ``n_constraints``: the number of (top-level) constraints,
2. ``n_integer``: the number of integer variables, and
3. ``n_boolean``: the number of Boolean variables.

It then runs this over every pickled model in ``journal_experiments`` and
aggregates the per-stage metrics into a single pandas DataFrame, indexed by the
model's filename (without the ``.pickle`` suffix). The DataFrame has a 2-level
column index ``(stage, metric)``.
"""
import os
import glob
import time
from typing import Dict, Tuple

import pandas as pd

from cpmpy import Model
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl

# transformations used by CPM_pysat.transform (mirrored here, stage by stage)
from cpmpy.transformations.normalize import toplevel_list, simplify_boolean
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.transformations.negation import push_down_negation
from cpmpy.transformations.linearize import (
    decompose_linear,
    linearize_constraint,
    linearize_reified_variables,
    only_positive_coefficients,
)
from cpmpy.transformations.flatten_model import flatten_constraint
from cpmpy.transformations.reification import only_implies, only_bv_reifies
from cpmpy.transformations.int2bool import int2bool


def _count_variables(cpm_cons) -> Tuple[int, int]:
    """Number of (integer, Boolean) variables occurring in ``cpm_cons``."""
    n_integer = 0
    n_boolean = 0
    for v in get_variables(cpm_cons):
        # note: _BoolVarImpl is not a subclass of _IntVarImpl, so this is exclusive
        if isinstance(v, _BoolVarImpl):
            n_boolean += 1
        elif isinstance(v, _IntVarImpl):
            n_integer += 1
    return n_integer, n_boolean


def experiment_transformation(model: Model) -> Dict[str, Dict[str, int]]:
    """Apply the PySAT transformation stack to ``model``, stage by stage.

    Arguments:
        model: a CPMpy :class:`~cpmpy.model.Model`.

    Returns:
        A dictionary mapping a transformation-stage name to a dictionary of
        metrics ``{"n_constraints": ..., "n_integer": ..., "n_boolean": ...}``
        as measured after that stage. It also includes an ``"input"`` entry for
        the untouched model constraints.
    """
    # an empty solver instance gives us the same state the transform() relies on
    # (a fresh common-subexpression map, integer-encoding map, supported sets, ...)
    solver = CPM_pysat()

    stats: Dict[str, Dict[str, int]] = {}

    def record(stage: str, cpm_cons) -> None:
        n_integer, n_boolean = _count_variables(cpm_cons)
        stats[stage] = {
            "n_constraints": len(cpm_cons),
            "n_integer": n_integer,
            "n_boolean": n_boolean,
        }

    def run_stage(stage: str, func, cpm_cons):
        """Run a transformation, logging its name and runtime, and record stats."""
        print(f"\t{stage}", end="", flush=True)
        start = time.perf_counter()
        cpm_cons = func(cpm_cons)
        runtime = time.perf_counter() - start
        print(f" ({runtime:.3f}s)", flush=True)
        record(stage, cpm_cons)
        return cpm_cons

    # stage 0: the raw model constraints
    cpm_cons = list(model.constraints)
    record("input", cpm_cons)

    cpm_cons = run_stage("toplevel_list", toplevel_list, cpm_cons)

    cpm_cons = run_stage(
        "no_partial_functions",
        lambda c: no_partial_functions(c, safen_toplevel={"div", "mod", "element"}),
        cpm_cons,
    )

    cpm_cons = run_stage("push_down_negation", push_down_negation, cpm_cons)

    cpm_cons = run_stage(
        "decompose_linear",
        lambda c: decompose_linear(
            c,
            supported=solver.supported_global_constraints,
            supported_reified=solver.supported_reified_global_constraints,
            csemap=solver._csemap,
        ),
        cpm_cons,
    )

    cpm_cons = run_stage("simplify_boolean", simplify_boolean, cpm_cons)

    cpm_cons = run_stage(
        "flatten_constraint",
        lambda c: flatten_constraint(c, csemap=solver._csemap),
        cpm_cons,
    )

    cpm_cons = run_stage(
        "linearize_reified_variables",
        lambda c: linearize_reified_variables(
            c, min_values=2, csemap=solver._csemap, ivarmap=solver.ivarmap
        ),
        cpm_cons,
    )

    cpm_cons = run_stage(
        "only_bv_reifies",
        lambda c: only_bv_reifies(c, csemap=solver._csemap),
        cpm_cons,
    )

    cpm_cons = run_stage(
        "only_implies",
        lambda c: only_implies(c, csemap=solver._csemap),
        cpm_cons,
    )

    cpm_cons = run_stage(
        "linearize_constraint",
        lambda c: linearize_constraint(
            c,
            supported=frozenset({"sum", "wsum", "->", "and", "or"}),
            csemap=solver._csemap,
        ),
        cpm_cons,
    )

    cpm_cons = run_stage(
        "int2bool",
        lambda c: int2bool(
            c, solver.ivarmap, encoding=solver.encoding, csemap=solver._csemap
        ),
        cpm_cons,
    )

    cpm_cons = run_stage(
        "only_positive_coefficients", only_positive_coefficients, cpm_cons
    )

    return stats


_HERE = os.path.dirname(os.path.abspath(__file__))


def run_experiments(models_dir: str = os.path.join(_HERE, "models")) -> pd.DataFrame:
    """Run :func:`experiment_transformation` on every pickled model in a folder.

    Arguments:
        models_dir: directory containing ``*.pickle`` serialized CPMpy models.

    Returns:
        A pandas DataFrame indexed by the model filename (without the
        ``.pickle`` suffix), with a 2-level column index ``(stage, metric)``
        where metric is one of ``n_constraints``, ``n_integer``, ``n_boolean``.
    """
    rows = {}
    for path in sorted(glob.glob(os.path.join(models_dir, "*.pickle"))):
        name = os.path.splitext(os.path.basename(path))[0]
        print(name)
        try:
            model = Model.from_file(path)
            stats = experiment_transformation(model)
        except Exception as e:  # keep going, report which model failed
            print(f"\t[FAIL] {type(e).__name__}: {e}")
            continue

        row = {}
        for stage, metrics in stats.items():
            for metric, value in metrics.items():
                row[(stage, metric)] = value
        rows[name] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["stage", "metric"])
    df.index.name = "model"
    return df


def load_results(path: str = os.path.join(_HERE, "transformation_stats.csv")) -> pd.DataFrame:
    """Read a previously saved results CSV back into a pandas DataFrame.

    Reconstructs the same structure produced by :func:`run_experiments`: the
    model name as index and a 2-level column index ``(stage, metric)``.

    Arguments:
        path: path to the CSV written by :func:`run_experiments`.

    Returns:
        A pandas DataFrame indexed by model name with a ``(stage, metric)``
        column MultiIndex.
    """
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.index.name = "model"
    df.columns.names = ["stage", "metric"]
    return df

def pretty_latex(df):

    def latex_escape(label):
        return str(label).replace("_", r"\_").replace("#", r"\#")

    # move the 'metric' column level onto the rows: each model (instance) now
    # gets one row per metric (#constraints / #integer vars / #bool vars),
    # while the stages become the columns
    df = df.stack(level="metric", future_stack=True)

    # order the metric rows consistently within each instance
    metric_order = ["n_constraints", "n_integer", "n_boolean"]
    df = df.reindex(metric_order, level="metric")

    # rename the metric row labels to something more compact
    renames = dict(n_constraints="#cons", n_integer="#int", n_boolean="#bool")
    df = df.rename(index=renames, level="metric")

    # round the numbers to 0 decimal places
    df = df.round(0)

    # escape underscores and "#" signs in stage names, model names, and metrics
    df = df.rename(index=latex_escape, columns=latex_escape)

    # print latex
    print(
        df.to_latex(
            index=True,
            escape=False,
            column_format=f"l|" + "c"*len(df.columns),
        )
    )


def plot_transformation_stats(df, stages=None, save_basename="transformation_stats"):
    """Visualize how the transformation pipeline reshapes the models.

    Because absolute sizes differ by orders of magnitude across instances, most
    panels normalize *per instance* (relative to the ``input`` stage), turning
    raw counts into comparable "blow-up factors".

    Produces a 3x2 figure:

    1. absolute #constraints per stage (boxplot, log scale),
    2. absolute #variables per stage (boxplot, log scale),
    3. constraint blow-up factor per stage (boxplot, log scale),
    4. variable blow-up factor per stage (boxplot, log scale),
    5. absolute #constraints trajectory per instance (spaghetti, log scale),
    6. boolean share of variables per stage (boxplot) -- exposes ``int2bool``.

    Arguments:
        df: DataFrame as returned by :func:`run_experiments` / :func:`load_results`
            (model index, ``(stage, metric)`` column MultiIndex).
        stages: optional ordered list of stages to include (defaults to all,
            in pipeline order as they appear in ``df``).
        save_basename: basename (without extension) for the saved figures;
            written as ``<basename>.pdf`` and ``<basename>.png`` in this folder.

    Returns:
        The matplotlib Figure.
    """
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save to file
    import matplotlib.pyplot as plt

    # select stages (preserve the order in which they appear in the columns)
    all_stages = list(dict.fromkeys(df.columns.get_level_values("stage")))
    if stages is None:
        stages = all_stages
    else:
        stages = [s for s in stages if s in all_stages]

    # pull out per-metric tables: rows = instances, cols = stages
    cons = df.xs("n_constraints", axis=1, level="metric")[stages]
    ints = df.xs("n_integer", axis=1, level="metric")[stages]
    bools = df.xs("n_boolean", axis=1, level="metric")[stages]
    total_vars = ints + bools

    # per-instance normalization relative to the first (input) stage
    base_stage = stages[0]
    cons_ratio = cons.div(cons[base_stage].replace(0, np.nan), axis=0)
    vars_ratio = total_vars.div(total_vars[base_stage].replace(0, np.nan), axis=0)

    # boolean share of variables (0 = all integer, 1 = all boolean)
    bool_share = (bools / total_vars.replace(0, np.nan))

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    box_style = dict(showfliers=True, patch_artist=True,
                     boxprops=dict(facecolor="#cfe2f3", edgecolor="#225"),
                     medianprops=dict(color="#cc0000", linewidth=2),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4))

    def _boxplot(ax, data_df, title, ylabel, logy=False, hline=None):
        # one box per stage, dropping NaNs
        data = [data_df[s].dropna().values for s in stages]
        ax.boxplot(data, labels=stages, **box_style)
        if logy:
            ax.set_yscale("log")
        if hline is not None:
            ax.axhline(hline, color="grey", linestyle="--", linewidth=1, zorder=0)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=60)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.3)

    # 1) absolute #constraints (log scale handles the spread across instances)
    _boxplot(axes[0, 0], cons,
             "Absolute #constraints per stage",
             "# constraints  (log)", logy=True)

    # 2) absolute #variables
    _boxplot(axes[0, 1], total_vars,
             "Absolute #variables per stage",
             "# variables  (log)", logy=True)

    # 3) constraint blow-up factor
    _boxplot(axes[1, 0], cons_ratio,
             "Constraint blow-up per stage (relative to input)",
             "# constraints / input  (log)", logy=True, hline=1.0)

    # 4) variable blow-up factor
    _boxplot(axes[1, 1], vars_ratio,
             "Variable blow-up per stage (relative to input)",
             "# variables / input  (log)", logy=True, hline=1.0)

    # 5) absolute #constraints trajectory (spaghetti) with median
    ax = axes[2, 0]
    x = range(len(stages))
    for _, row in cons.iterrows():
        ax.plot(x, row.values, color="#888", alpha=0.25, linewidth=1)
    ax.plot(x, cons.median(axis=0).values, color="#cc0000", linewidth=2.5,
            marker="o", label="median")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, rotation=60, ha="right")
    ax.set_title("Absolute #constraints per instance", fontweight="bold")
    ax.set_ylabel("# constraints  (log)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # 6) boolean share of variables
    _boxplot(axes[2, 1], bool_share,
             "Boolean share of variables per stage",
             "# bool / (# bool + # int)", logy=False)
    axes[2, 1].set_ylim(-0.05, 1.05)

    fig.suptitle("PySAT transformation pipeline: effect on model size",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    pdf_path = os.path.join(_HERE, f"{save_basename}.pdf")
    png_path = os.path.join(_HERE, f"{save_basename}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved figures to:\n  {pdf_path}\n  {png_path}")
    return fig


def plot_stats(df, stages=None, save_basename="transformation_absolute"):
    
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save to file
    import matplotlib.pyplot as plt
    import seaborn as sns

    # select stages (preserve the order in which they appear in the columns)
    all_stages = list(dict.fromkeys(df.columns.get_level_values("stage")))
    if stages is None:
        stages = all_stages
    else:
        stages = [s for s in stages if s in all_stages]

    def _to_long(metric, value_name):
        """Tidy (long-form) frame with columns [model, stage, <value_name>]."""
        wide = df.xs(metric, axis=1, level="metric")[stages]
        long = wide.reset_index().melt(
            id_vars="model", var_name="stage", value_name=value_name
        )
        return long

    def _save(fig, suffix):
        pdf_path = os.path.join(_HERE, f"{save_basename}_{suffix}.pdf")
        png_path = os.path.join(_HERE, f"{save_basename}_{suffix}.png")
        fig.tight_layout()
        # fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved figures to:\n  {pdf_path}\n  {png_path}")


    # ---- figure 1: absolute #constraints ----

    fig, ax = plt.subplots(figsize=(10, 6))
    cons_long = _to_long("n_constraints", "value")
    sns.ecdfplot(data=cons_long, x="value", hue = 'stage', hue_order=stages, stat="count", ax=ax)
    ax.set_title("#constraints per stage", fontweight="bold")
    ax.set_xlabel("# constraints  (log)")
    ax.set_xscale("log")
    ax.set_xlim(1, ax.get_xlim()[1])
    ax.set_ylabel("Number of instances")
    _save(fig, "constraints")

    # ---- figure 2: absolute #variables, hue = int/bool ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ints_long = _to_long("n_integer", "value")
    sns.ecdfplot(data=ints_long, x="value", hue = 'stage', hue_order=stages, stat="count", ax=ax)
    ax.set_title("#integer variables per stage", fontweight="bold")
    ax.set_xlabel("# integer variables  (log)")
    ax.set_xscale("log")
    ax.set_xlim(1, ax.get_xlim()[1])
    ax.set_ylabel("Number of instances")
    _save(fig, "variables_int")

    # ---- figure 3: absolute #boolean variables per stage ----
    fig, ax = plt.subplots(figsize=(10, 6))
    bools_long = _to_long("n_boolean", "value")
    sns.ecdfplot(data=bools_long, x="value", hue = 'stage', hue_order=stages, stat="count", ax=ax)
    ax.set_title("#boolean variables per stage", fontweight="bold")
    ax.set_xlabel("# boolean variables  (log)")
    ax.set_xscale("log")
    ax.set_xlim(1, ax.get_xlim()[1])
    ax.set_ylabel("Number of instances")
    _save(fig, "variables_bool")


if __name__ == "__main__":
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)
    # print(df)
    import sys
    if len(sys.argv) > 1:
        models_path = sys.argv[1]
    else:
        models_path = os.path.join(_HERE, "models")

    df = run_experiments(models_path)

    out = os.path.join(_HERE, "transformation_stats.csv")
    df.to_csv(out)
    print(f"\nSaved results to {out}")

    columns = [
        "input",
        "no_partial_functions",
        "push_down_negation",
        "decompose_linear",
        "flatten_constraint",
        "linearize_reified_variables",
        "linearize_constraint",
        "int2bool",
    ]

    df = load_results("transformation_stats.csv")
    # print(df[columns])

    pretty_latex(df[columns])

    # the two plots of interest: absolute #constraints and absolute #variables
    # (split by integer/boolean type)
    plot_stats(df, stages=columns)

    # the older, more detailed overview figure (kept for reference)
    # plot_transformation_stats(df)