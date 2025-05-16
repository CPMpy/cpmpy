

import matplotlib.pyplot as plt
import pandas as pd
import os

year = 2023
track = "CSP"

dfs = []
for fname in os.listdir("."):
    if fname.startswith("xcsp3_stats"):
        if "cse" in fname:
            df = pd.read_csv(fname)
            df['variant'] = "cse"
            dfs.append(df)
        else:
            df = pd.read_csv(fname)
            df['variant'] = "master"
            dfs.append(df)


dfs = pd.concat(dfs, axis=0, ignore_index=True)
dfs['full_variant'] = dfs['solver'] + "_" + dfs['variant']

print(dfs[['instance','transform_time','full_variant']][dfs['transform_time'] > 20])

import seaborn as sns

# info for variables

if False:

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.ecdfplot(
        data =dfs, 
        y = "nb_vars_trans",
        hue = "full_variant",
        ax = ax,
        stat="count"
    )

    ax.set_xlim(0,200)
    ax.set_title("Number of variables after transformation")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Number of instances")
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.ecdfplot(
        data =dfs, 
        y = "nb_cons_trans",
        hue = "full_variant",
        ax = ax,
        stat="count"
    )

    ax.set_xlim(0,200)
    ax.set_title("Number of constraints after transformation")
    ax.set_xlabel("Number of instances")
    ax.set_ylabel("Number of constraints")
    plt.show()

    


if False:

    fig, ax = plt.subplots(figsize=(5, 5))

    sns.ecdfplot(
        data =dfs, 
        y = "transform_time",
        hue = "full_variant",
        ax = ax,
        stat="count"
    )

    ax.set_title("Transformation time")
    ax.set_xlabel("Number of instances")
    ax.set_ylabel("Time (s)")

    ax.set_xlim(0,200)
    plt.show()



