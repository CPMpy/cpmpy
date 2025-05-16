

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("xcsp3_stats.csv")
df.set_index("instance", inplace=True)
df.sort_values(by="nb_cons_trans", inplace=True)
df['variant'] = 'master'
df['instance'] = df.index
df = df.sort_index()

df_cse = pd.read_csv("xcsp3_stats_cse.csv")
df_cse.set_index("instance", inplace=True)
df_cse.sort_values(by="nb_cons_trans", inplace=True)
df_cse['variant'] = 'cse'
df_cse['instance'] = df_cse.index
df_cse = df_cse.sort_index()

df_all = pd.concat([df, df_cse], axis=0)



import seaborn as sns

# info for variables

fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(x = df['nb_vars_trans'], y= df_cse['nb_vars_trans'])

ax.set_title("Number of variables after transformation (ortools)")
ax.set_xlabel("master")
ax.set_ylabel("cse")

ax.set_xscale("log")
ax.set_yscale("log")

lim = max(*ax.get_xlim(), *ax.get_ylim())
ax.plot([0, lim], [0, lim], ls="--", c=".3")

plt.show()


fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(x = df['nb_cons_trans'], y= df_cse['nb_cons_trans'])

ax.set_title("Number of constraints after transformation (ortools)")
ax.set_xlabel("master")
ax.set_ylabel("cse")

ax.set_xscale("log")
ax.set_yscale("log")

lim = max(*ax.get_xlim(), *ax.get_ylim())
ax.plot([0, lim], [0, lim], ls="--", c=".3")

plt.show()