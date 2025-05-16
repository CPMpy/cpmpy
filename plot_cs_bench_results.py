

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("xcsp3_stats_cse.csv")
df.set_index("instance", inplace=True)
df.sort_values(by="nb_cons_trans", inplace=True)
df['variant'] = 'cse'
df['instance'] = df.index

print(df)

df2 = pd.read_csv("xcsp3_stats.csv")
df2.set_index("instance", inplace=True)
df2.sort_values(by="nb_cons_trans", inplace=True)
df2['variant'] = 'master'
df2['instance'] = df2.index

print(df2)

df = pd.concat([df, df2], axis=1)

print(df[df['nb_vars_trans'] < df['nb_vars_orig']])


import seaborn as sns

# info for variables

fig, ax = plt.subplots(figsize=(5, 5))

sns.scatterplot(data=df, x="nb_vars_orig", y="nb_vars_trans", hue="variant", ax=ax)
ax.set_title("Number of variables")
ax.set_xlabel("Number of variables before transformation")
ax.set_ylabel("Number of variables after transformation")

lim = max(*ax.get_xlim(), *ax.get_ylim())
ax.plot([0, lim], [0, lim], ls="--", c=".3")
plt.show()


fig, ax = plt.subplots(figsize=(5, 5))

sns.scatterplot(data=df, x="nb_cons_orig", y="nb_cons_trans", hue="variant", ax=ax)
ax.set_title("Number of constraints")
ax.set_xlabel("Number of constraints before transformation")
ax.set_ylabel("Number of constraints after transformation")

lim = max(*ax.get_xlim(), *ax.get_ylim())
ax.plot([0, lim], [0, lim], ls="--", c=".3")
plt.show()

# now compar cse vs master
fig, ax = plt.subplots(figsize=(5, 5))

