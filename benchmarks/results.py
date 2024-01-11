import pandas as pd
import os
from os.path import join
import glob
import matplotlib.pyplot as plt

results = []
cwd = os.getcwd()
if 'y' in cwd[-2:]:
    results.extend(glob.glob(join("benchmarks", 'results', '*.csv')))
else:
    results.extend(glob.glob(join('results',"*.csv")))

last = pd.read_csv(results[-1],index_col=0)
sorted_df = last.loc['total'].sort_values()


sorted_df.plot(kind='pie', y='total', autopct='%1.0f%%')
plt.show()
plt.figure(figsize=(10, 10))
sorted_df.plot(kind='bar', y='total', stacked=True)
plt.show()