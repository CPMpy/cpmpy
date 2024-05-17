import pandas
import pandas as pd
import os
from os.path import join
import glob
import matplotlib.pyplot as plt

#threshold to get notified of instances that became faster or slower
slowthreshold = 0.2
fastthreshold = 0.2
#filename of reference instance, should be in results folder!
refname = 'diamondfree_ortools_2024-05-17 13.49.48.079323.csv'

results = []
cwd = os.getcwd()
if 'y' in cwd[-2:]:
    results.extend(glob.glob(join("benchmarks", 'results', '*.csv')))
    refname = join("benchmarks", 'results', refname)
else:
    results.extend(glob.glob(join('results',"*.csv")))
    refname = join('results', refname)

last = pd.read_csv(results[-1], index_col=0)
print(last)
refinstance = pd.read_csv(refname, index_col=0)
print(refinstance)
#sorted_df = last.loc['total'].sort_values()

#for the last file, plot totals in a pie chart and a bar chart
'''sorted_df.plot(kind='pie', y='total', autopct='%1.0f%%')
plt.figure(figsize=(10, 10))
sorted_df.plot(kind='bar', y='total', stacked=True)'''



#plots all the totals together for comparison
dfs_list = []
for r in results:
    df = pd.read_csv(r, index_col=0)

    dft = df.loc['total']
    dft.name = r[8:-26]
    dfs_list.append(dft)
    percentagedif = ((refinstance - df)/refinstance)
    dfthreshold = percentagedif[percentagedif >= fastthreshold].dropna(how='all')
    dfslowthreshold = percentagedif[percentagedif <= -slowthreshold].dropna(how='all')
    print('\n')
    print('results for ' + r)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    if dfslowthreshold.empty:
        print('no significantly slower instances')
    else:
        print('these instances became slower:')
        print(dfslowthreshold)
    if dfthreshold.empty:
        print('no significantly faster instances')
    else:
        print('these instances became faster:')
        print(dfthreshold)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')

totals = pandas.concat(dfs_list, axis=1)
totals.plot(kind='bar')
plt.show()