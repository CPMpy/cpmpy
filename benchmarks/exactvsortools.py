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
refname = 'quick_exact_2024-05-17 16.02.25.077955.csv'

results = []
cwd = os.getcwd()
if 'y' in cwd[-2:]:
    results.extend(glob.glob(join("benchmarks", 'results', '*.csv')))
    refname = join("benchmarks", 'results', refname)
else:
    results.extend(glob.glob(join('results', "*.csv")))
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
full_list = []
for r in results:
    df = pd.read_csv(r, index_col=0)
    full_list.append(df)
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

for x in dfs_list:
    print(x.name)
print(full_list[0])
dfexact = full_list[0] + full_list[1]
dfortools = full_list[2] + full_list[3]
print(dfexact)

# Create boolean masks
mask1 = dfexact['decompose'] != 140
mask2 = dfexact['solve'] != 0
mask3 = dfortools['decompose'] != 140
mask4 = dfortools['solve'] != 0
mask5 = dfexact['solve'] < 120
mask6 = dfortools['solve'] < 120
exacted = dfexact[mask1&mask2&mask3&mask4&mask5&mask6].sum()
exacted.name = 'exact'
ortooled = dfortools[mask3&mask4&mask1&mask2&mask5&mask6].sum()
ortooled.name = 'ortools'

print(exacted)
print(ortooled)



totals = pandas.concat([exacted,ortooled], axis=1)
totals.plot(kind='bar')

plt.show()
